#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---- If you use OpenAI:
# pip install openai
from openai import OpenAI


ALLOWED_CHANNELS = [
    "target_attributes",
    "environment_context",
    "cooccurring_objects",
    "imaging_artifacts",
    "hard_negatives",
    "sensitive_proxies",
]

ALLOWED_SAFETY = ["ok", "artifact", "sensitive_proxy"]

SCHEMA_VERSION = "concept_bank_v1"


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strip_code_fences(s: str) -> str:
    # Some models wrap JSON in ```json ... ```
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _basic_validate_concept_bank(obj: Dict[str, Any]) -> None:
    assert obj.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch"
    assert "task" in obj and "channels" in obj, "missing task/channels"
    ch = obj["channels"]
    assert isinstance(ch, list) and len(ch) > 0, "channels must be list"

    names = [c.get("name") for c in ch]
    for req in ALLOWED_CHANNELS:
        assert req in names, f"missing channel: {req}"

    for c in ch:
        name = c.get("name")
        assert name in ALLOWED_CHANNELS, f"unknown channel: {name}"
        concepts = c.get("concepts", [])
        assert isinstance(concepts, list), f"concepts must be list for {name}"
        for item in concepts:
            assert "text" in item and isinstance(item["text"], str)
            assert 1 <= len(item["text"].split()) <= 10, f"concept too long/short: {item['text']}"
            if "safety" in item:
                assert item["safety"] in ALLOWED_SAFETY, f"bad safety tag: {item['safety']}"


def build_prompt(task_id: str, modality: str, labels: List[Dict[str, str]]) -> Dict[str, str]:
    schema_skeleton = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "task_id": task_id,
            "modality": modality,
            "labels": [{"id": i, "name": lab["name"], "definition": lab.get("definition", "")} for i, lab in enumerate(labels)]
        },
        "channels": [
            {"name": "target_attributes", "concepts": []},
            {"name": "environment_context", "concepts": []},
            {"name": "cooccurring_objects", "concepts": [], "optional": True},
            {"name": "imaging_artifacts", "concepts": [], "optional": True},
            {"name": "hard_negatives", "concepts": [], "optional": True},
            {"name": "sensitive_proxies", "concepts": [], "policy": "diagnostic_only"},
        ],
        "generation": {
            "generator": "llm",
            "model": "",
            "timestamp_utc": "",
            "notes": ""
        }
    }

    system = (
        "You generate a concept vocabulary for a vision classifier to improve interpretability and robustness.\n"
        "All concepts MUST be visually detectable from a single still image (no audio, no motion/temporal info).\n"
        "Concepts must be concrete and literal (objects, textures, colors, materials, shapes, scene elements).\n"
        "Avoid abstract/semantic knowledge that is not directly visible.\n"
        "Avoid dataset-specific quirks (no dataset names, no split names, no file-format artifacts).\n"
        "Output valid JSON ONLY that matches the provided schema exactly, with no extra keys.\n"
    )


    user = (
        "Fill the JSON schema below by adding concepts to each channel.\n\n"
        "GLOBAL RULES (strict):\n"
        "1) Single-image only: the concept must be decidable from ONE still image.\n"
        "   - Forbidden: sounds/calls, migration, seasonality, typical behavior over time.\n"
        "2) Concrete wording: prefer nouns/adjectives describing visible things.\n"
        "   - Good: 'webbed feet', 'open water', 'tree canopy', 'sand shoreline', 'snow', 'motion blur'.\n"
        "   - Bad: 'migratory patterns', 'distinctive calls', 'aquatic features', 'perching behavior'.\n"
        "3) No umbrella terms: avoid vague categories like 'features', 'patterns' unless specific (e.g., 'striped plumage').\n"
        "4) CLIP-friendly: use short literal phrases (1–6 words) that a CLIP-like model can score.\n"
        "5) Diversity: avoid near-duplicates (e.g., don't include both 'colorful feathers' and 'bright plumage' unless meaningfully different).\n\n"
        "Channel guidelines:\n"
        "- target_attributes: visible parts/attributes of the subject itself (shape, beak, legs, plumage, silhouette).\n"
        "- environment_context: visible background/scene elements (water, shoreline, forest, reeds, sky, rocks).\n"
        "- cooccurring_objects: other visible objects often near the subject (boats, fish, insects, other birds, driftwood).\n"
        "- imaging_artifacts: acquisition/presentation artifacts (blur, overexposure, vignetting, JPEG artifacts, borders).\n"
        "- hard_negatives: visually similar categories that can be confused in a single photo (duck, gull, heron, sparrow, pigeon).\n"
        "- sensitive_proxies: diagnostic-only; add 0–8 only if broadly applicable; tag as sensitive_proxy.\n\n"
        "Counts:\n"
        "- target_attributes: 15–25\n"
        "- environment_context: 15–25\n"
        "- cooccurring_objects: 10–20\n"
        "- imaging_artifacts: 8–15\n"
        "- hard_negatives: 8–15\n"
        "- sensitive_proxies: 0–8\n\n"
        "For each concept item, include:\n"
        "- text (1–6 words)\n"
        "- safety: ok | artifact | sensitive_proxy\n\n"
        "FINAL CHECK before outputting JSON:\n"
        "- Remove any concept that requires time, sound, or non-visual inference.\n"
        "- Remove any vague 'X features'/'X patterns' phrase.\n"
        "- Remove duplicates/synonyms.\n\n"
        "Schema to fill (concept arrays are currently empty):\n"
        + json.dumps(schema_skeleton, indent=2)
    )

    return {"system": system, "user": user}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True, help="e.g., waterbirds_binary")
    ap.add_argument("--modality", required=True, choices=["natural_image", "medical_xray", "satellite", "document_image", "other"])
    ap.add_argument("--labels", required=True, help='JSON list, e.g. \'[{"name":"landbird"},{"name":"waterbird"}]\'')
    ap.add_argument("--out", required=True, help="output JSON path, e.g. artifacts/concepts/waterbirds_binary.json")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    labels = json.loads(args.labels)
    assert isinstance(labels, list) and all("name" in x for x in labels)

    prompt = build_prompt(args.task_id, args.modality, labels)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if client.api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set")

    resp = client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
    )

    text = resp.choices[0].message.content
    text = _strip_code_fences(text)

    obj = json.loads(text)
    # fill generation metadata
    obj.setdefault("generation", {})
    obj["generation"]["generator"] = "llm"
    obj["generation"]["model"] = args.model
    obj["generation"]["timestamp_utc"] = _now_utc_iso()

    _basic_validate_concept_bank(obj)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote concept bank -> {args.out}")


if __name__ == "__main__":
    main()
