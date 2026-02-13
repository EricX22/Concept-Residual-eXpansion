#!/usr/bin/env python3
import argparse
import json
import os
from typing import List

CHANNEL_USE_DEFAULT = {
    "target_attributes": True,
    "environment_context": True,
    "cooccurring_objects": True,
    "imaging_artifacts": False,     # default exclude (audit-only)
    "hard_negatives": True,
    "sensitive_proxies": False      # always exclude (diagnostic-only)
}

CHANNEL_ORDER = [
    "target_attributes",
    "environment_context",
    "cooccurring_objects",
    "imaging_artifacts",
    "hard_negatives",
    "sensitive_proxies",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="concept bank JSON")
    ap.add_argument("--out", required=True, help="meta JSON output")
    ap.add_argument("--include-artifacts", action="store_true", help="include imaging_artifacts in training")
    args = ap.parse_args()

    with open(args.bank, "r", encoding="utf-8") as f:
        bank = json.load(f)

    channels_by_name = {c["name"]: c for c in bank["channels"]}

    concept_text: List[str] = []
    channel_name: List[str] = []
    safety: List[str] = []
    use_in_training: List[bool] = []

    seen = set()

    for ch_name in CHANNEL_ORDER:
        ch = channels_by_name.get(ch_name, {"concepts": []})
        for item in ch.get("concepts", []):
            t = item["text"].strip()
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)

            s = item["safety"]  # required

            concept_text.append(t)
            channel_name.append(ch_name)
            safety.append(s)

            use = CHANNEL_USE_DEFAULT.get(ch_name, True)
            if ch_name == "imaging_artifacts" and args.include_artifacts:
                use = True

            # force-exclude sensitive
            if ch_name == "sensitive_proxies" or s == "sensitive_proxy":
                use = False

            use_in_training.append(bool(use))

    meta = {
        "schema_version": "concept_meta_v1",
        "task": bank["task"],
        "concept_text": concept_text,
        "channel": channel_name,
        "safety": safety,
        "use_in_training": use_in_training,
        "K": len(concept_text),
        "notes": {"include_imaging_artifacts": bool(args.include_artifacts)},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote concept meta -> {args.out} (K={meta['K']})")

if __name__ == "__main__":
    main()
