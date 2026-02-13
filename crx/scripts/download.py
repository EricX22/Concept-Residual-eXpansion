import argparse
import os
import json
import tarfile
import logging
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    os.makedirs(data_path, exist_ok=True)
    dataset_downloaders = {
        'celeba': download_celeba,
        'waterbirds': download_waterbirds,
    }
    for dataset in datasets:
        if dataset in dataset_downloaders:
            dataset_downloaders[dataset](data_path)
        else:
            no_downloader(dataset)


def no_downloader(dataset):
    print(f"Dataset {dataset} cannot be automatically downloaded. Please check the repo for download instructions.")


def download_waterbirds(data_path):
    logging.info("Downloading Waterbirds...")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar,
    )


def download_celeba(data_path):
    logging.info("Downloading CelebA...")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False
    )



def generate_metadata(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    dataset_metadata_generators = {
        'celeba': generate_metadata_celeba,
        'waterbirds': generate_metadata_waterbirds,
    }
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)


def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA...")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "celeba", "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_waterbirds(data_path):
    logging.info("Generating metadata for Waterbirds...")
    df = pd.read_csv(os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/metadata.csv"))
    df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})

    df[["id", "filename", "split", "y", "a"]].to_csv(
        os.path.join(data_path, "waterbirds", "metadata_waterbirds.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('datasets', nargs='+', type=str, default=[
        'celeba', 'waterbirds'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--download', action='store_true', default=False)
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path, args.datasets)
    generate_metadata(args.data_path, args.datasets)
