import os
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from dvc.repo import Repo


def get_git_commit_id():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def looks_like_dataset_dir(raw_dir):
    if not raw_dir.exists() or not raw_dir.is_dir():
        return False
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    return len(subdirs) >= 2


def try_dvc_pull(target_path):
    if not (
        Path(".dvc").exists() or Path("dvc.yaml").exists() or Path("data.dvc").exists()
    ):
        return False

    try:
        repo = Repo(".")
        repo.pull(targets=[str(target_path)])
        return True
    except Exception:
        return False


def download_from_yandex_public_link(public_url, dst_zip_path):
    api_url = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        f"?public_key={quote(public_url, safe='')}"
    )
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    href = response.json()["href"]

    with requests.get(href, stream=True, timeout=300) as download_response:
        download_response.raise_for_status()
        dst_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_zip_path, "wb") as file_obj:
            for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def ensure_data_available(raw_dir, yandex_public_url):
    if looks_like_dataset_dir(raw_dir):
        return

    pulled = try_dvc_pull(raw_dir)
    if pulled and looks_like_dataset_dir(raw_dir):
        return

    if not yandex_public_url:
        raise FileNotFoundError(
            f"Dataset not found in '{raw_dir}'. "
            "Provide data.yandex_public_url or configure DVC remote."
        )

    tmp_dir = Path("data") / "downloads"
    zip_path = tmp_dir / "dataset.zip"
    download_from_yandex_public_link(yandex_public_url, zip_path)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(raw_dir)

    if not looks_like_dataset_dir(raw_dir):
        nested_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
        if len(nested_dirs) == 1 and looks_like_dataset_dir(nested_dirs[0]):
            nested_root = nested_dirs[0]
            for item in nested_root.iterdir():
                item_dst = raw_dir / item.name
                if item_dst.exists():
                    continue
                os.replace(item, item_dst)

    if not looks_like_dataset_dir(raw_dir):
        raise RuntimeError(
            f"Downloaded data does not look like dataset at '{raw_dir}'. "
            "Expected class subfolders with images."
        )
