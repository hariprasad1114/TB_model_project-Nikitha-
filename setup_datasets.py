#!/usr/bin/env python3
"""Download and lay out the three CXR datasets this repo expects.

Folder names are hardcoded in cxr_ood/utils/cxr_dataset.py,
cxr_ood/evaluation/ood_detection.py and dashboard/backend/inference.py, so this
script only ever produces this exact structure (note Shenzhen's doubled
images/, and the space in the OOD folder names):

  <data-root>/TB_Chest_Radiography_Database/Normal/*.png
  <data-root>/TB_Chest_Radiography_Database/Tuberculosis/*.png
  <data-root>/Montgomery TB CXR/images/*.png
  <data-root>/Shenzhen TB CXR/images/images/*.png

Sources:
  TB database -> Kaggle (needs ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY)
  Montgomery  -> https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
  Shenzhen    -> https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

Stdlib only. Idempotent: a dataset whose target folders already hold images is
skipped, so re-running after a failure only redoes what is missing. Nothing
existing is deleted; --force re-extracts from the cached zip.

  python setup_datasets.py                              # all three
  python setup_datasets.py --datasets montgomery,shenzhen
  python setup_datasets.py --tb-zip ~/Downloads/TB_Chest_Radiography_Database.zip
  python setup_datasets.py --montgomery-url <zip-url> --shenzhen-url <zip-url>
  python setup_datasets.py --data-root Datasets/CXR --keep-zips

Datasets/ is gitignored. The OOD sets are only needed for evaluation stages;
no training stage ever reads them.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SOURCES = {
    "montgomery": "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    "shenzhen": "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
}
KAGGLE_SLUG = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


class Skip(Exception):
    """Dataset cannot be fetched here; report and keep going with the others."""


def count_images(path):
    p = Path(path)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix in IMG_EXTS)


def download_zip(url, dest, force=False):
    """Fetch a zip to dest and verify it really is one.

    The historical openi.nlm.nih.gov paths answer with HTTP 200 and a small
    "There seems to be an error" HTML page, so status codes alone prove nothing:
    the payload is checked for the PK signature before it is accepted.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        with open(dest, "rb") as f:
            if f.read(4) == b"PK\x03\x04":
                print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
                return dest
            print(f"  cached file is not a zip, re-downloading: {dest.name}")
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")  # stale partial from a killed run
    if tmp.exists():
        tmp.unlink()
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            ctype = (r.headers.get("Content-Type") or "").lower()
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 1e6:6.1f}/{total / 1e6:.1f} MB")
                    sys.stdout.flush()
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise Skip(f"download failed: {type(e).__name__}: {e} ({url})")
    size = tmp.stat().st_size
    with open(tmp, "rb") as f:
        magic = f.read(4)
    if magic != b"PK\x03\x04":
        tmp.unlink(missing_ok=True)
        raise Skip(
            f"{url} returned {ctype or 'an unknown content type'} ({size} bytes), "
            "not a zip - the mirror has moved or is serving an error page. \n"
            "          Pass a working mirror with the URL override, or fetch it "
            "manually and unzip into the data root. See 'Where to get them' in "
            "README.md; the Hugging Face mirror "
            "Famatsu123/montgomery-shenzhen-tuberculosis-cxr repackages both sets.")
    print(f"\r  saved {dest.name} ({size / 1e6:.1f} MB)")
    tmp.replace(dest)
    return dest


def safe_extract(zip_path, dest):
    """Extract a zip, refusing absolute or parent-escaping member paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Refusing unsafe path in {zip_path.name}: {member}")
        z.extractall(dest)
    return dest


def find_dir(root, name):
    """First directory named `name` (case-insensitive) under root."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def copy_images(src, dst, force=False):
    """Copy image files from src into dst (flat). Returns number copied."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in Path(src).iterdir():
        if not f.is_file() or f.suffix not in IMG_EXTS:
            continue
        out = dst / f.name
        if out.exists() and not force:
            n += 1
            continue
        shutil.copy2(f, out)
        n += 1
    return n


def kaggle_download(slug, out_dir):
    """Fetch the Kaggle dataset zip via the CLI, else the python package."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have_creds = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not have_creds:
        raise Skip(
            "no Kaggle credentials. Put your API token at "
            "~/.kaggle/kaggle.json (Kaggle > Account > Create New API Token), "
            "or set KAGGLE_USERNAME and KAGGLE_KEY, then re-run. "
            f"Manual alternative: download '{slug}' and unzip it into "
            f"{out_dir}.")

    if shutil.which("kaggle"):
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)]
        if os.name != "nt":
            cmd.append("--quiet")
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise Skip("neither the kaggle CLI nor the 'kaggle' python package "
                       "is installed. pip install kaggle, or download the "
                       f"dataset manually into {out_dir}.")
        print(f"  kaggle python API: {slug}")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(out_dir), unzip=False)

    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise Skip(f"kaggle produced no zip in {out_dir}")
    return zips[-1]


def setup_tb(data_root, cache, force, tb_zip=None):
    target = data_root / "TB_Chest_Radiography_Database"
    if count_images(target / "Normal") and count_images(target / "Tuberculosis") \
            and not force:
        print(f"  already laid out: {target}")
        return True
    if tb_zip:
        zip_path = Path(tb_zip)
        if not zip_path.is_file():
            raise Skip(f"--tb-zip not found: {zip_path}")
        print(f"  using local zip: {zip_path}")
    else:
        zip_path = kaggle_download(KAGGLE_SLUG, cache / "tb")
    stage = safe_extract(zip_path, cache / "tb" / "extracted")
    holder = next((d for d in [stage, *stage.rglob("*")] if d.is_dir()
                   and (d / "Normal").is_dir() and (d / "Tuberculosis").is_dir()),
                  None)
    if holder is None:
        raise Skip(f"no folder containing Normal/ and Tuberculosis/ inside "
                   f"{zip_path.name}; extract it manually and check the layout.")
    for cls in ("Normal", "Tuberculosis"):
        n = copy_images(holder / cls, target / cls, force=force)
        print(f"  {cls}: {n} images -> {target / cls}")
    return True


def setup_ood(name, url, data_root, cache, force):
    """Montgomery -> 'Montgomery TB CXR/images/'; Shenzhen -> '.../images/images/'."""
    folder = {"montgomery": "Montgomery TB CXR", "shenzhen": "Shenzhen TB CXR"}[name]
    images_dir = data_root / folder / "images"
    if name == "shenzhen":
        images_dir = images_dir / "images"
    if count_images(images_dir) and not force:
        print(f"  already laid out: {images_dir}")
        return True
    zip_path = download_zip(url, cache / f"{name}.zip")
    stage = safe_extract(zip_path, cache / f"{name}" / "extracted")
    src = find_dir(stage, "CXR_png")  # ChinaSet_AllFiles/CXR_png, MontgomerySet/CXR_png
    if src is None or not count_images(src):
        raise Skip(f"no CXR_png image folder inside {zip_path.name}")
    n = copy_images(src, images_dir, force=force)
    print(f"  {n} images -> {images_dir}")
    return True


def report(data_root):
    print("\n" + "=" * 72)
    print(f"{'dataset':<34} | {'folder':<40} | images")
    print("-" * 72)
    rows = [
        ("TB Chest Radiography Database (ID)", "TB_Chest_Radiography_Database/Normal"),
        ("TB Chest Radiography Database (ID)", "TB_Chest_Radiography_Database/Tuberculosis"),
        ("Montgomery (OOD, eval only)", "Montgomery TB CXR/images"),
        ("Shenzhen (OOD, eval only)", "Shenzhen TB CXR/images/images"),
    ]
    empty = []
    for label, rel in rows:
        n = count_images(data_root / rel)
        print(f"{label:<34} | {rel:<40} | {n}")
        if not n:
            empty.append(rel)
    print("=" * 72)
    if empty:
        print("Empty (scripts report 0 samples silently without these):")
        for rel in empty:
            print(f"  {rel}")
    return not empty


def main():
    ap = argparse.ArgumentParser(
        description="Download + lay out TB/Montgomery/Shenzhen CXR datasets.")
    ap.add_argument("--data-root", default="Datasets/CXR",
                    help="destination root the code reads from (default: %(default)s)")
    ap.add_argument("--cache-dir", default="Datasets/_downloads",
                    help="where zips and extraction staging live (default: %(default)s)")
    ap.add_argument("--datasets", default="tb,montgomery,shenzhen",
                    help="comma-separated subset of: tb, montgomery, shenzhen")
    ap.add_argument("--force", action="store_true",
                    help="re-copy even when the target already has images")
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep downloaded zips (they are ~700 MB for TB)")
    ap.add_argument("--tb-zip", default=None,
                    help="already-downloaded Kaggle TB zip (skips the Kaggle API)")
    ap.add_argument("--montgomery-url", default=SOURCES["montgomery"],
                    help="override the Montgomery zip URL (mirrors move)")
    ap.add_argument("--shenzhen-url", default=SOURCES["shenzhen"],
                    help="override the Shenzhen zip URL (mirrors move)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    cache = Path(args.cache_dir)
    if not cache.is_absolute():
        cache = REPO_ROOT / cache

    wanted = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in ("tb", "montgomery", "shenzhen")]
    if unknown:
        ap.error(f"unknown --datasets entries: {unknown}")

    print(f"data-root={data_root}\ncache-dir={cache}\ndatasets={wanted}")
    failed = []
    for name in wanted:
        print(f"\n=== [{name}]")
        try:
            urls = {"montgomery": args.montgomery_url, "shenzhen": args.shenzhen_url}
            if name == "tb":
                ok = setup_tb(data_root, cache, args.force, args.tb_zip)
            else:
                ok = setup_ood(name, urls[name], data_root, cache, args.force)
            if not ok:
                failed.append(name)
        except Skip as e:
            print(f"  SKIP: {e}")
            failed.append(name)
        except Exception as e:  # network, zip, permission - keep the other sets going
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    if not args.keep_zips:
        for z in cache.glob("*.zip"):
            z.unlink()
        tb_zips = cache / "tb"
        for z in tb_zips.glob("*.zip") if tb_zips.is_dir() else []:
            z.unlink()

    complete = report(data_root)
    if failed:
        print(f"\nNot set up: {failed}")
        return 1
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
