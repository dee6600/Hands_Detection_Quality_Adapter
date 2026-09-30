#!/usr/bin/env python3
"""Download the dataset from Zenodo into ./data and verify SHA-256 checksums.

    python scripts/download_data.py              # annotations only (~5 MB):
                                                 #   enough for tests + every Stage 1-5 number
    python scripts/download_data.py --with-video # + stereo video (~3.9 GB):
                                                 #   needed for Stage 6 and the visualisation scripts

Dataset DOI: 10.5281/zenodo.23056146 (CC BY 4.0).
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

ZENODO_RECORD = "23056146"  # Zenodo record number (the digits after "zenodo." in the DOI)
FILES = {
    "annotations": "hand_adapter_annotations_v1.zip",
    "video": "hand_adapter_videos_v1.zip",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMS = os.path.join(ROOT, "scripts", "data_checksums.sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-video", action="store_true")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    if not ZENODO_RECORD.isdigit():
        sys.exit("ZENODO_RECORD is not set yet in scripts/download_data.py")
    expected = dict(reversed(l.split()) for l in open(SUMS) if l.strip() and not l.startswith("#"))
    os.makedirs(a.data_dir, exist_ok=True)
    for kind in ["annotations"] + (["video"] if a.with_video else []):
        name = FILES[kind]
        dst = os.path.join(a.data_dir, name)
        if not os.path.exists(dst):
            url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
            print("downloading", url)
            urllib.request.urlretrieve(url, dst)
        got = sha256(dst)
        if got != expected[name]:
            sys.exit(f"checksum mismatch for {name}: {got}")
        with zipfile.ZipFile(dst) as z:
            z.extractall(a.data_dir)
        if not a.keep_zip:
            os.remove(dst)
        print("ok", name)
    n = sum(os.path.isfile(os.path.join(a.data_dir, d, "hand_boxes.json")) for d in os.listdir(a.data_dir))
    print(f"{n} clips in {a.data_dir}")


if __name__ == "__main__":
    main()
