"""Download public chest CT scans to try in the web app, as .zip files ready to upload.

The defaults come from collections the models never saw during training (The Cancer Imaging Archive):
  - SPIE-AAPM Lung CT Challenge (LUNGx): pathology-confirmed nodules. The patient ID tells the truth:
    "LC" = lung cancer, "BE" = benign.
  - LungCT-Diagnosis: patients with lung adenocarcinoma.

    python -m cadc.get_samples                       # the three default scans (~400 MB)
    python -m cadc.get_samples --list "SPIE-AAPM Lung CT Challenge"
    python -m cadc.get_samples --collection "SPIE-AAPM Lung CT Challenge" --patients CT-Training-LC002 CT-Training-BE001
"""
import argparse
from pathlib import Path

import requests

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULTS = [
    ("SPIE-AAPM Lung CT Challenge", "CT-Training-LC001"),   # confirmed lung cancer
    ("SPIE-AAPM Lung CT Challenge", "CT-Training-BE002"),   # confirmed benign
    ("LungCT-Diagnosis", "R_004"),                          # adenocarcinoma, small download
]


def series_for(collection):
    r = requests.get(f"{API}/getSeries", params={"Collection": collection, "Modality": "CT"}, timeout=60)
    r.raise_for_status()
    return r.json()


def download(series_uid, dest):
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(f"{API}/getImage", params={"SeriesInstanceUID": series_uid}, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    parser.add_argument("--collection", help="TCIA collection name")
    parser.add_argument("--patients", nargs="+", help="patient IDs in --collection")
    parser.add_argument("--list", metavar="COLLECTION", help="list the CT scans in a collection and exit")
    args = parser.parse_args()

    if args.list:
        for s in series_for(args.list):
            print(f"{s['PatientID']:<24} {s.get('ImageCount', '?'):>4} slices  {s.get('FileSize', 0) / 1e6:6.0f} MB  "
                  f"{s.get('SeriesDescription', '')}")
        return

    wanted = [(args.collection, p) for p in args.patients] if args.collection and args.patients else DEFAULTS
    args.out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for collection, patient in wanted:
        if collection not in cache:
            cache[collection] = series_for(collection)
        matches = [s for s in cache[collection] if s["PatientID"] == patient]
        if not matches:
            print(f"{patient}: not found in {collection}")
            continue
        s = max(matches, key=lambda x: x.get("ImageCount", 0))  # the main (largest) CT series
        dest = args.out / f"{patient}.zip"
        if dest.exists():
            print(f"{dest} already exists")
            continue
        print(f"downloading {patient} ({s.get('ImageCount', '?')} slices, {s.get('FileSize', 0) / 1e6:.0f} MB) ...", flush=True)
        download(s["SeriesInstanceUID"], dest)
        print(f"  saved {dest}")
    print(f"\nUpload any of these .zip files on the Analyse page: {args.out.resolve()}")


if __name__ == "__main__":
    main()
