"""Download BCI Competition IV data set 4 into data/raw/ and check it.

    python scripts/get_data.py --agree     # download, unzip, verify (~220 MB)
    python scripts/get_data.py --check     # only verify what is in data/raw/

The data are not redistributed in this repository. They come from the BCI
Competition IV website (https://www.bbci.de/competition/iv/), whose terms ask
you to cite the paper named in the data set description, Schalk et al. 2007,
J. Neural Eng. 4:264-275, and to reference the group that recorded the data
(K. J. Miller and G. Schalk). --agree confirms that you accept those terms.

If the download fails (proxy, firewall, site down), get the two zips in a
browser:
    https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip
    https://bbci.de/competition/iv/results/ds4/true_labels.zip
put them in data/raw/ and run this script again with --agree: it unzips any
zip it finds there instead of downloading it.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw")

ZIPS = {
    "BCICIV_4_mat.zip": "https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip",
    "true_labels.zip": "https://bbci.de/competition/iv/results/ds4/true_labels.zip",
}
# SHA-256 of the files the paper's numbers were computed from.
FILES = {
    "sub1_comp.mat": "0f479b5b0317bb4789ff7687de388382ef83d4c1eb360aac5fec0ef54ef23cf5",
    "sub2_comp.mat": "a62643ea293f1bf7b167286b736811a72ea20ea89dbe3dfcd3b94db4869381a6",
    "sub3_comp.mat": "f8c30847744e664922de6dd0c5560bb85fb0a07fcaee84b139e0b075762d99bb",
    "sub1_testlabels.mat": "f4dd37941f0d878e87d9d7297ada0493b5a44874dbaa659472d048d4b7555dea",
    "sub2_testlabels.mat": "fb902a3b63f7f53b41d6d548bbdd00f2164a239bd4e205e82042e3ffa3e3e8a4",
    "sub3_testlabels.mat": "8216229b86f04e9b25648a11f431a735fabc777e7e308c0445e5c391525c4897",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dst):
    part = dst + ".part"

    def hook(n, block, total):
        if total > 0:
            done = min(n * block, total)
            print(f"\r  {os.path.basename(dst)}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB",
                  end="", flush=True)

    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, part, hook)
    print()
    os.replace(part, dst)


def extract(zpath, dest):
    """Copy every expected .mat out of the zip into dest, whatever folder it
    sits in inside the archive. Returns the names it found."""
    found = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or name.lower() not in FILES:
                continue
            with z.open(info) as src, open(os.path.join(dest, name.lower()), "wb") as out:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            found.append(name.lower())
        if not found:
            print(f"  {os.path.basename(zpath)} has none of the expected files; it contains:")
            for n in z.namelist()[:20]:
                print(f"    {n}")
    return found


def check(raw=RAW):
    """Print one line per expected file. True if every file is present."""
    ok = True
    for name, want in FILES.items():
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            print(f"  MISSING   {name}")
            ok = False
        elif sha256(path) == want:
            print(f"  ok        {name}")
        else:
            print(f"  DIFFERENT {name}: present, but not the copy the paper used "
                  "(numbers may not match exactly)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agree", action="store_true",
                    help="accept the BCI Competition IV terms (cite the data set paper)")
    ap.add_argument("--check", action="store_true", help="only verify data/raw/")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--raw", default=RAW, help="where the .mat files go")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check(a.raw) else 1)
    if not a.agree:
        print(__doc__)
        print("Re-run with --agree to download.")
        sys.exit(2)

    os.makedirs(a.raw, exist_ok=True)
    for zname, url in ZIPS.items():
        zpath = os.path.join(a.raw, zname)
        if not os.path.exists(zpath):
            try:
                download(url, zpath)
            except OSError as exc:
                print(f"\n  could not download {zname}: {exc}\n"
                      f"  Fetch it in a browser from {url}, put it in {a.raw} and run "
                      "this script again.")
                continue
        found = extract(zpath, a.raw)
        print(f"  {zname}: extracted {', '.join(sorted(found)) or 'nothing'}")
        if found and not a.keep_zips:
            os.remove(zpath)

    print("\nchecking data/raw:")
    if check(a.raw):
        print("\nAll six files present. Next: python decoder/prep.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
