"""Scarica da GitHub il codice di YuE DOPO la PR #271 (commit 8ed3f25 su main) in lab_misure/yue_pr271/src.
Non installa niente: la cartella serve solo a misura_pr271.py. Stampa l'impronta dei sorgenti per il registro."""
import hashlib, io, sys, urllib.request, zipfile
from pathlib import Path

COMMIT = "8ed3f25"
URL = f"https://codeload.github.com/multimodal-art-projection/YuE/zip/{COMMIT}"
DEST = Path(__file__).resolve().parent / "yue_pr271"


def main():
    if (DEST / "src" / "yue2" / "placement.py").exists():
        print(f"gia' presente: {DEST}")
    else:
        print(f"scarico {URL} ...", flush=True)
        data = urllib.request.urlopen(URL, timeout=120).read()
        print(f"  {len(data) / 1e6:.1f} MB", flush=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                parts = n.split("/", 1)
                if len(parts) < 2 or not parts[1].startswith("src/yue2/") or n.endswith("/"):
                    continue
                out = DEST / parts[1]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(n))
    h = hashlib.sha256()
    files = sorted((DEST / "src" / "yue2").glob("*.py"))
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    print(f"codice della #271 pronto: {len(files)} file .py | SHA-256 sorgenti {h.hexdigest()}")
    print("contiene placement.py (la novita' --low-vram):", (DEST / "src" / "yue2" / "placement.py").exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())
