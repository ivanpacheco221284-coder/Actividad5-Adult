#!/usr/bin/env python3
"""
scripts/zip_mlruns.py
Comprime la carpeta 'mlruns' en mlruns.zip si existe.
Uso:
  python scripts/zip_mlruns.py --src ./mlruns --out mlruns.zip
"""
import os
import argparse
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            full = os.path.join(root, file)
            arcname = os.path.relpath(full, os.path.dirname(path))
            ziph.write(full, arcname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='mlruns')
    parser.add_argument('--out', type=str, default='mlruns.zip')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print('Source folder not found:', args.src)
        return

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        zipdir(args.src, z)
    print('Created', args.out)

if __name__ == '__main__':
    main()
