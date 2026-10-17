#!/usr/bin/env python3
"""
datos_prep.py
- Descarga el dataset Adult desde UCI.
- Limpia valores faltantes y formatea columnas.
- Guarda raw en datos/datos_ini/adult.data (CSV) y limpio en datos/datos_limp/clean_adult.csv
- Calcula sha256 del archivo raw y lo guarda en datos/datos_ini/dataset_version.txt
"""
import os
import hashlib
import pandas as pd

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMN_NAMES = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]

RAW_DIR = "datos/datos_ini"
CLEAN_DIR = "datos/datos_limp"

def download_raw(out_path=os.path.join(RAW_DIR, "adult.data")):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Descargando {ADULT_URL} -> {out_path}")
    df = pd.read_csv(ADULT_URL, header=None, names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True)
    df.to_csv(out_path, index=False)
    print(f"Raw data guardado en {out_path}")
    sha = compute_sha256(out_path)
    version_path = os.path.join(RAW_DIR, "dataset_version.txt")
    with open(version_path, "w") as f:
        f.write(sha + "\n")
    print(f"SHA256 dataset: {sha} (guardado en {version_path})")
    return out_path


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_adult(raw_path=os.path.join(RAW_DIR, "adult.data"),
                out_path=os.path.join(CLEAN_DIR, "clean_adult.csv")):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Cargando raw desde {raw_path}")
    df = pd.read_csv(raw_path, na_values=" ?", skipinitialspace=True)
    # Drop filas con missing
    df = df.dropna()
    # Strip whitespace en columnas object
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    # Map target a 0/1
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df.to_csv(out_path, index=False)
    print(f"Datos limpios guardados en {out_path} (shape={df.shape})")
    return out_path


if __name__ == "__main__":
    raw = download_raw()
    clean = clean_adult(raw)
