# CHANGELOG

## 2026-10-17 - initial
- Proyecto inicial creado.
- Dataset: Adult (UCI).
- Scripts:
  - fuentes/datos_prep.py: descarga, limpieza y versionado (sha256).
  - fuentes/train.py: entrenamiento (Logistic Regression, Random Forest), GridSearchCV, registro en MLflow.
  - fuentes/entrena.ipynb: notebook de ejemplo para Colab/Jupyter.
- README y requirements agregados.
- Nota: ejecutar `python fuentes/datos_prep.py` para descargar y versionar el dataset.
