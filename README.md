# Actividad5 - Clasificación: Adult (Census Income)

Resumen
- Dataset: Adult (UCI) — clasificación binaria (income >50K)
- Algoritmos comparados: Logistic Regression vs Random Forest
- Seguimiento: MLflow (local por defecto -> ./mlruns)
- Estructura:
  - datos/datos_ini: datos crudos y dataset_version.txt
  - datos/datos_limp: datos limpios
  - fuentes/: scripts y notebook

Requisitos
- Python 3.8+
- Instalar dependencias:
  pip install -r requirements.txt

Cómo reproducir (rápido)
1. Preparar datos:
   python fuentes/datos_prep.py
   Esto descargará adult.data, generará el sha256 en datos/datos_ini/dataset_version.txt y producirá datos/datos_limp/clean_adult.csv

2. Iniciar MLflow UI (opcional):
   mlflow ui --backend-store-uri ./mlruns --port 5000
   Abrir: http://localhost:5000

3. Entrenar y registrar experimentos:
   python fuentes/train.py --model logreg
   python fuentes/train.py --model rf

4. Revisar experimentos en MLflow. Cada run registra:
   - Parámetros: modelo, hiperparámetros (best), dataset_sha256
   - Métricas: accuracy, precision, recall, f1, roc_auc (si aplica)
   - Artefactos: modelo (pkl) y confusion matrix (png)

Notas sobre versionado de datos
- Se genera datos/datos_ini/dataset_version.txt con el SHA256 del CSV raw para trazabilidad.
- Si deseas versionar dataset en Git, el archivo adult.data (~3 MB) es razonable para commitear.
- Para datasets grandes, considera DVC o un bucket remoto.

Estructura del repo
Actividad5/
│── datos/
│   ├── datos_ini/
│   └── datos_limp/
│── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
│── README.md
│── CHANGELOG.md
│── requirements.txt
│── .gitignore

Contacto
- Autor: (Pablo Ivan Pacheco)
