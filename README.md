# Actividad5 - Clasificación: Adult (Census Income)

Resumen
- Dataset: Adult (UCI) — clasificación binaria (income >50K)
- Algoritmos comparados: Logistic Regression vs Random Forest
- Seguimiento: MLflow (local por defecto -> ./mlruns) o en Google Drive desde Colab
- Estructura:
  - datos/datos_ini: datos crudos y dataset_version.txt
  - datos/datos_limp: datos limpios
  - fuentes/: scripts y notebooks

Uso en Google Colab (resumen rápido)
1. Abre `fuentes/entrena_colab.ipynb` en Google Colab (usa "Open in Colab" o carga el notebook en Colab).
2. Ejecuta las celdas en orden.
3. (Opcional) Monta Google Drive cuando se solicite para persistir `mlruns`:
   - Si montas Drive, el notebook usará `file:/content/drive/MyDrive/Actividad5-mlruns` como `MLFLOW_TRACKING_URI`.
4. (Opcional) Para ver la UI de MLflow desde Colab, instala `pyngrok`, define `NGROK_AUTH_TOKEN` y ejecuta la celda correspondiente. La celda intentará iniciar `mlflow ui` en el puerto 5000 y exponerlo con ngrok.

Notas
- Si no expones la UI, puedes descargar la carpeta `mlruns` y abrirla localmente con:
  mlflow ui --backend-store-uri ./mlruns --port 5000

Recomendación
- Usa Google Drive si quieres mantener los experimentos entre sesiones de Colab.
