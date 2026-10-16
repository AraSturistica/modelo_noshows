# 1. Arquitectura y funcionamiento

El proyecto es una aplicación Gradio de predicción de riesgo de inasistencia a citas (no-show). En la carpeta están app.py, los metadatos de variables, el modelo XGBoost guardado y requirements.txt. No encontré módulos Python adicionales ni archivos de pruebas en la estructura visible.
Flujo principal:

1. Al importar app.py, carga model_features.json, crea un XGBClassifier, carga xgb_noshow.json y prepara un explicador SHAP (app.py:27–41).
2. La interfaz recibe datos de una cita y crea una fila con las 20 variables esperadas por el modelo. La lista de columnas y las categorías de barrio están en model_features.json.
3. predict() calcula la probabilidad de inasistencia con predict_proba, obtiene contribuciones SHAP y devuelve una tarjeta, un gráfico y un contexto para las funciones de IA ([app.py:241–261]).
4. La aplicación ofrece resumen para personal, recordatorio para pacientes y chat. Si no hay HF_TOKEN, usa plantillas o informa que el chat no está habilitado; si falla el servicio, hay rutas alternativas ([app.py:286–334, 351–380]).
5. Al final, demo.launch() inicia la interfaz ([app.py:630–631]).

# 2. Modelos y componentes

| Componente | Uso |
| :--- | :--- |
| **XGBoost (`XGBClassifier`)** | Modelo predictivo cargado desde `xgb_noshow.json`. |
| **SHAP (`TreeExplainer`)** | Explica las contribuciones de variables a cada predicción. |
| **Regresión logística** | Se reporta como modelo de comparación en el README, pero no se carga ni se ejecuta en `app.py`. |
| **Hugging Face `InferenceClient`** | Llama al modelo LLM configurado por `LLM_MODEL` para resumen, recordatorios y chat. |
| **Gradio** | Construye la interfaz y conecta los controles con las funciones. |
| **pandas / NumPy / Matplotlib** | Preparación de la fila, representación de datos y gráfico. `numpy` se importa, pero no observé que se use en el código revisado. |


