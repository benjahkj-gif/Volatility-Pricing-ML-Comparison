# Benchmarking de Volatilidad Implícita: Machine Learning vs Black-Scholes

## Descripción del Proyecto
Este proyecto desarrolla un entorno de validación (*benchmarking*) para comparar la capacidad predictiva de múltiples algoritmos de Machine Learning frente a la teoría clásica de fijación de precios de Black-Scholes. El objetivo es modelar la asimetría del riesgo de mercado (la "Sonrisa de Volatilidad") utilizando opciones financieras reales de Microsoft (MSFT).

El sistema extrae datos en vivo, aplica ingeniería de características espaciotemporales (Moneyness y decaimiento temporal) y evalúa tres arquitecturas distintas para corregir las ineficiencias del modelo analítico tradicional.

## Estructura de Datos (Resultados)
El dashboard se alimenta de los siguientes archivos estáticos generados por el motor de entrenamiento:

* `opciones_msft_procesado.csv`: Matriz de características depurada con las coordenadas geométricas de cada contrato ($K/S_t$) y la volatilidad real del mercado.
* `meta_descarga.json`: Metadatos de la extracción temporal (fecha de ejecución, precio spot del subyacente y horizonte temporal $T$).
* `resultados_rf.json`: Métricas de error (MAE, R2) y predicciones del modelo Random Forest.
* `resultados_svm.json`: Métricas de error y predicciones del modelo Support Vector Machine (SVR).
* `resultados_xgb.json`: Métricas de error y predicciones del modelo XGBoost.

## Requisitos y Dependencias
Para ejecutar el dashboard interactivo (`dashboard.py`) y reproducir la visualización de los modelos, asegúrate de tener instaladas las siguientes librerías en tu entorno virtual:

### Procesamiento y Matemática Financiera
* `pandas`: Manipulación e ingesta del archivo CSV.
* `numpy`: Operaciones vectoriales y transformaciones logarítmicas.
* `scipy`: Ecuaciones estadísticas para la validación analítica de Black-Scholes.

### Motores de Machine Learning
* `scikit-learn`: Requerido para procesar las métricas de error y cargar la estructura de la SVM y el Random Forest.
* `xgboost`: Algoritmo de *Gradient Boosting* optimizado para el benchmarking.

### Interfaz y Visualización (Dashboard)
* `streamlit`: Framework principal para renderizar la aplicación web interactiva (si estás usando Streamlit).
* `plotly` o `matplotlib`: Generación de gráficos interactivos de la Sonrisa de Volatilidad.

## Instrucciones de Instalación
Puedes instalar todas las dependencias requeridas ejecutando el siguiente comando en tu terminal:

```bash
pip install pandas numpy scipy scikit-learn xgboost streamlit plotly matplotlib
