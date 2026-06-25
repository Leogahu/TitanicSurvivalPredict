# Titanic Survival Predictor

Pipeline completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic. Este proyecto incluye un dashboard interactivo para probar predicciones en tiempo real.

## Tabla de Contenidos

- [Resultados del Modelo](#resultados-del-modelo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Cómo Usar](#cómo-usar)
- [Tecnologías](#tecnologías)
- [Características Utilizadas](#características-utilizadas)
- [Autor](#autor)

---

## Resultados del Modelo

| Métrica | Valor |
| :--- | :--- |
| **Mejor Modelo** | Random Forest |
| **Precisión (Cross-Validation)** | 82.6% |
| **Características Utilizadas** | 22 |
| **Datos de Entrenamiento** | 891 pasajeros |
| **Datos de Prueba** | 418 pasajeros |

### Comparativa de Modelos

| Modelo | Precisión CV |
| :--- | :---: |
| **Random Forest** | **82.6%** |
| Logistic Regression | 82.3% |
| SVM | 80.0% |

---

## Estructura del Proyecto

```text
AnalisisTitanic/
├── data/                  # Datos CSV
│   ├── train.csv          # Datos de entrenamiento
│   └── test.csv           # Datos de prueba
├── models/                # Modelos entrenados
│   ├── best_model.pkl    # Mejor modelo (Random Forest)
│   └── preprocessor.pkl  # Preprocesador entrenado
├── src/                   # Código fuente modular
│   ├── __init__.py        # Inicializador del paquete
│   ├── data_loader.py    # Carga de datos
│   ├── features.py       # Feature engineering (Ingeniería de variables)
│   ├── preprocessor.py   # Preprocesamiento y escalado
│   ├── model_trainer.py  # Entrenamiento de modelos
│   └── predict.py        # Predicciones e inferencia
├── dashboard/             # Aplicación web interactiva
│   └── app.py            # Dashboard interactivo con Streamlit
├── tests/                 # Pruebas unitarias
├── .gitignore             # Archivos ignorados por Git
├── README.md              # Este archivo
├── requirements.txt       # Dependencias del proyecto
└── run_pipeline.py        # Script principal de ejecución

---

## Instalacion

Sigue estos pasos en tu terminal para configurar el entorno localmente:

### 1. Clonar el repositorio

```bash
git clone https://github.com/Leogahu/TitanicSurvivalPredict.git
cd TitanicSurvivalPredict

### 2. Crear y activar el entorno virtual

**En Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate

**En Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate

### 3. Instalar dependencias
```bash
pip install -r requirements.txt

## Como Usar

### Ejecutar el Dashboard

El repositorio ya incluye el modelo entrenado (`models/best_model.pkl`) y el preprocesador (`models/preprocessor.pkl`). Para iniciar el dashboard:

```bash
streamlit run dashboard/app.py
