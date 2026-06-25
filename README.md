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
```

---

## Instalacion

Sigue estos pasos en tu terminal para configurar el entorno localmente:

### 1. Clonar el repositorio

```bash
git clone https://github.com/Leogahu/TitanicSurvivalPredict.git
cd TitanicSurvivalPredict
```
### 2. Crear y activar el entorno virtual

**En Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**En Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Como Usar

### Ejecutar el Dashboard

El repositorio ya incluye el modelo entrenado (`models/best_model.pkl`) y el preprocesador (`models/preprocessor.pkl`). Para iniciar el dashboard:

```bash
streamlit run dashboard/app.py
```

Luego abre tu navegador en: `http://localhost:8501`

### Secciones del Dashboard

| Seccion | Descripcion |
|---------|-------------|
| Exploracion de Datos | Visualizacion y analisis exploratorio de los datos del Titanic |
| Prediccion Individual | Formulario para predecir la supervivencia de un pasajero |
| Acerca de | Informacion del proyecto y tecnologias utilizadas |

### Re-entrenar el Modelo (Opcional)

Si deseas volver a entrenar los modelos con los datos originales:

1. Descarga los archivos desde Kaggle: [Titanic Data](https://www.kaggle.com/competitions/titanic/data)
2. Coloca `train.csv` y `test.csv` en la carpeta `data/`
3. Ejecuta el pipeline:

```bash
python run_pipeline.py
```
## Tecnologias

| Tecnologia | Uso |
|------------|-----|
| Python 3.14 | Lenguaje principal |
| Pandas | Procesamiento de datos |
| NumPy | Operaciones matematicas |
| Scikit-learn | Modelos de Machine Learning |
| Streamlit | Dashboard interactivo |
| Plotly | Visualizaciones dinamicas |
| Joblib | Guardar y cargar modelos |

---

## Caracteristicas Utilizadas

El modelo final utiliza 22 caracteristicas transformadas a partir de las siguientes variables base:

- **Datos Basicos**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

- **Ingenieria de Variables (Feature Engineering)**:
  - **Title**: Extraccion de titulos de cortesia (Mr, Mrs, Miss, Master, etc.)
  - **FamilySize**: Total de miembros del nucleo familiar a bordo
  - **IsAlone**: Variable binaria que identifica si viajaba sin acompanantes

- **Categorizaciones (Binning)**:
  - **AgeBin**: Segmentacion de edades en 5 rangos
  - **FareBin**: Distribucion del costo del billete en 4 rangos

---

## Autor

**Leonardo G.**

- GitHub: [Leogahu](https://github.com/Leogahu)
