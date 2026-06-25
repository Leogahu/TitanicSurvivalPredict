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

======================================================================
GUÍA DE INSTALACIÓN Y USO - TITANIC SURVIVAL PREDICTOR
======================================================================

----------------------------------------------------------------------
1. INSTALACIÓN
----------------------------------------------------------------------
Sigue estos pasos en tu terminal para configurar el entorno localmente:

A. Clonar el repositorio:
   git clone https://github.com/Leogahu/TitanicSurvivalPredict.git
   cd TitanicSurvivalPredict

B. Crear y activar el entorno virtual:
   
   En Windows:
   python -m venv .venv
   .venv\Scripts\activate

   En Linux / macOS:
   python -m venv .venv
   source .venv/bin/activate

C. Instalar dependencias:
   pip install -r requirements.txt


----------------------------------------------------------------------
2. CÓMO USAR
----------------------------------------------------------------------
A. Ejecutar el Dashboard:
   El repositorio ya incluye por defecto el modelo entrenado 
   (models/best_model.pkl) y el preprocesador (models/preprocessor.pkl).
   Para inicializar el dashboard interactivo ejecuta:

   streamlit run dashboard/app.py

   Luego, abre tu navegador web en la dirección local asignada:
   http://localhost:8501

B. Secciones del Dashboard:
   - Exploración de Datos: 
     Visualización y análisis exploratorio (EDA) de los datos del Titanic.
   - Predicción Individual: 
     Formulario en tiempo real para predecir la supervivencia de un pasajero específico.
   - Acerca de: 
     Información detallada sobre el proyecto y las tecnologías integradas.

C. Re-entrenar el Modelo (Opcional):
   Si deseas volver a entrenar los algoritmos con los datos originales:
   1. Descarga los archivos de data desde la competición de Kaggle: Titanic Data
   2. Coloca los archivos train.csv y test.csv dentro de la carpeta local data/
   3. Ejecuta el pipeline principal en tu terminal:

   python run_pipeline.py

   Esto evaluará los modelos candidatos de forma automática, actualizará 
   el archivo guardado con el mejor resultado en la carpeta models/ y 
   generará un archivo submission.csv óptimo para Kaggle.


----------------------------------------------------------------------
3. TECNOLOGÍAS UTILIZADAS
----------------------------------------------------------------------
- Python 3.14:   Lenguaje principal del ecosistema.
- Pandas:        Procesamiento y manipulación de datos estructurados.
- NumPy:         Operaciones matemáticas y cálculo matricial.
- Scikit-learn:  Modelado de Machine Learning, métricas y preprocesamiento.
- Streamlit:     Creación y despliegue del dashboard interactivo.
- Plotly:        Diseño de visualizaciones de gráficos dinámicos.
- Joblib:        Serialización, guardado y carga de modelos entrenados.


----------------------------------------------------------------------
4. CARACTERÍSTICAS UTILIZADAS (FEATURE ENGINEERING)
----------------------------------------------------------------------
El modelo final analiza un total de 22 características transformadas a 
partir de las siguientes variables base:

- Datos Básicos: 
  Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

- Ingeniería de Variables (Feature Engineering):
  * Title: Extracción de títulos de cortesía (Mr, Mrs, Miss, Master, etc.).
  * FamilySize: Total de miembros del núcleo familiar a bordo.
  * IsAlone: Variable binaria que identifica si el pasajero viajaba sin acompañantes.

- Categorizaciones (Binning):
  * AgeBin: Segmentación de edades estructurada en 5 rangos específicos.
  * FareBin: Distribución del costo del billete en 4 rangos de precios (cuartiles).


----------------------------------------------------------------------
5. INFORMACIÓN DEL AUTOR
----------------------------------------------------------------------
- Autor: Leonardo G.
- GitHub: @Leogahu
======================================================================
