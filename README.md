# Titanic Survival Predictor

Pipeline completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic. Este proyecto incluye un dashboard interactivo para probar predicciones en tiempo real.

## Tabla de Contenidos

- [Resultados del Modelo](#resultados-del-modelo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalacion](#instalacion)
- [Uso](#uso)
- [Dashboard](#dashboard)
- [Tecnologias](#tecnologias)
- [Autor](#autor)

## Resultados del Modelo

| Metrica | Valor |
|---------|-------|
| Mejor Modelo | Random Forest |
| Precision (Cross-Validation) | 82.6% |
| Caracteristicas Utilizadas | 22 |
| Datos de Entrenamiento | 891 pasajeros |
| Datos de Prueba | 418 pasajeros |

### Comparativa de Modelos

| Modelo | Precision CV |
|--------|--------------|
| Random Forest | 82.6% |
| Logistic Regression | 82.3% |
| SVM | 80.0% |

## Estructura del Proyecto
AnalisisTitanic/
├── data/ # Datos CSV (no incluidos en el repositorio)
├── models/ # Modelos entrenados
│ ├── best_model.pkl # Mejor modelo (Random Forest)
│ └── preprocessor.pkl # Preprocesador entrenado
├── src/ # Codigo fuente
│ ├── init.py # Inicializador del paquete
│ ├── data_loader.py # Carga de datos
│ ├── features.py # Feature engineering
│ ├── preprocessor.py # Preprocesamiento y escalado
│ ├── model_trainer.py # Entrenamiento de modelos
│ └── predict.py # Predicciones
├── dashboard/ # Aplicacion Streamlit
│ └── app.py # Dashboard interactivo
├── tests/ # Pruebas unitarias
├── .gitignore # Archivos ignorados por Git
├── README.md # Este archivo
├── requirements.txt # Dependencias
└── run_pipeline.py # Script principal

## Instalacion

### 1. Clonar el repositorio

```bash
git clone https://github.com/Leogahu/TitanicSurvivalPredict.git
cd TitanicSurvivalPredict

# Crear entorno virtual
python -m venv .venv

# Activar en Windows
.venv\Scripts\activate

# Activar en Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

### 2. Crear y activar entorno virtual
# Crear entorno virtual
python -m venv .venv

# Activar en Windows
.venv\Scripts\activate

# Activar en Linux/Mac
source .venv/bin/activate

### 3. Instalar dependencias
pip install -r requirements.txt

Como Usar
Ejecutar el Dashboard
El repositorio incluye el modelo ya entrenado (models/best_model.pkl) y el preprocesador (models/preprocessor.pkl). Para ejecutar el dashboard:

streamlit run dashboard/app.py

Luego abre tu navegador en: http://localhost:8501

Caracteristicas Utilizadas
El modelo utiliza 22 caracteristicas:

Datos basicos: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Feature engineering: Title (Mr/Mrs/Miss/etc), FamilySize, IsAlone

Categorias: AgeBin (5 categorias), FareBin (4 categorias)

Autor
Leonardo G.

GitHub: Leogahu