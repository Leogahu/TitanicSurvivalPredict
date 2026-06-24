# Titanic Survival Prediction

## Descripción
Pipeline completo de Machine Learning para predecir la supervivencia de pasajeros del Titanic.

## Resultados
- **Mejor Modelo**: Random Forest
- **Precisión (Cross-Validation)**: 82.6%
- **Features**: 22 características

## Estructura del Proyecto

AnalisisTitanic/
├── data/ # Datos CSV
├── src/ # Código fuente
│ ├── data_loader.py
│ ├── features.py
│ ├── preprocessor.py
│ ├── model_trainer.py
│ └── predict.py
├── models/ # Modelos guardados
├── dashboard/ # App Streamlit
└── tests/ # Pruebas unitarias


## Tecnologías
- Python 3.14
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly

## Cómo ejecutar
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python run_pipeline.py

# Ejecutar dashboard
streamlit run dashboard/app.py

