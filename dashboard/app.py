# dashboard/app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import TitanicDataLoader
from src.features import FeatureEngineer
from src.preprocessor import TitanicPreprocessor

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon=":ship:",
    layout="wide"
)

st.title("Titanic Survival Prediction Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    loader = TitanicDataLoader()
    combined = loader.get_combined_data()
    train = combined[combined['Dataset'] == 'train'].copy()
    test = combined[combined['Dataset'] == 'test'].copy()
    return train, test

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        return model
    except:
        return None

@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = TitanicPreprocessor()
        if preprocessor.load('models/preprocessor.pkl'):
            return preprocessor
        return None
    except:
        return None

train_df, test_df = load_data()
model = load_model()
preprocessor = load_preprocessor()

st.sidebar.title("Navegacion")
page = st.sidebar.radio(
    "Selecciona una seccion:",
    ["Exploracion de Datos", "Prediccion Individual", "Acerca de"]
)

if page == "Exploracion de Datos":
    st.header("Exploracion de Datos del Titanic")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pasajeros", len(train_df))
    with col2:
        st.metric("Sobrevivientes", int(train_df['Survived'].sum()))
    with col3:
        st.metric("Tasa de Supervivencia", f"{train_df['Survived'].mean()*100:.1f}%")
    with col4:
        st.metric("Clases", train_df['Pclass'].nunique())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Supervivencia por Clase")
        plot_df = train_df.copy()
        plot_df['Survived_Label'] = plot_df['Survived'].map({0: 'No Sobrevivio', 1: 'Sobrevivio'})
        fig = px.histogram(
            plot_df, 
            x='Pclass', 
            color='Survived_Label',
            barmode='group', 
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            labels={'Pclass': 'Clase', 'count': 'Numero de Pasajeros'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Supervivencia por Sexo")
        plot_df = train_df.copy()
        plot_df['Survived_Label'] = plot_df['Survived'].map({0: 'No Sobrevivio', 1: 'Sobrevivio'})
        fig = px.histogram(
            plot_df, 
            x='Sex', 
            color='Survived_Label',
            barmode='group', 
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            labels={'Sex': 'Sexo', 'count': 'Numero de Pasajeros'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Distribucion de Edades")
        plot_df = train_df.copy()
        plot_df['Survived_Label'] = plot_df['Survived'].map({0: 'No Sobrevivio', 1: 'Sobrevivio'})
        fig = px.histogram(
            plot_df, 
            x='Age', 
            color='Survived_Label',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            labels={'Age': 'Edad', 'count': 'Numero de Pasajeros'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Distribucion de Tarifas")
        plot_df = train_df.copy()
        plot_df['Survived_Label'] = plot_df['Survived'].map({0: 'No Sobrevivio', 1: 'Sobrevivio'})
        fig = px.histogram(
            plot_df, 
            x='Fare', 
            color='Survived_Label',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            labels={'Fare': 'Tarifa', 'count': 'Numero de Pasajeros'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    with st.expander("Ver Datos de Entrenamiento"):
        display_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        display_df = train_df[display_cols].copy()
        display_df['Survived'] = display_df['Survived'].map({0: 'No', 1: 'Si'})
        st.dataframe(display_df.head(100), use_container_width=True)

elif page == "Prediccion Individual":
    st.header("Prediccion Individual de Supervivencia")
    
    if model is None:
        st.warning("No se encontro un modelo entrenado. Ejecuta run_pipeline.py primero.")
        st.info("""
        Para entrenar el modelo:
        1. Abre la terminal
        2. Ejecuta: python run_pipeline.py
        3. Espera a que termine
        4. Reinicia este dashboard
        """)
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Clase del Pasajero", [1, 2, 3], 
                              help="1 = Primera Clase, 2 = Segunda Clase, 3 = Tercera Clase")
        sex = st.selectbox("Sexo", ["male", "female"])
        age = st.slider("Edad", 0, 100, 30)
        
    with col2:
        sibsp = st.number_input("Hermanos/Conyuges a bordo", 0, 8, 0,
                                help="Numero de hermanos o conyuges a bordo")
        parch = st.number_input("Padres/Hijos a bordo", 0, 6, 0,
                                help="Numero de padres o hijos a bordo")
        fare = st.slider("Tarifa pagada", 0.0, 500.0, 30.0, step=5.0)
        embarked = st.selectbox("Puerto de Embarque", ["S", "C", "Q"],
                                help="S = Southampton, C = Cherbourg, Q = Queenstown")
    
    if st.button("Predecir Supervivencia", type="primary"):
        try:
            passenger_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Embarked': [embarked],
                'Name': ['Pasajero'],
                'Ticket': [''],
                'Cabin': ['']
            })
            
            engineer = FeatureEngineer()
            featured = engineer.create_features(passenger_data)
            
            if preprocessor is None:
                st.warning("El preprocesador no esta disponible. Ejecuta run_pipeline.py primero.")
                st.stop()
            
            try:
                processed = preprocessor.transform(featured)
                prediction = model.predict(processed)[0]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(processed)[0][1]
                else:
                    proba = 0.5
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("### El pasajero SOBREVIVIRIA al Titanic")
                    else:
                        st.error("### El pasajero NO SOBREVIVIRIA al Titanic")
                    st.metric("Probabilidad de Supervivencia", f"{proba*100:.1f}%")
                
                with col2:
                    fig = go.Figure(data=[go.Bar(
                        x=['No Sobrevive', 'Sobrevive'],
                        y=[1-proba, proba],
                        marker_color=['#FF6B6B', '#4ECDC4']
                    )])
                    fig.update_layout(
                        title="Probabilidad de Supervivencia",
                        height=300,
                        yaxis_title="Probabilidad"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except ValueError as e:
                st.warning("El preprocesador no esta entrenado correctamente.")
                st.info("""
                **Para predicciones reales:**
                1. Ejecuta: python run_pipeline.py
                2. Luego reinicia este dashboard
                """)
                
                survival_score = 0
                if sex == 'female':
                    survival_score += 50
                if pclass == 1:
                    survival_score += 30
                elif pclass == 2:
                    survival_score += 10
                if age < 12:
                    survival_score += 20
                elif age > 60:
                    survival_score += 10
                
                st.subheader("Demo - Prediccion Basada en Reglas")
                st.metric("Puntaje de Supervivencia", f"{survival_score}%")
                if survival_score > 50:
                    st.success("El pasajero probablemente SOBREVIVIRIA")
                else:
                    st.error("El pasajero probablemente NO SOBREVIVIRIA")
                
        except Exception as e:
            st.error(f"Error en la prediccion: {e}")
            st.info("Asegurate de haber ejecutado run_pipeline.py primero.")

else:
    st.header("Acerca del Proyecto")
    
    st.markdown("""
    ### Proyecto Titanic - Prediccion de Supervivencia
    
    Este proyecto implementa un pipeline completo de Machine Learning para predecir 
    la supervivencia de pasajeros del Titanic basado en diversas caracteristicas.
    
    ### Resultados del Modelo
    - **Mejor Modelo**: Random Forest
    - **Precision (CV)**: 82.6%
    - **Caracteristicas**: 22 features
    - **Datos de Entrenamiento**: 891 pasajeros
    
    ### Caracteristicas Utilizadas
    - Clase del Pasajero (Pclass)
    - Sexo
    - Edad
    - Numero de hermanos/conyuges a bordo (SibSp)
    - Numero de padres/hijos a bordo (Parch)
    - Tarifa pagada
    - Puerto de embarque (Embarked)
    - Titulo extraido del nombre
    - Tamano de la familia
    - Categorias de edad
    - Categorias de tarifa
    - Indicador de viaje en solitario
    
    ### Tecnologias Utilizadas
    - Python 3.14
    - Pandas y NumPy para procesamiento de datos
    - Scikit-learn para modelos de Machine Learning
    - Streamlit para el dashboard interactivo
    - Plotly para visualizaciones
    
    ### Estructura del Proyecto

    AnalisisTitanic/
    ├── data/ # Archivos CSV
    ├── src/ # Modulos de codigo fuente
    │ ├── data_loader.py
    │ ├── features.py
    │ ├── preprocessor.py
    │ ├── model_trainer.py
    │ └── predict.py
    ├── models/ # Modelos guardados
    ├── dashboard/ # Aplicacion Streamlit
    └── tests/ # Pruebas unitarias
                
### Como Usar
1. Entrenar el modelo: `python run_pipeline.py`
2. Iniciar el dashboard: `streamlit run dashboard/app.py`
3. Usar la pestana "Prediccion Individual" para probar escenarios


### Autor
Leonardo G.
""")