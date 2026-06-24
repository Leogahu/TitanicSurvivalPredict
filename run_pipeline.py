# run_pipeline.py
from src.data_loader import TitanicDataLoader
from src.features import FeatureEngineer
from src.preprocessor import TitanicPreprocessor
from src.model_trainer import TitanicModelTrainer
import pandas as pd
import joblib

print("=" * 60)
print("PIPELINE COMPLETO DEL TITANIC")
print("=" * 60)

# 1. Cargar datos
loader = TitanicDataLoader()
combined_df = loader.get_combined_data()

# 2. Feature Engineering
engineer = FeatureEngineer()
featured_df = engineer.create_features(combined_df)

# 3. Preprocesamiento
preprocessor = TitanicPreprocessor()
train_data = preprocessor.fit_transform(featured_df[featured_df['Dataset'] == 'train'])
test_data = preprocessor.transform(featured_df[featured_df['Dataset'] == 'test'])

print(f"\nDatos procesados:")
print(f"   Entrenamiento: {train_data.shape}")
print(f"   Prueba: {test_data.shape}")

# 4. Separar características y objetivo
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# 5. Entrenar modelos
trainer = TitanicModelTrainer()
results = trainer.train_and_evaluate(X_train, y_train)

# 6. Guardar el mejor modelo
trainer.save_model()

# 7. Generar submission para Kaggle (TODO EN EL MISMO SCRIPT)
print("\n" + "=" * 60)
print("GENERANDO SUBMISSION PARA KAGGLE")
print("=" * 60)

# Obtener datos de prueba originales
test_original = combined_df[combined_df['Dataset'] == 'test'].copy()

# Aplicar feature engineering a los datos de prueba
test_featured = engineer.create_features(test_original)

# Preprocesar usando el preprocesador ya entrenado
test_processed = preprocessor.transform(test_featured)

# Cargar el mejor modelo guardado
best_model = joblib.load('models/best_model.pkl')

# Predecir
predictions = best_model.predict(test_processed)

# Crear archivo de submission
submission = pd.DataFrame({
    'PassengerId': test_original['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Submission guardada en: submission.csv")
print(f"Total de predicciones: {len(submission)}")
print(f"Distribución de predicciones:")
print(submission['Survived'].value_counts())

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 60)