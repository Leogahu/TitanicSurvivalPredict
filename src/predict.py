# src/predict.py
import pandas as pd
import joblib
import logging
from src.data_loader import TitanicDataLoader
from src.features import FeatureEngineer
from src.preprocessor import TitanicPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TitanicPredictor:
    """
    Usa el modelo entrenado para hacer predicciones en nuevos datos.
    """
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        self.model = joblib.load(model_path)
        self.preprocessor = TitanicPreprocessor()
        self.logger = logging.getLogger(__name__)
        
    def predict_and_save(self, test_df: pd.DataFrame, output_path: str = 'submission.csv'):
        """
        Hace predicciones y las guarda en formato CSV para Kaggle.
        """
        passenger_ids = test_df['PassengerId']
        
        processed = self.preprocessor.transform(test_df)
        
        predictions = self.model.predict(processed)
        
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Submission guardada en: {output_path}")
        return submission