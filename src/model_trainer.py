# src/model_trainer.py
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TitanicModelTrainer:
    """me
    Entrena y evalúa modelos para el Titanic.
    """
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            'SVM': SVC(gamma=0.01, C=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        self.logger = logging.getLogger(__name__)

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame = None, y_test: pd.Series = None):
        """
        Entrena y evalúa todos los modelos.
        """
        self.results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Entrenando: {name}")
            self.logger.info(f"{'='*50}")
            
            try:
                # Entrenar
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                self.logger.info(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
                
                # Evaluar en test si hay datos
                test_score = cv_mean  # Por defecto, usar CV
                if X_test is not None and y_test is not None:
                    y_pred = model.predict(X_test)
                    test_score = accuracy_score(y_test, y_pred)
                    self.logger.info(f"Test Accuracy: {test_score:.4f}")
                    self.logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                    self.logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
                
                self.results[name] = {
                    'model': model,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'test_score': test_score
                }
                
                # Guardar el mejor modelo (basado en CV)
                if cv_mean > self.best_score:
                    self.best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                self.logger.error(f"Error entrenando {name}: {e}")
        
        # Mostrar resumen
        self._print_summary()
        return self.results

    def _print_summary(self):
        """Muestra un resumen de todos los resultados."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(" RESUMEN DE RESULTADOS")
        self.logger.info(f"{'='*50}")
        
        for name, result in self.results.items():
            self.logger.info(f"  {name}:")
            self.logger.info(f"    CV Score: {result['cv_score']:.4f} (+/- {result['cv_std']:.4f})")
            self.logger.info(f"    Test Score: {result['test_score']:.4f}")
        
        if self.best_model_name:
            self.logger.info(f"\n MEJOR MODELO: {self.best_model_name}")
            self.logger.info(f"   CV Score: {self.best_score:.4f}")

    def save_model(self, filepath: str = 'models/best_model.pkl'):
        """
        Guarda el mejor modelo en disco.
        """
        if self.best_model is None:
            self.logger.warning(" No hay modelo para guardar.")
            return False
        
        # Crear el directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            joblib.dump(self.best_model, filepath)
            self.logger.info(f" Modelo guardado en: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f" Error guardando el modelo: {e}")
            return False

    def load_model(self, filepath: str = 'models/best_model.pkl'):
        """
        Carga un modelo guardado desde disco.
        """
        try:
            self.best_model = joblib.load(filepath)
            self.logger.info(f" Modelo cargado desde: {filepath}")
            return self.best_model
        except Exception as e:
            self.logger.error(f" Error cargando el modelo: {e}")
            return None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Hace predicciones con el mejor modelo.
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Ejecuta train_and_evaluate primero.")
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Hace predicciones de probabilidad con el mejor modelo.
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Ejecuta train_and_evaluate primero.")
        
        # Algunos modelos no tienen predict_proba (como SVM con kernel='rbf')
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            self.logger.warning("El modelo no soporta predict_proba. Usando decision_function.")
            return self.best_model.decision_function(X)