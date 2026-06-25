# src/preprocessor.py (versión final y definitiva)
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TitanicPreprocessor:
    """
    Preprocesa los datos para el modelo: one-hot encoding, manejo de nulos y escalado.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.logger = logging.getLogger(__name__)

    def _safe_to_numeric(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Convierte de forma segura columnas a numérico.
        Los valores no numéricos se convierten a NaN.
        """
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns:
                # Intentar convertir a numérico, errores -> NaN
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                # Si toda la columna es NaN, la eliminamos
                if df_copy[col].isna().all():
                    self.logger.warning(f"Columna '{col}' no tiene valores numéricos, será eliminada")
                    df_copy = df_copy.drop(columns=[col])
        return df_copy

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Survived') -> pd.DataFrame:
        """
        Aplica el preprocesamiento completo al DataFrame de entrenamiento.
        """
        df_processed = df.copy()
        self.logger.info("Aplicando preprocesamiento...")

        # 1. Eliminar columnas que no se usarán en el modelo
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Dataset']
        for col in columns_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                self.logger.info(f"Eliminada columna: {col}")

        # 2. One-hot encoding para variables categóricas
        categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title', 'AgeBin', 'FareBin']
        existing_cats = [col for col in categorical_cols if col in df_processed.columns]
        
        if existing_cats:
            self.logger.info(f"Aplicando one-hot encoding a: {existing_cats}")
            df_processed = pd.get_dummies(df_processed, columns=existing_cats, drop_first=True)
            self.logger.info(f"One-hot encoding completado. Columnas: {len(df_processed.columns)}")
        
        # 3. Convertir TODAS las columnas a numérico
        all_columns = [col for col in df_processed.columns if col != target_col]
        df_processed = self._safe_to_numeric(df_processed, all_columns)
        
        # 4. Guardar las columnas de características
        self.feature_columns = [col for col in df_processed.columns if col != target_col]
        
        self.logger.info(f"Columnas numéricas: {len(self.feature_columns)}")
        self.logger.info(f"Columnas: {self.feature_columns}")
        
        # 5. Manejar valores nulos
        for col in self.feature_columns:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                # Usar asignación directa, no inplace
                df_processed[col] = df_processed[col].fillna(mean_val)
                self.logger.info(f"Rellenados nulos en '{col}' con media: {mean_val:.2f}")

        # 6. Escalado
        self.logger.info("Aplicando escalado...")
        self.scaler.fit(df_processed[self.feature_columns])
        scaled_features = self.scaler.transform(df_processed[self.feature_columns])
        df_scaled = pd.DataFrame(scaled_features, columns=self.feature_columns, index=df_processed.index)

        # 7. Añadir la variable objetivo
        if target_col in df_processed.columns:
            df_scaled[target_col] = df_processed[target_col]

        self.logger.info(f" Preprocesamiento completado. {len(self.feature_columns)} características.")
        return df_scaled

    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Aplica el mismo preprocesamiento a los datos de prueba.
        """
        if self.feature_columns is None:
            raise ValueError("Debes llamar a fit_transform primero con los datos de entrenamiento.")

        df_processed = df.copy()
        self.logger.info("Transformando datos de prueba...")

        # 1. Eliminar columnas que no se usarán
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Dataset']
        for col in columns_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])

        # 2. Aplicar one-hot encoding
        categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title', 'AgeBin', 'FareBin']
        existing_cats = [col for col in categorical_cols if col in df_processed.columns]
        
        if existing_cats:
            df_processed = pd.get_dummies(df_processed, columns=existing_cats, drop_first=True)

        # 3. Convertir a numérico
        all_columns = list(df_processed.columns)
        df_processed = self._safe_to_numeric(df_processed, all_columns)

        # 4. Asegurar que tiene todas las columnas del entrenamiento
        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # 5. Reordenar
        df_processed = df_processed[self.feature_columns]

        # 6. Rellenar nulos
        for col in self.feature_columns:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(mean_val)

        # 7. Escalar
        scaled_features = self.scaler.transform(df_processed)
        df_scaled = pd.DataFrame(scaled_features, columns=self.feature_columns, index=df.index)

        # 8. Mantener objetivo si existe
        if target_col and target_col in df.columns:
            df_scaled[target_col] = df[target_col]

        self.logger.info("Transformación de prueba completada.")
        return df_scaled

    def save(self, filepath: str = 'models/preprocessor.pkl'):
        """
        Guarda el preprocesador entrenado en disco.
        """
        import joblib
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        self.logger.info(f"Preprocesador guardado en: {filepath}")

    def load(self, filepath: str = 'models/preprocessor.pkl'):
        """
        Carga un preprocesador guardado desde disco.
        """
        import joblib
        from pathlib import Path
        
        if not Path(filepath).exists():
            self.logger.warning(f"No se encontró preprocesador en: {filepath}")
            return False
        
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.logger.info(f"Preprocesador cargado desde: {filepath}")
        return True