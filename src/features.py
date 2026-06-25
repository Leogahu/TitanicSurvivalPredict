# src/features.py
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """
    Aplica transformaciones e ingeniería de características al conjunto de datos.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Definir los bins una sola vez para consistencia
        self.age_bins = [0, 12, 18, 35, 60, 100]
        self.age_labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
        self.fare_bins = [0, 10, 30, 60, 600]  # Definir bins fijos
        self.fare_labels = ['Low', 'Medium', 'High', 'VeryHigh']

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones al DataFrame.
        """
        df = df.copy()
        self.logger.info("Aplicando ingeniería de características...")

        # 1. Título del pasajero (extraído del nombre)
        df['Title'] = df['Name'].apply(self._extract_title)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', \
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # 2. Tamaño de la familia
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # 3. Agrupar edades usando bins consistentes
        df['AgeBin'] = pd.cut(
            df['Age'], 
            bins=self.age_bins, 
            labels=self.age_labels,
            include_lowest=True
        )

        # 4. Agrupar tarifas usando bins consistentes
        df['FareBin'] = pd.cut(
            df['Fare'].fillna(0), 
            bins=self.fare_bins, 
            labels=self.fare_labels,
            include_lowest=True
        )

        # 5. ¿Viajaba solo?
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        self.logger.info("Ingeniería de características completada.")
        return df

    @staticmethod
    def _extract_title(name: str) -> str:
        """Extrae el título del nombre (ej. 'Mr.', 'Mrs.')"""
        title_search = re.search(r' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return "Unknown"