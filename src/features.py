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

        # 3. Agrupar edades (bins)
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                             labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

        # 4. Agrupar tarifas (bins)
        df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')

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