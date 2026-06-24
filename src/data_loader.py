# src/data_loader.py
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TitanicDataLoader:
    """
    Carga y unifica los datos de entrenamiento y prueba del Titanic.
    """
    def __init__(self, data_dir: str = None):
        """
        Inicializa el cargador con la ruta a los datos.
        
        Args:
            data_dir: Ruta a la carpeta con los CSV. Si es None, se usa 'data/' 
                     relativo a la ubicación de este archivo.
        """
        if data_dir is None:
            # Obtiene la ruta del directorio donde está este archivo (src/)
            base_dir = Path(__file__).resolve().parent.parent
            self.data_dir = base_dir / 'data'
        else:
            self.data_dir = Path(data_dir)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Buscando datos en: {self.data_dir}")

    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Carga los datasets de entrenamiento y prueba."""
        train_path = self.data_dir / 'train.csv'
        test_path = self.data_dir / 'test.csv'

        if not train_path.exists():
            raise FileNotFoundError(f"No se encuentra train.csv en {self.data_dir}")

        self.logger.info("Cargando datos de entrenamiento...")
        train_df = pd.read_csv(train_path)
        self.logger.info(f"Train: {train_df.shape[0]} filas, {train_df.shape[1]} columnas.")

        test_df = None
        if test_path.exists():
            self.logger.info("Cargando datos de prueba...")
            test_df = pd.read_csv(test_path)
            self.logger.info(f"Test: {test_df.shape[0]} filas, {test_df.shape[1]} columnas.")
        else:
            self.logger.warning(f"No se encontró test.csv en {self.data_dir}")

        return train_df, test_df

    def get_combined_data(self) -> pd.DataFrame:
        """Une los conjuntos de train y test para un preprocesamiento conjunto."""
        train, test = self.load_data()
        if test is None:
            return train

        # Añadimos una columna para saber el origen
        train['Dataset'] = 'train'
        test['Dataset'] = 'test'

        # Combinamos
        combined = pd.concat([train, test], ignore_index=True, sort=False)
        self.logger.info(f"Datos combinados: {len(combined)} filas.")
        return combined