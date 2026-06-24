# diagnostic.py
import pandas as pd
from src.data_loader import TitanicDataLoader
from src.features import FeatureEngineer

print("=" * 70)
print("DIAGNÓSTICO DE DATOS DEL TITANIC")
print("=" * 70)

# 1. Cargar datos
loader = TitanicDataLoader()
combined_df = loader.get_combined_data()

print(f"\n Datos combinados: {combined_df.shape[0]} filas, {combined_df.shape[1]} columnas")

# 2. Aplicar feature engineering
engineer = FeatureEngineer()
featured_df = engineer.create_features(combined_df)

print(f"\n Después de feature engineering: {featured_df.shape[0]} filas, {featured_df.shape[1]} columnas")

# 3. Ver tipos de datos
print("\n" + "=" * 70)
print("TIPOS DE DATOS POR COLUMNA")
print("=" * 70)

problem_columns = []
for col in featured_df.columns:
    dtype = featured_df[col].dtype
    if dtype == 'object' or dtype == 'string' or dtype == 'str':
        problem_columns.append(col)
        print(f"  {col}: {dtype} (texto)")
        print(f"   Ejemplos: {featured_df[col].head(3).tolist()}")
    else:
        print(f" {col}: {dtype}")

# 4. Ver columnas problemáticas
if problem_columns:
    print("\n" + "=" * 70)
    print(f" COLUMNAS PROBLEMÁTICAS ({len(problem_columns)})")
    print("=" * 70)
    for col in problem_columns:
        unique_vals = featured_df[col].unique()
        print(f"\n {col}")
        print(f"   Tipo: {featured_df[col].dtype}")
        print(f"   Valores únicos: {unique_vals[:10]}")
        print(f"   ¿Son numéricos? {all(pd.to_numeric(unique_vals, errors='coerce').notna())}")
else:
    print("\n Todas las columnas son numéricas!")

# 5. Ver columnas que deberían ser numéricas pero no lo son
print("\n" + "=" * 70)
print("VERIFICACIÓN DE COLUMNAS CON POTENCIALES PROBLEMAS")
print("=" * 70)

numeric_candidates = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
for col in numeric_candidates:
    if col in featured_df.columns:
        if featured_df[col].dtype == 'object' or featured_df[col].dtype == 'string':
            print(f" {col}: es texto, debería ser numérico")
            print(f"   Ejemplos: {featured_df[col].head(3).tolist()}")
        else:
            print(f" {col}: es numérico ({featured_df[col].dtype})")
    else:
        print(f" {col}: no existe en el DataFrame")

# 6. Recomendación
print("\n" + "=" * 70)
print("RECOMENDACIÓN")
print("=" * 70)
if problem_columns:
    print(f"Hay {len(problem_columns)} columnas problemáticas.")
    print("Esto causa el error 'Cannot perform reduction 'mean' with string dtype'.")
    print("\nLa solución es convertir estas columnas a numérico antes de calcular la media.")
else:
    print(" Todas las columnas son numéricas. El preprocesador debería funcionar.")