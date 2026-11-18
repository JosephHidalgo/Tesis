import pandas as pd

# Carga tu archivo CSV (asegúrate de cambiar 'dataset.csv' por la ruta real de tu archivo)
df = pd.read_csv('data.csv')

# Cuenta las frecuencias de cada valor en la columna 'mora'
conteo = df['mora'].value_counts()

print("Conteo de la columna 'mora' (Pandas):")
print(conteo)