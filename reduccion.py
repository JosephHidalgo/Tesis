import pandas as pd

# Cargar el dataset
df = pd.read_csv("data.csv")

# Columnas a suprimir
cols_a_eliminar = [
    "dias_lab", "exp_sf", "nivel_ahorro", "linea_sf",
    "deuda_sf", "score", "zona", "clasif_sbs"
]

# Eliminar (errors="ignore" evita fallos si alguna columna no existe)
df = df.drop(columns=cols_a_eliminar, errors="ignore")

# Guardar el resultado
df.to_csv("dataset_reducido.csv", index=False)

print("Columnas restantes:", list(df.columns))
print(df.head())