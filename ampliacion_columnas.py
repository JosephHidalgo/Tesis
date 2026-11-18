import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semilla para reproducibilidad

df = pd.read_csv("dataset_reducido.csv")
n = len(df)

# 1. Género
df["genero"] = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])

# 2. Estado civil (depende de la edad)
def estado_civil(edad):
    if edad < 25:
        p = [0.75, 0.05, 0.20, 0.00, 0.00]
    elif edad < 40:
        p = [0.30, 0.30, 0.35, 0.04, 0.01]
    elif edad < 55:
        p = [0.12, 0.45, 0.28, 0.12, 0.03]
    else:
        p = [0.08, 0.50, 0.15, 0.12, 0.15]
    return rng.choice(
        ["SOLTERO", "CASADO", "CONVIVIENTE", "DIVORCIADO", "VIUDO"], p=p
    )

df["estado_civil"] = df["edad"].apply(estado_civil)

# 3. Sector económico
sectores = ["COMERCIO", "TRANSPORTES", "EDUCATIVO", "SERVICIOS",
            "AGROPECUARIO", "CONSTRUCCION", "MANUFACTURA", "SALUD", "OTROS"]
probs = [0.30, 0.12, 0.08, 0.18, 0.10, 0.08, 0.07, 0.04, 0.03]
df["sector_economico"] = rng.choice(sectores, size=n, p=probs)



# 4. Experiencia con otras entidades financieras (binaria: 1 = sí, 0 = no)
#    La probabilidad de haber trabajado con otras entidades sube con la edad
#    y con el ingreso (de ~20% en jóvenes de bajo ingreso hasta ~80%)
z = (
    -1.5
    + 0.07 * (df["edad"] - 18)
    + 0.4 * (np.log(df["ingreso"].clip(lower=500)) - np.log(2500))
)
p_exp = 1 / (1 + np.exp(-z))
df["exp_otras_entidades"] = rng.binomial(1, p_exp)

# 5. Forma de pago
df["forma_pago"] = rng.choice(
    ["DIARIO", "SEMANAL", "QUINCENAL", "MENSUAL"],
    size=n, p=[0.10, 0.20, 0.25, 0.45]
)

# 6. Número de cuotas (depende de la forma de pago)
rangos_cuotas = {
    "DIARIO": (30, 90),
    "SEMANAL": (8, 26),
    "QUINCENAL": (6, 18),
    "MENSUAL": (6, 36),
}
df["num_cuotas"] = df["forma_pago"].apply(
    lambda f: rng.integers(rangos_cuotas[f][0], rangos_cuotas[f][1] + 1)
)

# 7. Tasa de interés (TEA %, rango típico de microfinanzas)
#    Base ~ 35%, sube con menor ingreso, sin experiencia y mora previa
tasa = (
    35
    + 8 * (df["mora"] == 1)
    - 4 * df["exp_otras_entidades"]
    - 4 * (np.log(df["ingreso"].clip(lower=500)) - np.log(2500))
    + rng.normal(0, 5, size=n)
)
df["tasa_interes"] = tasa.clip(18, 90).round(2)

# Guardar
df.to_csv("dataset_simulado.csv", index=False)

print("Columnas:", list(df.columns))
print(df.head())