#Limpieza de datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el dataframe a través de pandas
df = pd.read_excel("datos/Exoplanets1.xlsx")

# Obtener el tipo de dato de las variables
df.info()

# Obtener el formato del dataframe
df.head()

# %%
#Esta fue una lista obtenida iterativamente por José Feliciano y Diego Vértiz tomando en cuenta el Diccionario de Datos de NASA Exoplanet Archive y decidiendo cuáles datos no son inicialmente los relevantes para el proyecto
listaNO = [
    "default_flag",
    "disc_year",
    "disc_facility",
    "soltype",
    "pl_controv_flag",
    "pl_refname",
    "st_refname",
    "st_met",
    "st_meterr1",
    "st_meterr2",
    "st_metlim",
    "st_metratio",
    "sy_refname",
    "rastr",
    "decstr",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_gaiamag",
    "sy_gaiamagerr1",
    "sy_gaiamagerr2",
    "rowupdate",
    "pl_pubdate",
    "releasedate",
    "xinglong Station"
]

# %%
#Eliminamos los datos que no son relevantes inicialmente 
for col in listaNO:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# %%
df.info()

# Crear una lista de las columnas con 'err1', 'err2' o 'lim' en su nombre
errores_y_limites = [col for col in df.columns if any(keyword in col for keyword in ['err1', 'err2', 'lim'])]

# Mostrar la lista para verificar
print("Columnas a eliminar:")
print(errores_y_limites)

df_cleaned = df.drop(columns=errores_y_limites)

df_cleaned.info()

listaNO2 = ["pl_bmassj", "pl_insol","st_spectype","pl_eqt","pl_bmassprov", "ra","pl_radj", "dec", "discoverymethod", "hostname"]

for col in listaNO2:
    if col in df_cleaned.columns:
        df_cleaned.drop(col, axis=1, inplace=True)

df_cleaned2 = df_cleaned.dropna()
df_cleaned2.info()
firstcleanexcel = df_cleaned2.to_excel("datos/Exoplanets2.xlsx")


