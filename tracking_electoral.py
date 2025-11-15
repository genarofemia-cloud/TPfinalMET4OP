#%%
#Primer paso: importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import scipy.stats as st
from statsmodels.sandbox.stats.runs import cochrans_q
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import os

#%%
#Segundo paso: importar el archivo
var ="C:/Users/userx/Downloads/mieencuesta.csv" 
pd.read_csv(var)
def cargar_datos(ruta):
    try:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        _, extension = os.path.splitext(ruta)
        extension = extension.lower().strip()
        if extension == ".csv":
            df = pd.read_csv(ruta, encoding="utf-8")
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(ruta)
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo Excel convertido y guardado como CSV → {ruta_csv}")
        elif extension == ".json":
            df = pd.read_json(ruta)
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo JSON convertido y guardado como CSV → {ruta_csv}")
        elif extension == ".txt":
            df = pd.read_csv(ruta, sep="\t", encoding="utf-8")
            ruta_csv = ruta.replace(extension, ".csv")
            df.to_csv(ruta_csv, index=False, encoding="utf-8")
            print(f"Archivo TXT convertido y guardado como CSV → {ruta_csv}")
        else:
            raise ValueError(f"Formato no soportado: {extension}")
        print(f"Archivo cargado correctamente ({extension}) → {ruta}")
        print(f"   Filas: {len(df)} | Columnas: {len(df.columns)}")
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except pd.errors.EmptyDataError:
        print("Error: el archivo está vacío.")
    except pd.errors.ParserError:
        print("Error: formato de archivo incorrecto o corrupto.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error inesperado al cargar el archivo: {e}")
    return None
df.columns = (
            df.columns
              .str.strip()   
              .str.lower()  
              .str.replace(" ", "_") 
              .str.replace(r"[^a-z0-9_]", "", regex=True)  
        )
columnas_requeridas = [
            "fecha", "encuesta", "estrato", "sexo", "edad", "nivel_educativo", "cantidad_de_integrantes_en_el_hogar", "imagen_del_candidato", "voto", "voto_anterior"
        ]
try:
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el archivo: {faltantes}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except pd.errors.EmptyDataError:
    print("Error: el archivo está vacío.")
except pd.errors.ParserError:
    print("Error: formato de archivo incorrecto o corrupto.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error inesperado al cargar el archivo: {e}")
df['fecha'] = pd.to_datetime(df['fecha'])

#%%
#Tercer paso: manipular los valores faltantes
df = df[~df[['voto', 'imagen_del_candidato']].isna().all(axis=1)] #si faltan las 2 variables claves, descartar encuesta
df = df[df['edad'] >= 16] #si alguien tiene menos de 16, se borra la encuesta ya que no puede votar
df = df[~df['encuesta'].duplicated()] #si está duplicado, borrarlo
df['fecha'] = df['fecha'].interpolate()
ddf = df.dropna(subset=['estrato'])
df = df.dropna(subset=['sexo'])
df = df.dropna(subset=['edad'])
df['nivel_educativo'] = df['nivel_educativo'].fillna('Desconocido')
df['cantidad_de_integrantes_en_el_hogar'] = df['cantidad_de_integrantes_en_el_hogar'].fillna('Desconocido')
df_full_va = df[df['voto_anterior'].notna()]
df_missing_va = df[df['voto_anterior'].isna()]
if len(df_missing_va) > 0:
    features_va = ['edad', 'sexo', 'estrato', 'nivel_educativo']
    X_full_va = pd.get_dummies(df_full_va[features_va], drop_first=True)
    y_full_va = df_full_va['voto_anterior']
    model_va = LogisticRegression(multi_class='multinomial', max_iter=2000)
    model_va.fit(X_full_va, y_full_va)
    X_missing_va = pd.get_dummies(df_missing_va[features_va], drop_first=True)
    X_missing_va = X_missing_va.reindex(columns=X_full_va.columns, fill_value=0)
    preds_va = model_va.predict(X_missing_va)
    df.loc[df['voto_anterior'].isna(), 'voto_anterior'] = preds_va
df_full_voto = df[df['voto'].notna()]
df_missing_voto = df[df['voto'].isna()]
if len(df_missing_voto) > 0:
    features_voto = [
        'edad', 'sexo', 'estrato',
        'nivel_educativo', 'voto_anterior'
    ]
    X_full_voto = pd.get_dummies(df_full_voto[features_voto], drop_first=True)
    y_full_voto = df_full_voto['voto']
    model_log = LogisticRegression(
        multi_class='multinomial',
        max_iter=2000
    )
    model_log.fit(X_full_voto, y_full_voto)
    X_missing_voto = pd.get_dummies(df_missing_voto[features_voto], drop_first=True)
    X_missing_voto = X_missing_voto.reindex(columns=X_full_voto.columns, fill_value=0)
    preds_voto = model_log.predict(X_missing_voto)
    df.loc[df['voto'].isna(), 'voto'] = preds_voto
df_full_img = df[df['imagen_del_candidato'].notna()]
df_missing_img = df[df['imagen_del_candidato'].isna()]
if len(df_missing_img) > 0:
    features_img = [
        'edad', 'sexo', 'estrato',
        'nivel_educativo', 'voto', 'voto_anterior'
    ]
    X_full_img = pd.get_dummies(df_full_img[features_img], drop_first=True)
    y_full_img = df_full_img['imagen_del_candidato']
    model_lin = LinearRegression()
    model_lin.fit(X_full_img, y_full_img)
    X_missing_img = pd.get_dummies(df_missing_img[features_img], drop_first=True)
    X_missing_img = X_missing_img.reindex(columns=X_full_img.columns, fill_value=0)
    preds_img = model_lin.predict(X_missing_img)
    df.loc[df['imagen_del_candidato'].isna(), 'imagen_del_candidato'] = preds_img
df

# %%
#Cuarto Paso: definir la ventana
df = df.sort_values('fecha')
df['Ventana_D'] = df['fecha']
df['Ventana_S'] = df['fecha'].dt.to_period('W')

# %%
#Quinto Paso: pesos por ventanas
df['peso_d'] = 1
df['peso_s'] = 1
df['edad_cat'] = pd.cut(
    df['edad'],
    bins=[15, 29, 44, 59, 120],
    labels=['16-29', '30-44', '45-59', '60+']
)
targets = {
    'sexo': {'Femenino': 0.53, 'Masculino': 0.47},
    'edad_cat': {'16-29': 0.29, '30-44': 0.29, '45-59':0.21, '60+': 0.21},
    'estrato': {'Ciudad Autónoma de Buenos Aires':0.07,'Buenos Aires':0.38,'Catamarca':0.02,'Chaco':0.02,'Chubut':0.01,'Córdoba':0.09,'Corrientes':0.03,'Entre Ríos':0.03,'Formosa':0.01,'Jujuy':0.02,'La Pampa':0.01,'La Rioja':0.01,'Mendoza':0.04,'Misiones':0.02,'Neuquén':0.01,'Río Negro':0.01,'Salta':0.03,'San Juan':0.02,'San Luis':0.01,'Santa Cruz':0.01,'Santa Fe':0.08,'Santiago del Estero':0.02,'Tierra del Fuego':0.01,'Tucumán':0.04}
}
for var in targets.keys():
    df[var] = df[var].astype(str)
def rake(df, weight_col, targets, max_iter=50, tol=1e-6):
    df = df.copy()
    for i in range(max_iter):
        old_weights = df[weight_col].copy()
        for var, target_dist in targets.items():
            current_totals = (
                df.groupby(var)[weight_col].sum() / df[weight_col].sum()
            ).to_dict()
            ratios = {
                cat: target_dist[cat] / current_totals.get(cat, 1)
                for cat in target_dist
            }
            df[weight_col] *= df[var].map(ratios)
        if np.max(np.abs(df[weight_col] - old_weights)) < tol:
            break
    return df
df['peso_d'] = (
    df.groupby('Ventana_D', group_keys=False)
      .apply(lambda g: rake(g, 'peso_d', targets)['peso_d'])
)
df['peso_s'] = (
    df.groupby('Ventana_S', group_keys=False)
      .apply(lambda g: rake(g, 'peso_s', targets)['peso_s'])
)
df

# %%
#Sexto Paso: ANALIZAR LA EVOLUCION DE LA INTENCIÓN DE VOTO A LO LARGO DEL TIEMPO
#ventana diaria
df['Imagen_Ponderada_d'] = df['peso_d'] * df['imagen_del_candidato']
tracking_imagen_diario = (
    df.groupby('Ventana_D')
      .apply(lambda g: g['Imagen_Ponderada_d'].sum() / g['peso_d'].sum())
      .reset_index(name='trackeo')
)
tracking_imagen_diario

# %%
#Séptimo Paso: Graficar la evolución de la imagen
plt.figure(figsize=(10,5))
plt.plot(tracking_imagen_diario['Ventana_D'], tracking_imagen_diario['trackeo'], marker='o')
plt.xlabel('Ventana (diaria)', fontsize = 10)
plt.ylabel('Imagen promedio', fontsize = 10)
plt.title('Evolución de la imagen del candidato (ventana diaria)', fontsize = 16)
plt.tight_layout()
plt.show()

# %%
#Octavo Paso: Analizar la evolución de la intención de voto del candidato
#ventana diaria
candidatos = df['voto'].unique().tolist()
for c in candidatos:
    df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    df[f'vota_{c}_ponderada'] = df[f'vota_{c}'] * df['peso_d']
tracking_voto_diario = (
    df.groupby('Ventana_D')
      .apply(lambda g: pd.Series({
          f"Vota_{c}": (g[f'vota_{c}_ponderada'].sum() / g['peso_d'].sum()) * 100
          for c in candidatos
      }))
      .reset_index()
)
tracking_voto_diario.round(1)
