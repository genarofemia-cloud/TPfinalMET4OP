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
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import geopandas as gpd

#%%
#Segundo paso: importar el archivo
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
ruta ="C:/Users/userx/Downloads/mieencuesta.csv" 
df = cargar_datos(ruta)
df.columns = (
            df.columns
              .str.strip()   
              .str.lower()  
              .str.replace(" ", "_")  
        )
columnas_requeridas = [
    "fecha", "encuesta", "estrato", "sexo", "edad",
    "nivel_educativo", "cantidad_de_integrantes_en_el_hogar",
    "imagen_del_candidato", "voto", "voto_anterior"
]
faltantes = [col for col in columnas_requeridas if col not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas requeridas en el archivo: {faltantes}")
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.rename(columns={
    "cantidad_de_integrantes_en_el_hogar": "integrantes_hogar"
})

#%%Tercer Paso: normalización de variables
df['estrato'] = df['estrato'].astype(str).str.strip().str.lower()
df['region'] = df['estrato'].map({
    'buenos aires': 'Región Pampeana',
    'ciudad autónoma de buenos aires': 'Región Pampeana',
    'córdoba': 'Región Pampeana',
    'entre ríos': 'Región Pampeana',
    'la pampa': 'Región Pampeana',
    'santa fe': 'Región Pampeana',
    'catamarca': 'Región NOA',
    'jujuy': 'Región NOA',
    'la rioja': 'Región NOA',
    'salta': 'Región NOA',
    'santiago del estero': 'Región NOA',
    'tucumán': 'Región NOA',
    'chaco': 'Región NEA',
    'corrientes': 'Región NEA',
    'formosa': 'Región NEA',
    'misiones': 'Región NEA',
    'mendoza': 'Región Cuyo',
    'san juan': 'Región Cuyo',
    'san luis': 'Región Cuyo',
    'chubut': 'Región Patagonia',
    'neuquén': 'Región Patagonia',
    'río negro': 'Región Patagonia',
    'santa cruz': 'Región Patagonia',
    'tierra del fuego': 'Región Patagonia'
})
df['nivel_educativo'] = df['nivel_educativo'].astype(str).str.strip().str.lower()
df['nivel_educativo'] = df['nivel_educativo'].replace({
    'sin estudios': 'primaria' #para evitar el colapso del raking, se agrupa
})
df['sexo'] = df['sexo'].astype(str).str.strip().str.lower()

#%%
#Cuarto paso: manipular los valores faltantes para las VI
df = df[~df[['voto', 'imagen_del_candidato']].isna().all(axis=1)] #si faltan las 2 variables claves, descartar encuesta
df = df[df['edad'] >= 16] #si alguien tiene menos de 16, se borra la encuesta ya que no puede votar
df = df[~df['encuesta'].duplicated()] #si está duplicado, borrarlo
df = df.dropna(subset=['fecha'])
df = df.dropna(subset=['estrato'])
df = df.dropna(subset=['nivel_educativo'])
def normalizar_nivel_educativo(x):
    niveles_base = ["primaria", "secundaria", "terciario", "universitario", "posgrado"]
    x = str(x).lower().strip()
    for nivel in niveles_base:
        if x.startswith(nivel):
            return nivel
    return x
df['nivel_educativo'] = df['nivel_educativo'].apply(normalizar_nivel_educativo)
df = df.dropna(subset=['sexo'])
df = df.dropna(subset=['edad'])
df['integrantes_hogar'] = df['integrantes_hogar'].fillna('Desconocido')

#%% 
#Quinto Paso: calcular los valores faltantes para las VD
#Evaluacion modelos
print("Evaluación de modelos de regresión para imputación: logística y lineal\n")
#Evaluación modelo: voto_anterior
print("EVALUACIÓN VOTO_ANTERIOR\n")
df_eval_va = df[df['voto_anterior'].notna()].copy()
features_va_eval = ['edad', 'sexo', 'region', 'nivel_educativo']
X_va = pd.get_dummies(df_eval_va[features_va_eval], drop_first=True)
y_va = df_eval_va['voto_anterior'].astype('category')
y_va_num = y_va.cat.codes
mapeo_va = dict(enumerate(y_va.cat.categories))
X_train_va, X_test_va, y_train_va, y_test_va = train_test_split(
    X_va, y_va_num,
    test_size=0.3,
    random_state=42,
    stratify=y_va_num
)
model_va_eval = LogisticRegression(
    multi_class='multinomial',
    solver='newton-cg',
    max_iter=2000
)
model_va_eval.fit(X_train_va, y_train_va)
y_pred_va = model_va_eval.predict(X_test_va)
print("Accuracy:", accuracy_score(y_test_va, y_pred_va))
labels_va = np.unique(y_test_va)
names_va = [mapeo_va[i] for i in labels_va]
print("\nClassification report:")
print(classification_report(
    y_test_va, y_pred_va,
    labels=labels_va,
    target_names=names_va,
    zero_division=0
))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_va, y_pred_va, labels=labels_va))
#Evaluacion modelo: voto
print("\nEVALUACIÓN VOTO\n")
df_eval_v = df[df['voto'].notna()].copy()
features_voto_eval = ['edad', 'sexo', 'region', 'nivel_educativo', 'voto_anterior']
X_v = pd.get_dummies(df_eval_v[features_voto_eval], drop_first=True)
y_v = df_eval_v['voto'].astype('category')
y_v_num = y_v.cat.codes
mapeo_v = dict(enumerate(y_v.cat.categories))
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_v, y_v_num,
    test_size=0.3,
    random_state=42,
    stratify=y_v_num
)
model_v_eval = LogisticRegression(
    multi_class='multinomial',
    solver='newton-cg',
    max_iter=2000
)
model_v_eval.fit(X_train_v, y_train_v)
y_pred_v = model_v_eval.predict(X_test_v)
print("Accuracy:", accuracy_score(y_test_v, y_pred_v))
labels_v = np.unique(y_test_v)
names_v = [mapeo_v[i] for i in labels_v]
print("\nClassification report:")
print(classification_report(
    y_test_v, y_pred_v,
    labels=labels_v,
    target_names=names_v,
    zero_division=0
))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_v, y_pred_v, labels=labels_v))
print("\nEVALUACIÓN IMAGEN_DEL_CANDIDATO\n")
#Evaluacion modelo: imagen_del_candidato
df_eval_img = df[df['imagen_del_candidato'].notna()].copy()
features_img_eval = ['edad', 'sexo', 'region', 'nivel_educativo', 'voto', 'voto_anterior']
X_img = pd.get_dummies(df_eval_img[features_img_eval], drop_first=True)
y_img = df_eval_img['imagen_del_candidato']
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img,
    test_size=0.3,
    random_state=42
)
model_img_eval = LinearRegression()
model_img_eval.fit(X_train_img, y_train_img)
y_pred_img = model_img_eval.predict(X_test_img)
print("MAE:", mean_absolute_error(y_test_img, y_pred_img))
print("RMSE:", np.sqrt(mean_squared_error(y_test_img, y_pred_img)))
print("R²:", r2_score(y_test_img, y_pred_img))
def imputar_categorica(df, variable_objetivo, variables_predictoras):
    df_full = df[df[variable_objetivo].notna()]
    df_miss = df[df[variable_objetivo].isna()]
    if len(df_miss) == 0:
        return df
    X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
    y_full = df_full[variable_objetivo]
    model = LogisticRegression(
        multi_class='multinomial',
        solver='newton-cg',
        max_iter=2000
    )
    model.fit(X_full, y_full)
    X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
    X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
    preds = model.predict(X_miss)
    df.loc[df[variable_objetivo].isna(), variable_objetivo] = preds
    return df
def imputar_numerica(df, variable_objetivo, variables_predictoras):
    df_full = df[df[variable_objetivo].notna()]
    df_miss = df[df[variable_objetivo].isna()]
    if len(df_miss) == 0:
        return df
    X_full = pd.get_dummies(df_full[variables_predictoras], drop_first=True)
    y_full = df_full[variable_objetivo]
    model = LinearRegression()
    model.fit(X_full, y_full)
    X_miss = pd.get_dummies(df_miss[variables_predictoras], drop_first=True)
    X_miss = X_miss.reindex(columns=X_full.columns, fill_value=0)
    preds = model.predict(X_miss)
    df.loc[df[variable_objetivo].isna(), variable_objetivo] = preds
    return df
df = imputar_categorica(df, variable_objetivo='voto_anterior', variables_predictoras=['edad', 'sexo', 'estrato', 'nivel_educativo'])
df = imputar_categorica(df, variable_objetivo='voto', variables_predictoras=['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto_anterior'])
df = imputar_numerica(df, variable_objetivo='imagen_del_candidato', variables_predictoras=['edad', 'sexo', 'estrato', 'nivel_educativo', 'voto', 'voto_anterior'])
df['imagen_del_candidato'] = df['imagen_del_candidato'].clip(lower=0, upper=100)
df

# %%
#Sexto Paso: definir la ventana
df = df.sort_values('fecha')
df['Ventana_D'] = df['fecha']
df['Ventana_S'] = df['fecha'].dt.to_period('W')
df['Ventana_M'] = df['fecha'].dt.to_period('M')

# %%
#Séptimo Paso: elegir tipo de ponderación
df['peso_d'] = 1 #a priori todos toman peso = 1
df['peso_s'] = 1
df['peso_m'] = 1
df['edad_cat'] = pd.cut( #categorizar edades
    df['edad'],
    bins=[15, 29, 44, 59, 120],
    labels=['16-29', '30-44', '45-59', '60+']
)
TARGETS_PREDETERMINADOS = { #target predeterminado
    'sexo': {'Femenino': 0.53, 'Masculino': 0.47},
    'edad_cat': {'16-29': 0.29, '30-44': 0.29, '45-59':0.21, '60+': 0.21},
    'region': {'Región Pampeana': 0.68,'Región NOA': 0.14,'Región NEA': 0.08,'Región Cuyo': 0.07,'Región Patagonia': 0.06},
    'nivel_educativo': {'primaria': 0.29, 'secundaria': 0.44, 'terciario': 0.12, 'universitario': 0.13, 'posgrado': 0.02,}
}
def leer_targets_desde_csv(path): #opcion por si se quiere aplicar otros parametros de otra poblacion
    """
    Esperamos un CSV con columnas:
    variable, categoria, proporcion
    Ej:
    sexo,femenino,0.53
    sexo,masculino,0.47
    estrato,Buenos Aires,0.38
    (tenga en cuenta que las categorías deben matchear con las de su df cargado)
    """
    df_t = pd.read_csv(path)
    targets = {}
    for var in df_t['variable'].unique():
        sub = df_t[df_t['variable'] == var]
        targets[var] = dict(zip(sub['categoria'], sub['proporcion']))
    return targets
def elegir_targets(): #seleccionamos tipo de target
    print("Opciones de parámetros para elegir:")
    print("  N - Nacional (usa targets predefinidos)")
    print("  A - Archivo externo (CSV con targets)")
    opcion = input("Elegí N o A: ").strip().lower()
    if opcion == 'n':
        print("Usando targets NACIONALES predefinidos.")
        return TARGETS_PREDETERMINADOS
    elif opcion == 'a':
        path = input("Ruta al archivo CSV con targets: ").strip()
        print(f"Leyendo targets desde {path} ...")
        return leer_targets_desde_csv(path)
    else:
        print("Opción inválida, uso nacional por defecto.")
        return TARGETS_PREDETERMINADOS
targets = elegir_targets()
for var in targets.keys(): #nos aseguramos de que esté todo en string para que no haya inconvenientes
    df[var] = df[var].astype(str)
target_df = prepare_marginal_dist_for_raking(targets) #raking
target_weights = pd.Series(1.0, index=target_df.index, name="w_target")
vars_rake = ['sexo', 'edad_cat', 'region', 'nivel_educativo']
def aplicar_rake_diario(grupo):
    try:
        res = rake(
            sample_df=grupo[vars_rake],
            sample_weights=grupo['peso_d'],
            target_df=target_df,
            target_weights=target_weights,
            variables=vars_rake
        )
        grupo['peso_d'] = res['weight']
    except ValueError as e:
        print(f"No se pudo hacer raking en ventana {grupo.name} ({'peso_d'}): {e}")
        w = grupo['peso_d'].fillna(1)
        grupo['peso_d'] = w / w.mean()
    return grupo
def aplicar_rake_semanal(grupo):
    try:
        res = rake(
            sample_df=grupo[vars_rake],
            sample_weights=grupo['peso_s'],
            target_df=target_df,
            target_weights=target_weights,
            variables=vars_rake
        )
        grupo['peso_s'] = res['weight']
    except ValueError as e:
        print(f"No se pudo hacer raking en ventana {grupo.name} ({'peso_s'}): {e}")
        w = grupo['peso_s'].fillna(1)
        grupo['peso_s'] = w / w.mean()
    return grupo
def aplicar_rake_mensual(grupo):
    try:
        res = rake(
            sample_df=grupo[vars_rake],
            sample_weights=grupo['peso_m'],
            target_df=target_df,
            target_weights=target_weights,
            variables=vars_rake
        )
        grupo['peso_m'] = res['weight']
    except ValueError as e:
        print(f"No se pudo hacer raking en ventana {grupo.name} ({'peso_m'}): {e}")
        w = grupo['peso_m'].fillna(1)
        grupo['peso_m'] = w / w.mean()
    return grupo
df_rake_diario = df.groupby('Ventana_D', group_keys=False).apply(aplicar_rake_diario) #aplicamos raking para cada ventana
df_rake_semana = df.groupby('Ventana_S', group_keys=False).apply(aplicar_rake_semanal)
df_rake_mensual = df.groupby('Ventana_M', group_keys=False).apply(aplicar_rake_mensual)
df['peso_d'] = df_rake_diario['peso_d'] #al df original le damos los pesos calculados
df['peso_s'] = df_rake_semana['peso_s']
df['peso_m'] = df_rake_mensual['peso_m']
def trimming_pesos(pesos, factor=3): #trimming para evitar colapsos
    promedio = np.mean(pesos)
    min_peso = promedio / factor
    max_peso = promedio * factor
    return np.clip(pesos, min_peso, max_peso)
df['peso_d'] = trimming_pesos(df['peso_d']) #aplicamos lo del trimming a los pesos
df['peso_s'] = trimming_pesos(df['peso_s'])
df['peso_m'] = trimming_pesos(df['peso_m'])
def normalizar_pesos(pesos): #normalizacion después del trimming
    return pesos / pesos.sum() * len(pesos)
df['peso_d'] = normalizar_pesos(df['peso_d']) #aplicamos la normalización a los pesos
df['peso_s'] = normalizar_pesos(df['peso_s'])
df['peso_m'] = normalizar_pesos(df['peso_m'])
df

# %%
#Octavo Paso: TRACKING DIARIO
#ventana diaria
def tracking_diario():
    tracking_imagen_diario = (
        df.groupby('Ventana_D')
          .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_d']))
          .reset_index(name='trackeo')
    )
    print(tracking_imagen_diario)
    tracking_imagen_diario.round(1)
    plt.figure(figsize=(10,5))
    plt.plot(tracking_imagen_diario['Ventana_D'], tracking_imagen_diario['trackeo'], marker='o')
    plt.xlabel('Ventana (diaria)', fontsize = 10)
    plt.ylabel('Imagen promedio', fontsize = 10)
    plt.title('Evolución de la imagen del candidato (ventana diaria)', fontsize = 16)
    plt.tight_layout()
    plt.show()
    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_diario = (
        df.groupby('Ventana_D')
          .apply(
              lambda g: pd.Series({
                  f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_d']) * 100
                  for c in candidatos
              })
          )
          .reset_index()
    )
    print(tracking_voto_diario.round(1))
    cols_voto = [col for col in tracking_voto_diario.columns if col.startswith('Vota_')]
    tracking_voto_diario.set_index('Ventana_D')[cols_voto].plot(figsize=(10,5))
    plt.xlabel('Ventana(diaria)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana diaria)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()
    print(
      'Los datos muestran que, durante el período analizado, la media diaria de la imagen del candidato fue:',
      round(tracking_imagen_diario['trackeo'].mean(),1),
      ', con un desvío estándar de:',
      round(tracking_imagen_diario['trackeo'].std(),1),
      ', siendo el valor más bajo que alcanzó:',
      round(tracking_imagen_diario['trackeo'].min(),1),
      'el día:',
      tracking_imagen_diario.loc[tracking_imagen_diario['trackeo'].idxmin()]["Ventana_D"].strftime("%Y-%m-%d"),
      ', y el valor más alto que alcanzó:',
      round(tracking_imagen_diario['trackeo'].max(),1),
      'el día:',
      tracking_imagen_diario.loc[tracking_imagen_diario['trackeo'].idxmax()]["Ventana_D"].strftime("%Y-%m-%d")
    )
    ultimo_relevo = df['Ventana_D'].max()
    mapa_imagen = (
        df.groupby(['Ventana_D', 'estrato'])
            .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_d']))
            .reset_index(name='imagen_estratificada')
    )
    mapa_imagen_ultima = mapa_imagen[mapa_imagen['Ventana_D'] == ultimo_relevo]
    provincias_gdf = gpd.read_file("C:/Users/charo/Downloads/provincias/provincias.shp", encoding="utf-8")
    provincias_gdf.rename(columns={'iso_nombre': 'estrato'}, inplace=True)
    provincias_gdf['estrato'] = provincias_gdf['estrato'].astype(str).str.strip().str.lower()
    gdf_mapa_imagen = provincias_gdf.merge(
        mapa_imagen_ultima[['estrato', 'imagen_estratificada']],
        on='estrato',
        how='left'
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_mapa_imagen.plot(
        column='imagen_estratificada',
        cmap='RdYlGn',  
        legend=True,
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        missing_kwds={
        "color": "lightgrey",
        "edgecolor": "black",
        "hatch": "///",
        }
        )
    ax.set_title(f"Imagen promedio del candidato por provincia\nÚltimo día: {ultimo_relevo}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# %%
#Noveno paso: TRACKING SEMANAL
#ventana semanal
def tracking_semanal():
    tracking_imagen_semanal = (
        df.groupby('Ventana_S')
          .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_s']))
          .reset_index(name='trackeo')
    )
    print(tracking_imagen_semanal.round(1))
    plt.figure(figsize=(10,5))
    tracking_imagen_semanal.set_index('Ventana_S')['trackeo'].plot(marker='o')
    plt.xlabel('Ventana (semanal)', fontsize = 10)
    plt.ylabel('Imagen promedio', fontsize = 10)
    plt.title('Evolución de la imagen del candidato (ventana semanal)', fontsize = 16)
    plt.tight_layout()
    plt.show()
    candidatos = df['voto'].unique().tolist()
    for c in candidatos:
        df[f'vota_{c}'] = (df['voto'] == c).astype(int)
    tracking_voto_semanal = (
        df.groupby('Ventana_S')
          .apply(
              lambda g: pd.Series({
                  f"Vota_{c}": np.average(g[f'vota_{c}'], weights=g['peso_s']) * 100
                  for c in candidatos
              })
          )
          .reset_index()
    )
    print(tracking_voto_semanal.round(1))
    cols_voto = [col for col in tracking_voto_semanal.columns if col.startswith('Vota_')]
    tracking_voto_semanal.set_index('Ventana_S')[cols_voto].plot(figsize=(10,5))
    plt.xlabel('Ventana(semanal)', fontsize=10)
    plt.ylabel('Intención de voto (%)', fontsize=10)
    plt.title('Tracking de intención de voto (ventana semanal)', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(title="Candidato")
    plt.tight_layout()
    plt.show()
    print(
      'Los datos muestran que, durante el período analizado, la media semanal de la imagen del candidato fue:',
      round(tracking_imagen_semanal['trackeo'].mean(),1),
      ', con un desvío estándar de:',
      round(tracking_imagen_semanal['trackeo'].std(),1),
      ', siendo el valor más bajo que alcanzó:',
      round(tracking_imagen_semanal['trackeo'].min(),1),
      'la semana:',
      tracking_imagen_semanal.loc[tracking_imagen_semanal['trackeo'].idxmin()]["Ventana_S"].strftime("%Y-%m-%d"),
      ', y el valor más alto que alcanzó:',
      round(tracking_imagen_semanal['trackeo'].max(),1),
      'la semana:',
      tracking_imagen_semanal.loc[tracking_imagen_semanal['trackeo'].idxmax()]["Ventana_S"].strftime("%Y-%m-%d")
    )
    ultimo_relevo = df['Ventana_S'].max()
    mapa_imagen = (
        df.groupby(['Ventana_S', 'estrato'])
            .apply(lambda g: np.average(g['imagen_del_candidato'], weights=g['peso_s']))
            .reset_index(name='imagen_estratificada')
    )
    mapa_imagen_ultima = mapa_imagen[mapa_imagen['Ventana_S'] == ultimo_relevo]
    provincias_gdf = gpd.read_file("C:/Users/charo/Downloads/provincias/provincias.shp", encoding="utf-8")
    provincias_gdf.rename(columns={'iso_nombre': 'estrato'}, inplace=True)
    provincias_gdf['estrato'] = provincias_gdf['estrato'].astype(str).str.strip().str.lower()
    gdf_mapa_imagen = provincias_gdf.merge(
        mapa_imagen_ultima[['estrato', 'imagen_estratificada']],
        on='estrato',
        how='left'
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_mapa_imagen.plot(
        column='imagen_estratificada',
        cmap='RdYlGn',  
        legend=True,
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        missing_kwds={
        "color": "lightgrey",
        "edgecolor": "black",
        "hatch": "///",
        }
        )
    ax.set_title(f"Imagen promedio del candidato por provincia\nÚltima semana: {ultimo_relevo}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

