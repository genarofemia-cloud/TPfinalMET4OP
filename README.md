# 📊 Tracking Electoral – Metodología en Opinión Pública (Olego — 2° Cuatrimestre 2025)

Este proyecto fue desarrollado como trabajo final para la materia Metodología en Opinión Pública, dictada por la cátedra Olego (Ciencia Política – UBA) durante el segundo cuatrimestre de 2025.

## 📌 Objetivo 

Diseñar un algoritmo en Python para realizar un tracking electoral, analizando la evolución de la imagen e intención de voto de un candidato a lo largo del tiempo a partir de datos de encuestas, utilizando las herramientas metodológicas aprendidas en la materia.

## 📌 Descripción general

El proyecto implementa un pipeline completo para el análisis de encuestas electorales que incluye:
* Limpieza y normalización del dataset.
* Imputación de valores faltantes mediante regresiones logística y lineal.
* Ponderación por raking a partir de targets censales o personalizados.
* Construcción de trackings diario, semanal y mensual de:
  * imagen del candidato
  * intención de voto
* Generación de mapas provinciales de:
  * imagen del candidato
  * intención de voto por candidato
* Cálculo de intervalos de confianza (95%).
* Realización de un test de hipótesis sobre el cambio temporal de la imagen.

## 🧩 Estructura del pipeline (14 bloques)
A continuación, un resumen limpio y claro de lo que hace cada bloque del código:

### 1️⃣ Importación de librerías
Se cargan todas las herramientas necesarias para manejo de datos, gráficos, modelado estadístico, geoprocesamiento y ponderación.

### 2️⃣ Carga del archivo de encuesta
El script admite CSV, Excel, JSON o TXT. Si no es CSV, se convierte automáticamente.
Además:
* Verifica que existan todas las columnas requeridas.
* Estandariza nombres de columnas.
* Convierte la variable fecha al formato correcto.

### 3️⃣ Normalización de variables 
Para evitar inconsistencias:
* Las provincias se asignan a regiones (NOA, NEA, Cuyo, etc.).
* El nivel educativo se estandariza a categorías ordenadas (primaria, secundaria, etc.).
* “Sin estudios” se agrupa dentro de primaria para evitar problemas de ponderación.
* Variables como sexo y estrato se uniformizan en minúsculas y sin espacios.

### 4️⃣ Limpieza de valores faltantes (variables independientes)
El script elimina/corrige casos imposibles o incompletos:
* Se descartan encuestas donde falten ambas variables clave: imagen y voto.
* Se eliminan menores de 16 años.
* Se eliminan los registros duplicados.
* Se descartan filas sin información esencial (fecha, sexo, edad, estrato).
* Se normaliza nivel educativo y se rellena “integrantes_hogar” cuando falta.

### 5️⃣ Imputación de las variables dependientes
Para no perder casos, se imputan:
* voto_anterior → regresión logística
* voto → regresión logística
* imagen_del_candidato → regresión lineal
  
Antes, se evalúa el desempeño de cada modelo:

a) Modelo para voto_anterior
* Predictores: edad, sexo, región, nivel educativo
* Regresión logística multinomial
* Métricas: Accuracy, classification report, matriz de confusión

b) Modelo para voto
* Predictores: edad, sexo, región, nivel educativo, voto_anterior
* Mismo procedimiento que el anterior

c) Modelo para imagen_del_candidato
* Regresión lineal
* Métricas: MAE, RMSE, R²

### 6️⃣ Definición de ventanas temporales
Se crean tres ventanas temporales:
* Ventana_D → día
* Ventana_S → semana
* Ventana_M → mes
  
Sirven para generar trackings a distinta escala.

### 7️⃣ Ponderación: raking + trimming + normalización
Se toma la base ya limpia e imputada y se le asignan pesos muestrales para que la encuesta parezca tener la misma composición que la población real.
* Se definen targets poblacionales (sexo, edad, región y nivel educativo), ya sea usando valores nacionales predefinidos o de un CSV externo.
* Se aplica raking (ajuste iterativo de proporciones) dentro de cada ventana de tiempo
* Se evitan pesos extremos mediante un trimming suave.
* Se normalizan los pesos para que mantengan una escala coherente.

### 8️⃣, 9️⃣ y 🔟 Trackings (diario, semanal, mensual)
Para cada tipo de tracking se define una función que:
* Calcula la imagen promedio del candidato por ventana.
* Grafica la serie temporal.
* Calcula la intención de voto para cada candidato (% ponderado).
* Grafica la evolución de intención de voto.
* Informa media, desvío, mínimo/máximo y fechas.
* Genera un mapa provincial con la imagen promedio de la última ventana.

### 1️⃣1️⃣ Elección del tipo de tracking
El usuario elige si quiere tracking:
* Diario
* Semanal
* Mensual

Y el programa ejecuta automáticamente el módulo correspondiente.

### 1️⃣2️⃣ Mapa de intención de voto por candidato
El usuario puede elegir qué candidato quiere analizar, y el sistema genera un mapa provincial con su intención de voto en la última ventana disponible (diaria, semanal o mensual).

### 1️⃣3️⃣ Intervalos de confianza (95%)
Para cada ventana se calcula:
* Promedio ponderado
* n efectivo
* Margen de error
* Intervalo [LI ; LS]

Se hace para:
* Imagen del candidato
* Intención de voto

### 1️⃣4️⃣ Test de hipótesis: cambio en la imagen del candidato
Se compara la imagen entre:
* la primera ventana semanal
* la última ventana semanal

Dependiendo del tamaño de la ventana:
* ≥30 casos → Test paramétrico
* <30 casos → Test no paramétrico

Si la muestra es lo suficientemente grande (≥30 casos)
* homocedasticidad → test de t
* heterocedasticidad → test t de Welch

Si la muestra es pequeña (<30 casos)
* homocedasticidad → test de Mann-Whitney

El test determina si el cambio es estadísticamente significativo.

## ⚙️ Requisitos
### 🔧 Python
Python 3.9 o superior
### 📦 Librerías
* Pandas
* NumPy
* SciPy
* Matplotlib
* Scikit-learn
* GeoPandas
* balance 
### 📁 Archivos necesarios
* Encuesta (CSV recomendado)
* Shapefile de provincias (.shp, .shx, .dbf, etc.)
* (Opcional) CSV de targets de raking con columnas:
  * variable
  * categoria
  * proporción

## ▶️ ¿Cómo ejecutar el script?
1) Clonar el repositorio o copiar el .py.
2) Ajustar las rutas (encuesta, shapefile, etc.).
3) Instalar dependencias (pip o conda).
4) Ejecutar: python tracking_electoral.py
5) Seguir los pasos interactivos:
   * Elegir tipo de targets (N o A)
   * Elegir tipo de tracking (D, S, M)
   * Escribir el candidato para el mapa final

## ✒️ Autores 

**Charo Sanchez Inda**

**Genaro Femia**

**Malena Vera**

**María Jose Perez** 

