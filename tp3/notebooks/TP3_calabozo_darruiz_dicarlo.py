# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3.12 (TP MD)
#     language: python
#     name: tp-mineria
# ---

# %% [markdown]
# # Trabajo Práctico N° 3: SVM y Random Forest
#
# | | |
# |---|---|
# | **Año** | 2026 |
# | **Materia** | Minería de Datos, Tecnicatura Universitaria en Inteligencia Artificial |
# | **Integrantes** | Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo |
#
# **Objetivo:** recomendar un cultivo (`Clase`, 10 clases) a partir de 16 parámetros de suelo y clima (`crop_recommendation_geo.csv`), con los modelos de las unidades 5 y 6: SVM con kernel lineal, SVM con kernel gaussiano (RBF) y Random Forest.
#
# **Contenido:** 1. Preparación de datos, 2. SVM con kernel lineal, 3. SVM con kernel gaussiano (RBF), 4. Random Forest, 5. Comparación de modelos y 6. Conclusiones.
#
# Como pide la consigna, todos los modelos se evalúan con la misma validación cruzada estratificada de 5 folds sobre todo el conjunto limpio, con `random_state = 13`. Cuando reutilizamos o adaptamos código de los notebooks de clase, el comentario de la celda dice de cuál sale.

# %%
import os

# Análisis exploratorio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Preparación de los datos y validación cruzada
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Métricas
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# random_state que pide la consigna para los modelos y la validación cruzada
SEMILLA = 13

# Ruta relativa a la carpeta del notebook (tp3/notebooks/)
RUTA_DATOS = "../data/crop_recommendation_geo.csv"

# %% [markdown]
# ---
# ## 1. Preparación de datos
#
# ### 1.1 Estructura y valores faltantes

# %%
# El archivo usa ";" como separador y coma decimal. La variable objetivo se llama "clase" en el archivo: la
# renombramos "Clase", como en la consigna
datos = pd.read_csv(RUTA_DATOS, sep=";", decimal=",")
datos = datos.rename(columns={"clase": "Clase"})
datos.head()

# %%
# Tipo de cada columna y cantidad de valores no nulos
datos.info()

# %%
# Filas con algún valor faltante
datos[datos.isna().any(axis=1)]

# %% [markdown]
# Hay 50004 muestras (una menos que las 50005 que indica la consigna) con 16 variables predictoras numéricas y la variable objetivo, `Clase`, que es texto. Los únicos faltantes son 3, y están en la misma fila, la 25085 (`cafe`), a la que le faltan `ph`, `cec` y `humedad`. La eliminamos: imputarla con la media o la mediana inventaría 3 de sus 16 valores para recuperar una sola fila de más de 50.000.

# %% [markdown]
# ### 1.2 Registros duplicados

# %%
# keep=False marca todas las filas que tienen una copia; duplicated() marca solo las copias de más
print(f"Filas que tienen al menos una copia: {datos.duplicated(keep=False).sum()}")
print(f"Copias de más: {datos.duplicated().sum()}")

# Cultivo de las copias
datos[datos.duplicated()]["Clase"].value_counts()

# %%
# Filas con los mismos 16 atributos y distinta clase: se repiten si no miramos la clase, pero no son copias
ATRIBUTOS = list(datos.columns.drop("Clase"))
repetidas_sin_clase = datos.duplicated(subset=ATRIBUTOS, keep=False)
copias = datos.duplicated(keep=False)
datos[repetidas_sin_clase & ~copias].sort_values(ATRIBUTOS)

# %% [markdown]
# Hay 198 filas repetidas, y cada una aparece exactamente dos veces (396 filas forman 198 pares). Casi todas las copias son de `lenteja` (109) y de `manzana` (55). Las eliminamos y nos quedamos con la primera aparición de cada fila: si quedaran, en la validación cruzada una copia podría caer en los folds de entrenamiento y la otra en el de evaluación, y el modelo se evaluaría con filas que ya vio. Además, cada una pesaría el doble en el entrenamiento.
#
# Aparte, 7 pares de filas tienen los mismos 16 atributos y distinta clase: 6 de `garbanzo` y `lenteja`, y 1 de `maiz` y `manzana`. No son copias, porque cambia la clase: es el mismo suelo y el mismo clima con dos cultivos recomendados, algo posible si los dos se adaptan a ese lugar. Las conservamos, porque nada indica que una de las dos clases sea un error, y sacar una sería elegir al azar. Ningún modelo puede acertar las dos filas de un par, así que son al menos 7 errores seguros, menos del 0,02 % de los datos.

# %% [markdown]
# ### 1.3 Distribución de las variables predictoras
#
# Los histogramas sirven para ver la forma y el rango de cada variable, y si hay valores aislados, lejos del resto. Esperamos que las variables que miden cantidades en el suelo (`N`, `P`, `soc`, `exch_bases` y `cec`), la lluvia y la elevación sean asimétricas, con muchos lugares con valores bajos y unos pocos con valores muy altos. El pH, las fracciones de arcilla y arena, la humedad y las temperaturas tienen rangos acotados, y esperamos distribuciones más simétricas.

# %%
# Histograma de cada variable en una grilla de 4 × 4, como en U5_SVMClassifier_wine.ipynb (sección "Distribución
# de variables"), pero sin separar por clase: con 10 cultivos superpuestos no se leería
fig, axes = plt.subplots(4, 4, figsize=(18, 13))
for ax, columna in zip(axes.flatten(), ATRIBUTOS):
    sns.histplot(data=datos, x=columna, kde=True, element="step", ax=ax)
    ax.set(ylabel="")
fig.suptitle("Distribución de las variables predictoras (datos originales)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se cumple en parte. `N`, `P`, `soc`, `lluvia` y `elevacion` son asimétricas a la derecha, con colas largas, como esperábamos, y `cec` también, aunque menos; `exch_bases` tiene varios picos. Las variables de rango acotado tampoco son simétricas: `ph` tiene dos picos, cerca de 5,5 y de 7, `humedad` también tiene dos, y `temperatura` y `temp_min_mensual` se acumulan en los valores altos, con una cola larga hacia el frío. Con 10 cultivos de climas distintos, esos picos pueden ser grupos de cultivos; lo vemos separado por cultivo en 1.4.
#
# Lo que más llama la atención son cinco variables cuyo eje se estira mucho más allá de los datos: `ph` llega a 25, cuando la escala del pH va de 0 a 14; `arcilla` y `humedad`, que son porcentajes, a 142 y 148; y `temperatura` y `temp_max_mensual`, a 44 y 53 grados, cuando casi todos los valores están por debajo de 30 y de 40. Son valores aislados que revisamos en 1.4.

# %% [markdown]
# ### 1.4 Valores atípicos
#
# Si la cátedra cargó valores erróneos, deberían aparecer como puntos aislados y, en algunos casos, fuera del rango posible de la variable: el pH va de 0 a 14, y la humedad relativa y las fracciones de arcilla y arena, de 0 a 100 %. En cambio, en las variables asimétricas de 1.3 esperamos que la regla del rango intercuartílico marque muchos atípicos que son solo la cola de la distribución. Los diagramas de caja por cultivo sirven para distinguir los dos casos: un error queda lejos de los valores de su cultivo, y una cola es continua y propia de algunos cultivos.

# %%
# Función de U3_Clustering.ipynb (sección "Remocion de Outliers"), la misma que usamos en el TP1 y el TP2:
# devuelve solo las filas que quedan dentro de los límites del rango intercuartílico
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


# Adaptada del "Conteo de atípicos por columna" de U4_DecisionTreeRegresion.ipynb, como en el TP2: cuenta con
# remove_outliers_iqr y devuelve la tabla en lugar de imprimirla. Las filas con faltantes no se cuentan
def contar_atipicos(df, columnas):
    """Cantidad y porcentaje de atípicos de cada columna, según la regla del rango intercuartílico."""
    resultados = []
    for col in columnas:
        cantidad = df[col].notna().sum() - len(remove_outliers_iqr(df, col))
        resultados.append({"Variable": col, "Cantidad": cantidad, "Porcentaje": round(100 * cantidad / len(df), 1)})
    return pd.DataFrame(resultados)


contar_atipicos(datos, ATRIBUTOS)

# %%
# Diagrama de caja de cada variable separado por cultivo, como el box plot por clase de RF_heartBinomial.ipynb (U6)
ORDEN_CLASES = sorted(datos["Clase"].unique())
fig, axes = plt.subplots(4, 4, figsize=(18, 17))
for ax, columna in zip(axes.flatten(), ATRIBUTOS):
    sns.boxplot(data=datos, x=columna, y="Clase", order=ORDEN_CLASES, fliersize=1, ax=ax)
    ax.set(ylabel="")
plt.tight_layout()
fig.suptitle("Valores atípicos de cada variable según el cultivo (datos originales)", y=1.01)
plt.show()

# %% [markdown]
# Se cumple. La regla marca muchos atípicos en las variables asimétricas: 3665 en `soc` (7,3 %), 3453 en `P` (6,9 %), 2825 en `elevacion` (5,6 %), 2162 en `cec` y 1483 en `N`. En los diagramas de caja no son puntos aislados sino colas continuas, más largas en algunos cultivos: los valores altos de `P` son sobre todo de `maiz` y `arroz`; los de `soc`, de `coco`, `arroz` y `banana`; y los de `elevacion`, de `banana`, `cafe` y `manzana`. Son valores posibles, propios de los lugares donde se dan esos cultivos, y los conservamos: sacarlos borraría parte de lo que distingue a esas clases, y con miles de atípicos por variable perderíamos muchas filas.
#
# Las excepciones son unos pocos puntos aislados, muy lejos de los demás valores de su cultivo: dos en `ph` (uno de `cafe` y otro de `manzana`) y uno en cada una de `arcilla`, `humedad`, `temperatura`, `temp_max_mensual` y `radiacion_solar`, todos de `manzana`. Los buscamos con los rangos posibles.
#
# Los diagramas muestran también cómo cambia cada variable entre cultivos. Ninguna separa sola a los 10, pero varias separan grupos. Con las temperaturas, `lenteja`, `manzana` y `maiz` son los de clima más frío, y `coco`, el más cálido y con menos variación. Con el pH, `algodon`, `garbanzo`, `lenteja` y `manzana` están en suelos neutros, cerca de 7, y `banana`, `cafe` y `coco`, en suelos ácidos, cerca de 5,5. `banana` y `cafe` tienen distribuciones muy parecidas en casi todas las variables, y `maiz` es el cultivo con los rangos más amplios.

# %%
# Filas con valores imposibles: pH mayor a 14, o humedad relativa, arcilla o arena por encima del 100 %
fuera_de_rango = datos[(datos["ph"] > 14) | (datos["humedad"] > 100) | (datos["arcilla"] > 100)
                       | (datos["arena"] > 100)]
fuera_de_rango

# %% [markdown]
# Son dos filas. La 25111 (`cafe`) tiene un pH de 25,3. La 30012 (`manzana`) tiene seis valores imposibles o aislados a la vez: pH de 18, arcilla de 142 %, humedad de 148 %, temperatura media de 44,6 grados, media del mes más cálido de 53 grados y radiación solar de 6,27, que son los puntos aislados de `temperatura`, `temp_max_mensual` y `radiacion_solar` en los diagramas de caja. Las tratamos como errores de carga y las eliminamos. No corregimos los valores con la mediana: en la 30012 habría que reemplazar más de un tercio de la fila, y si una fila tiene un valor imposible, sus otros valores tampoco son confiables.

# %% [markdown]
# ### 1.5 Limpieza

# %%
# Sacamos la fila con faltantes, las copias de las filas duplicadas y las filas con valores fuera de rango
datos_limpios = datos.dropna()
datos_limpios = datos_limpios.drop_duplicates()
datos_limpios = datos_limpios.drop(fuera_de_rango.index)
print(f"Quedan {len(datos_limpios)} de {len(datos)} filas")
datos_limpios.describe().round(2)

# %% [markdown]
# Quedan 49803 filas: sacamos 201 (la fila con faltantes, las 198 copias y las 2 filas con valores imposibles), y ninguna columna tiene faltantes. Los máximos quedan en línea con el resto de cada distribución: `ph` llega a 8,95, `arcilla` a 68,63, `humedad` a 97,37, `temperatura` a 30,38 y `temp_max_mensual` a 38,22, y el mínimo de `radiacion_solar` sube a 7,58. El mínimo de `exch_bases` pasa a ser 1: el 0,1 de la fila 25111 era el único valor por debajo de 1, otra señal de que esa fila estaba mal cargada.

# %% [markdown]
# ### 1.6 Distribución de clases

# %%
# Cantidad y porcentaje de muestras de cada cultivo, como target_balance de los notebooks de la U4 y la U5
pd.DataFrame({"Cantidad": datos_limpios["Clase"].value_counts(),
              "Porcentaje": (datos_limpios["Clase"].value_counts(normalize=True) * 100).round(2)})

# %% [markdown]
# Las clases quedan casi balanceadas, de 4891 muestras (`lenteja`, 9,82 %) a 5000 (`banana`, `cafe` y `coco`, 10,04 %). Es el leve desbalance que menciona la consigna, y viene de la limpieza: `lenteja` y `manzana`, las dos clases con más copias (1.2), son las que quedan con menos muestras.

# %% [markdown]
# ### 1.7 Correlaciones
#
# La matriz de correlación sirve para ver qué variables llevan información parecida. Esperamos correlaciones fuertes entre las tres temperaturas y entre la humedad y la lluvia; una negativa entre arcilla y arena, que son fracciones del mismo suelo (si hay más de una, queda menos lugar para la otra); y una negativa entre la elevación y las temperaturas, porque en altura hace más frío.

# %%
# Matriz de correlación de Pearson, con el mapa de calor de U5_SVMClassifier_wine.ipynb
correlaciones = datos_limpios[ATRIBUTOS].corr()
plt.figure(figsize=(13, 11))
sns.heatmap(correlaciones, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
            annot=True, fmt=".2f", annot_kws={"size": 8}, square=True)
plt.title("Correlación de Pearson entre las variables predictoras (datos limpios)")
plt.show()

# %% [markdown]
# Se cumple. La correlación más alta es la de `temperatura` con `temp_min_mensual` (0,95), y con `temp_max_mensual` es de 0,75; `humedad` y `lluvia` tienen 0,67, y `arcilla` y `arena`, −0,65. La elevación tiene −0,65 con `temp_max_mensual`, pero solo −0,43 con la temperatura media. Lo que no esperábamos es la relación del pH con el clima: baja con la humedad (−0,79) y con la lluvia (−0,78), es decir, los suelos de los lugares más húmedos y lluviosos son más ácidos, y sube con las bases intercambiables (0,72).
#
# Salvo `temperatura` y `temp_min_mensual` (0,95), ningún par está tan correlacionado como para que una variable repita a la otra. Conservamos las 16 variables que plantea la consigna.

# %% [markdown]
# ### 1.8 Codificación, estandarización y folds

# %%
# LabelEncoder, como pide la consigna: cada cultivo pasa a un número de 0 a 9, en orden alfabético
codificador = LabelEncoder()
y = codificador.fit_transform(datos_limpios["Clase"])
X = datos_limpios[ATRIBUTOS]
pd.DataFrame({"Cultivo": codificador.classes_}).rename_axis("Código").T

# %%
# Los 5 folds estratificados que usan todos los modelos. El archivo viene ordenado por cultivo: shuffle=True mezcla
# las filas antes de repartirlas, y es lo que hace que random_state tenga efecto
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

# Muestras de cada cultivo en cada fold
conteos = {}
for numero, (indices_train, indices_test) in enumerate(FOLDS.split(X, y), start=1):
    conteos[f"Fold {numero}"] = pd.Series(codificador.classes_[y[indices_test]]).value_counts()
conteos = pd.DataFrame(conteos)
conteos.loc["Total"] = conteos.sum()
conteos

# %% [markdown]
# Cada fold tiene 9960 o 9961 filas, con entre 978 y 1000 de cada cultivo: la estratificación mantiene en los cinco las proporciones de 1.6. Usamos estos mismos folds para todos los modelos, así las métricas de los tres salen de exactamente las mismas divisiones.


# %%
# Cada modelo es un pipeline que estandariza y después clasifica, como el make_pipeline de Unidad4.py. En cada
# vuelta de la validación cruzada, el escalador aprende la media y el desvío solo de los 4 folds de entrenamiento
def crear_modelo(clasificador):
    """Pipeline que estandariza los 16 atributos con StandardScaler y después aplica el clasificador."""
    return make_pipeline(StandardScaler(), clasificador)


# Así quedan los atributos del entrenamiento del primer fold después de estandarizar
indices_train, indices_test = next(FOLDS.split(X, y))
escalador = StandardScaler().set_output(transform="pandas")
escalador.fit_transform(X.iloc[indices_train]).describe().T[["mean", "std", "min", "max"]].round(2)

# %% [markdown]
# La estandarización va dentro de cada modelo: en cada una de las 5 vueltas de la validación cruzada, el escalador se ajusta con los 4 folds de entrenamiento y se aplica al fold de evaluación, que así no interviene en la media ni en el desvío (apunte de la Unidad 4, «Estandarización vs Separación de datos»). En el entrenamiento del primer fold, todas las variables quedan con media 0 y desvío 1. Las colas largas de 1.3 se ven en los máximos: `P` llega a 15,32 desvíos de la media, `N` a 13,82 y `soc` a 11,25, mientras que `ph` y `humedad` no pasan de 2,5.

# %% [markdown]
# ---
# ## 2. SVM con kernel lineal
#
# ### 2.1 Búsqueda del parámetro C
#
# En 1.4 vimos que algunos cultivos se distinguen por una o dos variables, como `coco` y `lenteja` por la temperatura, pero que otros se superponen en casi todas, como `banana` y `cafe`. Esperamos que el SVM lineal acierte los primeros y se confunda con los segundos, y que la exactitud suba con C hasta estabilizarse. Además, el apunte de la Unidad 5 («SVM: Costo Computacional y el parámetro C») dice que lo que más influye en el tiempo de entrenamiento de un SVM es la cantidad de observaciones: con unas 40.000 en cada entrenamiento, esperamos tiempos largos.
#
# Probamos 6 valores de C en escala logarítmica, de 0,001 a 100, cada uno 10 veces el anterior: los 5 del ejemplo de la consigna y 0,001, para ver qué pasa con un C todavía más chico. Cada valor se entrena y se evalúa 5 veces, una con cada fold como evaluación y los otros 4 como entrenamiento: son 6 × 5 = 30 entrenamientos. Antes de ver los resultados fijamos el criterio para elegir, que usamos también con los otros dos modelos: la configuración con mayor exactitud media en los 5 folds, porque con las clases casi balanceadas (1.6) la exactitud pesa a todos los cultivos casi por igual. El desvío entre folds nos dice si la diferencia con las demás configuraciones es clara.

# %%
# Las tres métricas que pide la consigna, con promedio weighted en la precisión y la exhaustividad, como
# pd_metricas_clasifier de gas_detection.ipynb (U6)
METRICAS = {"Precisión": "precision_weighted", "Exhaustividad": "recall_weighted", "Exactitud": "accuracy"}


# GridSearchCV como en U5_SVMClassifier_wine.ipynb (sección "Optimización del modelo"), con los 5 folds de 1.8, las
# tres métricas y n_jobs=-1 para entrenar en paralelo, como en Unidad5.py. refit=False: no reentrena la mejor
# combinación con todos los datos, porque eso lo hacemos solo donde hace falta (4.3 y 5.2).
# Como cada búsqueda tarda varios minutos, la primera vez guardamos su tabla de resultados en ../data/ y las
# siguientes la leemos de ahí. Si cambia la grilla o la preparación de los datos, hay que borrar ese archivo
def buscar_hiperparametros(nombre, clasificador, grilla):
    """Evalúa cada combinación de la grilla en los 5 folds y devuelve la tabla de resultados de GridSearchCV."""
    ruta = f"../data/busqueda_{nombre}.pkl"
    if not os.path.exists(ruta):
        busqueda = GridSearchCV(crear_modelo(clasificador), grilla, cv=FOLDS, scoring=METRICAS, refit=False,
                                n_jobs=-1)
        busqueda.fit(X, y)
        pd.DataFrame(busqueda.cv_results_).to_pickle(ruta)
    return pd.read_pickle(ruta)


def tabla_resultados(resultados, hiperparametros):
    """Media y desvío de cada métrica en los 5 folds, y tiempo medio de entrenamiento, de cada combinación."""
    tabla = pd.DataFrame()
    for parametro, nombre in hiperparametros.items():
        tabla[nombre] = resultados[f"param_{parametro}"]
    for metrica in METRICAS:
        tabla[f"{metrica} media"] = resultados[f"mean_test_{metrica}"].round(4)
        tabla[f"{metrica} desvío"] = resultados[f"std_test_{metrica}"].round(4)
    tabla["Entrenamiento por fold (s)"] = resultados["mean_fit_time"].round(1)
    return tabla


def mejor_configuracion(resultados):
    """Fila de la combinación con mayor exactitud media en los 5 folds (la primera, si hay empate)."""
    return resultados.loc[resultados["mean_test_Exactitud"].idxmax()]


# %%
# En el pipeline, cada hiperparámetro lleva adelante el nombre de su paso: svc__C es el C del paso svc.
# random_state no cambia nada en un SVC (solo se usa con probability=True), pero lo fijamos como pide la consigna
GRILLA_LINEAL = {"svc__C": [0.001, 0.01, 0.1, 1, 10, 100]}
resultados_lineal = buscar_hiperparametros("svm_lineal", SVC(kernel="linear", random_state=SEMILLA), GRILLA_LINEAL)
tabla_lineal = tabla_resultados(resultados_lineal, {"svc__C": "C"})
tabla_lineal

# %% [markdown]
# Las columnas de exhaustividad y de exactitud son iguales, y lo van a ser en todos los modelos: la exhaustividad weighted promedia la de cada cultivo pesada por su cantidad de muestras, y esa cuenta da siempre la proporción total de aciertos. La precisión weighted sí puede diferir, aunque acá queda muy cerca de la exactitud. Por eso, en los gráficos usamos solo la exactitud. Para ver la tendencia, graficamos la exactitud y el tiempo de entrenamiento según C.

# %%
# Exactitud media con el desvío entre folds como barra de error, y tiempo de entrenamiento, según C
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].errorbar(tabla_lineal["C"], tabla_lineal["Exactitud media"], yerr=tabla_lineal["Exactitud desvío"],
                 marker="o", capsize=4)
axes[0].set(xscale="log", xlabel="C", ylabel="Exactitud media en los 5 folds", title="Exactitud")
axes[1].plot(tabla_lineal["C"], tabla_lineal["Entrenamiento por fold (s)"], marker="o", color="tab:orange")
axes[1].set(xscale="log", xlabel="C", ylabel="Segundos", title="Tiempo de entrenamiento por fold")
fig.suptitle("SVM con kernel lineal según C")
plt.tight_layout()
plt.show()

# %% [markdown]
# La exactitud se comporta como esperábamos: sube de 0,5657 con C = 0,001 a 0,6219 con C = 1, y desde ahí casi no cambia (0,6230 con C = 10 y 0,6232 con C = 100), con diferencias mucho menores que el desvío entre folds, de alrededor de 0,005. Los tiempos también son largos, como anticipaba el apunte: entre 61 y 98 segundos por fold hasta C = 1, y después se disparan, a 217 segundos con C = 10 y 867 con C = 100, casi nueve veces más que con C = 1 sin ganar exactitud.
#
# Lo que no esperábamos es que la exactitud se estanque tan abajo: el SVM lineal acierta el cultivo en 6 de cada 10 filas. Con 10 cultivos que se superponen en varias variables (1.4), separarlos con hiperplanos no alcanza, y por más que suba C, el modelo no mejora.

# %% [markdown]
# ### 2.2 Mejor configuración

# %%
mejor_lineal = mejor_configuracion(resultados_lineal)
tabla_lineal.loc[[mejor_lineal.name]]

# %% [markdown]
# Elegimos **C = 100**, la de mayor exactitud media (0,6232, con un desvío de 0,0047 entre folds), con el criterio que fijamos en 2.1. La elección no es firme: le gana a C = 10 por 0,0002 y a C = 1 por 0,0013, mucho menos que el desvío entre folds, así que en la práctica los tres valores rinden lo mismo, y C = 100 es el que más tarda en entrenar.

# %% [markdown]
# ---
# ## 3. SVM con kernel gaussiano (RBF)
#
# ### 3.1 Búsqueda de C y gamma
#
# Con el kernel RBF el SVM puede armar fronteras curvas, así que esperamos que supere al lineal, sobre todo en los cultivos que se superponen en varias variables. Para gamma esperamos un máximo en valores intermedios, con la exactitud cayendo en los dos extremos, y que el mejor C dependa del gamma.
#
# Probamos 5 valores de C (de 0,1 a 1000) y 5 de gamma (de 0,001 a 10), en escala logarítmica: los 4 del ejemplo de la consigna para cada uno, más C = 1000 y gamma = 10, para ver si el máximo queda dentro de la grilla y no en un borde. Son 5 × 5 = 25 combinaciones, cada una entrenada y evaluada en los 5 folds: 125 entrenamientos. Además de la tabla, un mapa de calor muestra la interacción entre C y gamma que pide la consigna, junto con el tiempo de entrenamiento de cada combinación.

# %%
GRILLA_RBF = {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [0.001, 0.01, 0.1, 1, 10]}
resultados_rbf = buscar_hiperparametros("svm_rbf", SVC(kernel="rbf", random_state=SEMILLA), GRILLA_RBF)
tabla_rbf = tabla_resultados(resultados_rbf, {"svc__C": "C", "svc__gamma": "gamma"})

# Ordenada de mayor a menor exactitud media
tabla_rbf.sort_values("Exactitud media", ascending=False)


# %%
# Exactitud media y tiempo de entrenamiento de cada par de valores de los dos hiperparámetros, con el heatmap de
# seaborn que usamos para las correlaciones. sort=False mantiene el orden de la grilla en filas y columnas
def graficar_interaccion(tabla, filas, columnas, titulo):
    """Mapas de calor de la exactitud media y del tiempo de entrenamiento por fold de cada combinación."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    exactitud = tabla.pivot_table(index=filas, columns=columnas, values="Exactitud media", sort=False)
    sns.heatmap(exactitud, annot=True, fmt=".3f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Exactitud media en los 5 folds")
    tiempo = tabla.pivot_table(index=filas, columns=columnas, values="Entrenamiento por fold (s)", sort=False)
    sns.heatmap(tiempo, annot=True, fmt=".0f", cmap="Oranges", ax=axes[1])
    axes[1].set_title("Tiempo de entrenamiento por fold (segundos)")
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()


graficar_interaccion(tabla_rbf, "C", "gamma", "SVM con kernel RBF según C y gamma")

# %% [markdown]
# Se cumple. El RBF supera al lineal, como esperábamos: llega a 0,886 de exactitud, contra 0,623. La exactitud tiene su máximo en valores intermedios de gamma y cae en los dos extremos: con gamma = 0,001 no pasa de 0,745, y con gamma = 10 se derrumba a entre 0,617 y 0,643 para C de 1 en adelante (y a 0,206 con C = 0,1). El mejor C depende de gamma, y las mejores combinaciones forman una diagonal en el mapa: con gamma = 0,01 el mejor C es 1000 (0,845), con gamma = 0,1 es 100 (0,885) y con gamma = 1 es 10 (0,886). Cuanto mayor es gamma, menor es el C que necesita. El máximo queda dentro de la grilla, no en un borde, así que no hace falta ampliarla.
#
# El tiempo de entrenamiento también depende de los dos: con gamma = 0,1 están las combinaciones más rápidas (de 38 a 76 segundos por fold), y con gamma = 10, las más lentas (de 264 a 424 segundos), que son además las de peor exactitud.
#
# En la tabla, con gamma = 10 la precisión se separa mucho de la exactitud: con C = 1, por ejemplo, la precisión es 0,8775 y la exactitud 0,6172. Que la precisión siga alta con una exactitud tan baja quiere decir que casi todos los cultivos, cuando el modelo los predice, están bien, y que los errores se juntan en unos pocos cultivos que el modelo predice de más.

# %% [markdown]
# ### 3.2 Mejor configuración

# %%
mejor_rbf = mejor_configuracion(resultados_rbf)
tabla_rbf.loc[[mejor_rbf.name]]

# %% [markdown]
# Elegimos **C = 10 y gamma = 1**, la combinación de mayor exactitud media (0,8864, con un desvío de 0,0035 entre folds). La segunda, C = 100 y gamma = 0,1, queda muy cerca (0,8850), a menos de la mitad del desvío entre folds, así que la diferencia no es firme. Esa combinación varía menos entre folds (desvío de 0,0018) y entrena en 48 segundos por fold en lugar de 128, pero con el criterio de 2.1 nos quedamos con la de mayor exactitud media.

# %% [markdown]
# ---
# ## 4. Random Forest
#
# ### 4.1 Búsqueda de n_estimators y max_depth
#
# En 1.4, varios grupos de cultivos se separaban con un corte en una sola variable (la temperatura, el pH o la lluvia), que es justamente lo que hace cada nodo de un árbol. Por eso esperamos que el Random Forest acierte más que los dos SVM. Esperamos también que la profundidad pese más que la cantidad de árboles: un árbol de 5 niveles tiene a lo sumo 32 hojas para 10 cultivos, y con más niveles la exactitud debería subir hasta estabilizarse. Sumar árboles debería mejorar cada vez menos, mientras el tiempo de entrenamiento crece en proporción.
#
# Probamos 5 valores de `n_estimators` (50, 100, 200, 300 y 500) y 7 de `max_depth` (de 5 a 30, de 5 en 5, y sin límite): son 5 × 7 = 35 combinaciones, cada una entrenada y evaluada en los 5 folds, 175 entrenamientos. Agregamos profundidades intermedias a las del ejemplo de la consigna (5, 10, 20 y sin límite) para ver dónde se estabiliza la exactitud. Mostramos los resultados con la misma tabla y el mismo mapa de calor que en el RBF.
#
# El Random Forest no necesita datos estandarizados (notebook de árboles de clasificación de la U4), pero usamos el mismo pipeline que en los SVM para que los tres modelos se evalúen igual. No cambia los resultados: estandarizar no cambia qué filas quedan de cada lado de un corte.

# %%
GRILLA_RF = {"randomforestclassifier__n_estimators": [50, 100, 200, 300, 500],
             "randomforestclassifier__max_depth": [5, 10, 15, 20, 25, 30, None]}
resultados_rf = buscar_hiperparametros("random_forest", RandomForestClassifier(random_state=SEMILLA), GRILLA_RF)
tabla_rf = tabla_resultados(resultados_rf, {"randomforestclassifier__n_estimators": "n_estimators",
                                            "randomforestclassifier__max_depth": "max_depth"})
# max_depth=None es sin límite de profundidad
tabla_rf["max_depth"] = tabla_rf["max_depth"].fillna("sin límite")

# Ordenada de mayor a menor exactitud media
tabla_rf.sort_values("Exactitud media", ascending=False)

# %%
graficar_interaccion(tabla_rf, "max_depth", "n_estimators", "Random Forest según max_depth y n_estimators")

# %% [markdown]
# Se cumple. La profundidad es lo que más cambia la exactitud: con `max_depth = 5` no pasa de 0,646, con 10 llega a 0,843, con 15 a 0,904, y desde 25 se estabiliza entre 0,913 y 0,917. Para una misma profundidad, pasar de 50 a 500 árboles mejora menos de 0,01 (sin límite de profundidad, de 0,914 a 0,917), mientras que el tiempo crece en proporción a la cantidad de árboles: sin límite de profundidad, de 11 a 96 segundos por fold. Todas las combinaciones desde `max_depth = 15` superan al mejor SVM RBF (0,8864), como esperábamos.

# %% [markdown]
# ### 4.2 Mejor configuración

# %%
mejor_rf = mejor_configuracion(resultados_rf)
tabla_rf.loc[[mejor_rf.name]]

# %% [markdown]
# Elegimos **`n_estimators = 500` y `max_depth = 30`** (exactitud media de 0,9167, con un desvío de 0,0034 entre folds). Empata exactamente con 500 árboles sin límite de profundidad, y ante un empate el criterio de 2.1 toma la primera combinación de la grilla, que además tiene un desvío apenas menor (0,0034 contra 0,0036). Como en los SVM, las diferencias con las combinaciones que siguen son chicas: 300 árboles sin límite de profundidad quedan a 0,0003 y entrenan en 70 segundos por fold, en lugar de 122.

# %% [markdown]
# ### 4.3 Importancia de las variables
#
# Analizamos la importancia del bosque con la mejor configuración, reentrenado con todos los datos limpios. Por 1.4, esperamos que las más importantes sean las variables que más cambian entre cultivos: las temperaturas, el pH, la lluvia y la humedad. Como `temperatura` y `temp_min_mensual` están muy correlacionadas (0,95, en 1.7), muchos cortes se pueden hacer con cualquiera de las dos, y esperamos que la importancia se reparta entre ambas en lugar de concentrarse en una.

# %%
# El bosque con la mejor configuración, reentrenado con los 49803 datos limpios (los 5 folds juntos). n_jobs=-1
# entrena los árboles en paralelo y no cambia el resultado
bosque = crear_modelo(RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)).set_params(**mejor_rf["params"])
bosque.fit(X, y)

# Importancia de cada variable, con el gráfico de barras de Ensambles_RandomForest.ipynb (U6, sección "Importancia
# de las características"). bosque[-1] es el último paso del pipeline, el Random Forest
importancias = pd.DataFrame({"variable": ATRIBUTOS, "importancia": bosque[-1].feature_importances_})
importancias = importancias.sort_values(by="importancia", ascending=False)

plt.figure(figsize=(8, 6))
g = sns.barplot(data=importancias, x="importancia", y="variable")
sns.despine(bottom=True, left=True)
g.set_title(f"Importancia de las variables en el Random Forest elegido\n(n_estimators = "
            f"{tabla_rf.loc[mejor_rf.name, 'n_estimators']}, max_depth = {tabla_rf.loc[mejor_rf.name, 'max_depth']}, "
            "entrenado con todos los datos limpios)")
g.set(xlabel=None, ylabel=None, xticks=[])
for barras in g.containers:
    g.bar_label(barras, padding=2, fmt="%.3f")
plt.show()

# %% [markdown]
# Se cumple en parte. Las ocho variables más importantes son todas de clima o de ubicación, con importancias de 0,071 a 0,117: las tres temperaturas, la radiación solar, la lluvia, la humedad, la estacionalidad de las lluvias y la elevación. Las ocho del suelo quedan abajo, entre 0,029 y 0,045. El pH, que en 1.4 separaba a los cultivos de suelos ácidos de los de suelos neutros, es la más importante de las del suelo (0,045), pero queda novena, cuando esperábamos que estuviera entre las primeras. Una explicación posible es que esa separación ya la dan las variables de clima, con las que el pH está muy correlacionado (−0,79 con la humedad y −0,78 con la lluvia, en 1.7): un corte por humedad o por lluvia separa casi los mismos cultivos que uno por pH.
#
# `temperatura` y `temp_min_mensual` se reparten la importancia, como esperábamos: las dos quedan arriba (0,083 y 0,117), y juntas suman 0,200, más del doble que cualquier otra variable.

# %% [markdown]
# ---
# ## 5. Comparación de modelos
#
# ### 5.1 Métricas de la mejor configuración de cada algoritmo

# %%
# La fila de la mejor configuración de cada algoritmo en las tablas de 2, 3 y 4
COLUMNAS = list(tabla_lineal.columns.drop("C"))
comparacion = pd.DataFrame([tabla_lineal.loc[mejor_lineal.name, COLUMNAS],
                            tabla_rbf.loc[mejor_rbf.name, COLUMNAS],
                            tabla_rf.loc[mejor_rf.name, COLUMNAS]],
                           index=["SVM lineal", "SVM RBF", "Random Forest"])
configuracion_rf = tabla_rf.loc[mejor_rf.name]
comparacion.insert(0, "Configuración", [
    f"C = {mejor_lineal['param_svc__C']:g}",
    f"C = {mejor_rbf['param_svc__C']:g}, gamma = {mejor_rbf['param_svc__gamma']:g}",
    f"n_estimators = {configuracion_rf['n_estimators']}, max_depth = {configuracion_rf['max_depth']}"])
comparacion

# %% [markdown]
# El Random Forest es el mejor en las tres métricas: 0,9167 de exactitud, 0,030 más que el SVM RBF (0,8864) y 0,29 más que el SVM lineal (0,6232). La diferencia con el RBF es casi nueve veces el desvío entre folds, así que, a diferencia de las elecciones dentro de cada modelo, esta sí es firme. Los tres son estables: el desvío entre folds no pasa de 0,005 en ninguno, y los del Random Forest y el RBF son casi iguales (0,0034 y 0,0035).
#
# En tiempo de entrenamiento, el Random Forest (122 segundos por fold) y el RBF (128) están parejos, y el lineal es el más lento (867) por el C = 100 elegido en 2.2: con C = 1, que rinde casi lo mismo, tarda 98 segundos. Estos tiempos se midieron durante las búsquedas, que entrenaban varios modelos a la vez, así que sirven para comparar los modelos entre sí, pero no son los de un modelo entrenado solo.

# %% [markdown]
# ### 5.2 Matrices de confusión
#
# La consigna permite generar las matrices con uno de los folds de la validación cruzada o con un split 80/20 aparte. Usamos el primer fold (el mismo de 1.8), así las matrices salen del mismo esquema que las métricas: cada modelo se entrena con los otros 4 folds y predice las filas del fold 1, unas 1000 de cada cultivo.
#
# Por 1.4, esperamos que los errores se concentren en `banana` y `cafe`, que se parecen en casi todas las variables, y en `maiz`, el cultivo con los rangos más amplios, que se superpone con varios. También esperamos alguna confusión entre `garbanzo` y `lenteja`, que comparten 6 de los pares con los mismos atributos de 1.2.

# %%
# Primer fold de la validación cruzada (el mismo de 1.8): cada modelo se entrena con los otros 4 folds y predice este
X_train, X_test = X.iloc[indices_train], X.iloc[indices_test]
y_train, y_test = y[indices_train], y[indices_test]

# La mejor configuración de cada algoritmo
modelos = {"SVM lineal": crear_modelo(SVC(kernel="linear", random_state=SEMILLA)).set_params(**mejor_lineal["params"]),
           "SVM RBF": crear_modelo(SVC(kernel="rbf", random_state=SEMILLA)).set_params(**mejor_rbf["params"]),
           "Random Forest": crear_modelo(RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)).set_params(
               **mejor_rf["params"])}

# Entrenar los tres modelos tarda varios minutos (casi 15 el SVM lineal con C = 100): como con las búsquedas, la
# primera vez guardamos las predicciones en ../data/ y las siguientes las leemos de ahí
RUTA_PREDICCIONES = "../data/predicciones_fold1.pkl"
if not os.path.exists(RUTA_PREDICCIONES):
    predicciones = pd.DataFrame()
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones[nombre] = modelo.predict(X_test)
    predicciones.to_pickle(RUTA_PREDICCIONES)
predicciones = pd.read_pickle(RUTA_PREDICCIONES)

for nombre in predicciones:
    print(f"{nombre}: exactitud {accuracy_score(y_test, predicciones[nombre]):.4f} en el fold 1")

# %%
# Adaptado de plot_multiple_conf_matrices de gas_detection.ipynb (U6): la misma grilla de 2 × 2 con una matriz por
# modelo, pero con ConfusionMatrixDisplay (como en el TP2), el mismo mapa de colores para las tres y los nombres de
# los cultivos en los ejes
fig, axes = plt.subplots(2, 2, figsize=(17, 15))
axes = axes.flatten()
for ax, nombre in zip(axes, predicciones):
    ConfusionMatrixDisplay.from_predictions(y_test, predicciones[nombre], display_labels=codificador.classes_,
                                            cmap="Blues", colorbar=False, xticks_rotation=45, ax=ax)
    ax.set(title=nombre, xlabel="Predicción", ylabel="Real")
fig.delaxes(axes[3])   # sobra el cuarto lugar de la grilla
fig.suptitle("Matrices de confusión en el fold 1")
plt.tight_layout()
plt.show()

# %% [markdown]
# Las exactitudes en el fold 1 (0,6182, 0,8888 y 0,9152) están cerca de las medias de los 5 folds de 5.1, así que este fold es representativo.
#
# Se cumple en parte. `maiz` es el cultivo que más les cuesta a los tres modelos, como esperábamos por sus rangos amplios: de sus 1000 filas, el Random Forest acierta 778, el RBF 787 y el lineal 388, y los errores se reparten entre todos los demás cultivos. El RBF, además, predice `maiz` de más: le asigna filas de casi todos los cultivos (64 de `arroz` y 49 de `algodon`, por ejemplo). También se cumple la confusión entre `garbanzo` y `lenteja`: 49 filas de `garbanzo` clasificadas como `lenteja` son el error más frecuente del Random Forest, y en el RBF son 42.
#
# Con `banana` y `cafe`, en cambio, solo se cumple para el lineal: confunde 200 filas de `banana` con `cafe` y 152 de `cafe` con `banana`, mientras que el RBF (25 y 17) y el Random Forest (26 y 15) casi no las confunden. Aunque se parecen en cada variable por separado (1.4), los dos modelos no lineales encuentran combinaciones de variables que las separan. El lineal tiene además otros errores grandes, como 188 filas de `algodon` clasificadas como `frijol`, 168 de `cafe` como `coco` y 147 de `manzana` como `maiz`. Un error que no anticipamos aparece en los dos mejores modelos: `arroz` clasificado como `coco` (55 filas en el RBF y 42 en el Random Forest).

# %% [markdown]
# ### 5.3 Elección del modelo
#
# Elegimos **el Random Forest, con `n_estimators = 500` y `max_depth = 30`**. Es el que mejor clasifica, con una diferencia firme sobre el RBF (5.1), y sus errores están donde los anticipaba el análisis de 1.4. Es tan estable entre folds como el RBF (desvío de 0,0034 contra 0,0035), y su tiempo de entrenamiento es parecido, pero se puede bajar mucho casi sin perder exactitud: con 100 árboles sin límite de profundidad, la exactitud media es 0,9149 y entrena en unos 22 segundos por fold (4.1). Además, es el más fácil de interpretar: la importancia de las variables muestra en qué se apoya la recomendación, sobre todo en el clima (4.3). Los SVM no dan esa información de forma directa, y el apunte de la Unidad 5 («Desventajas de SVM») dice que son difíciles de interpretar, sobre todo con kernels no lineales.
#
# El SVM lineal queda descartado por su exactitud: separar estos cultivos con hiperplanos no alcanza. El RBF es la alternativa más cercana, pero clasifica peor y no permite ver en qué se basa cada recomendación.

# %% [markdown]
# ---
# ## 6. Conclusiones
#
# 1. El conjunto tenía tres problemas cargados a propósito: una fila con 3 valores faltantes, 198 filas duplicadas (sobre todo de `lenteja` y `manzana`) y 2 filas con valores imposibles (una con un pH de 25, y otra con seis valores imposibles o aislados, entre ellos un pH de 18 y una humedad de 148 %). Las eliminamos y quedaron 49803 filas, con las clases casi balanceadas. Los miles de atípicos que marca la regla del rango intercuartílico en las variables de suelo, la lluvia y la elevación son colas propias de algunos cultivos, y los conservamos. También conservamos 7 pares de filas con los mismos atributos y distinta clase, que ningún modelo puede acertar enteros.
# 2. El SVM lineal no alcanza para este problema: su exactitud se estanca en 0,62 desde C = 1, y subir C solo multiplica el tiempo de entrenamiento (de 98 a 867 segundos por fold). Confunde sobre todo cultivos que se parecen en cada variable por separado, como `banana` y `cafe`.
# 3. El SVM RBF llega a 0,886 de exactitud con C = 10 y gamma = 1. El mejor C depende de gamma: cuanto mayor es gamma, menor es el C que necesita, y con gamma = 10 el modelo se derrumba.
# 4. El Random Forest, con 500 árboles de hasta 30 niveles, es el mejor modelo: llega a 0,917 de exactitud. La profundidad pesa mucho más que la cantidad de árboles, y las variables de clima pesan más que las del suelo.
# 5. Dentro de cada modelo, las mejores configuraciones se diferencian menos que el desvío entre folds, y la elección entre ellas no es firme. Entre modelos, en cambio, las diferencias son claras.
# 6. Elegimos el Random Forest por su exactitud, su estabilidad entre folds, su tiempo de entrenamiento y porque se puede interpretar. Sus errores están donde los anticipó el análisis exploratorio: en `maiz`, el cultivo con los rangos más amplios, y entre `garbanzo` y `lenteja`.
