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
# # Trabajo Práctico N° 2: árboles de decisión, Bayes ingenuo y k-NN
#
# | | |
# |---|---|
# | **Año** | 2026 |
# | **Materia** | Minería de Datos, Tecnicatura Universitaria en Inteligencia Artificial |
# | **Integrantes** | Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo |
#
# **Objetivo:** aplicar los modelos de las unidades 4 y 5 a dos problemas: estimar la ganancia (`Profit`) de 1000 empresas a partir de sus gastos (regresión) y predecir la droga que se le receta a un paciente (`Droga`) a partir de sus datos clínicos (clasificación).
#
# **Contenido**
#
# - Parte A – Regresión: 1. Preparación de datos y 2. Árboles de decisión para regresión.
# - Parte B – Clasificación: 3. Preparación de datos, 4. Árboles de decisión para clasificación, 5. Bayes Ingenuo, 6. k-NN y 7. Comparación de modelos.
# - Conclusiones.
#
# Cuando reutilizamos o adaptamos código de los notebooks de clase, el comentario de la celda dice de cuál sale.

# %%
# Análisis exploratorio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

# Preparación de los datos y búsqueda de hiperparámetros
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler

# Modelos
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Métricas
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)

# random_state que pide la consigna para las particiones y los modelos
SEMILLA = 63

# Las dos particiones entrenamiento/test que pide la consigna, con su proporción de test
PARTICIONES = {"80/20": 0.2, "70/30": 0.3}

# Rutas relativas a la carpeta del notebook (tp2/notebooks/)
RUTA_EMPRESAS = "../data/1000_Companies.csv"
RUTA_DROGAS = "../data/drugType.csv"

# %% [markdown]
# ---
# # Parte A – Regresión (dataset 1000_Companies.csv)
#
# ## 1. Preparación de datos
#
# ### 1.1 Estructura, valores faltantes y duplicados

# %%
# El archivo empieza con una marca BOM: con "utf-8-sig" la primera columna se lee como "RyD_Spend"
empresas = pd.read_csv(RUTA_EMPRESAS, encoding="utf-8-sig")
empresas.head()

# %%
# Tipo de cada columna y cantidad de valores no nulos
empresas.info()

# %% [markdown]
# Hay 1000 empresas y 5 columnas: los tres gastos (`RyD_Spend`, `Administration` y `MarketingSpend`), el estado (`State`, la única categórica) y la ganancia, `Profit`, que es la variable a estimar. Ninguna columna tiene valores faltantes.

# %%
# Filas repetidas: keep=False muestra la fila original y su copia
empresas[empresas.duplicated(keep=False)]

# %% [markdown]
# Hay una fila duplicada: la 755 repite la 335. La sacamos en la limpieza.

# %%
GASTOS = ["RyD_Spend", "Administration", "MarketingSpend"]
NUMERICAS_EMPRESAS = GASTOS + ["Profit"]

# Resumen de las variables numéricas
empresas[NUMERICAS_EMPRESAS].describe().round(0)

# %%
# Cantidad de empresas con gasto 0: podría ser un dato faltante cargado como cero
(empresas[GASTOS] == 0).sum()

# %%
# Cantidad de empresas de cada estado
empresas["State"].value_counts()

# %% [markdown]
# Dos empresas tienen gasto 0 en I+D y tres en marketing: son valores posibles (una empresa puede no invertir en eso), así que los conservamos. `State` tiene tres estados con cantidades parecidas, de 322 a 344 empresas. Los máximos de `Profit` (476.485) y de `Administration` (321.652) están muy lejos de su tercer cuartil (155.577 y 129.139); lo revisamos en los atípicos.

# %% [markdown]
# ### 1.2 Distribuciones
#
# **Hipótesis.** Los gastos y la ganancia son montos de empresas de distintos tamaños, así que esperamos distribuciones amplias y quizás algunos valores extremos.

# %%
# Histograma de cada variable numérica, como en el notebook de árboles de regresión de la U4 (sección "Distribuciones")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, columna in zip(axes, NUMERICAS_EMPRESAS):
    sns.histplot(data=empresas, x=columna, kde=True, element="step", ax=ax)
fig.suptitle("Distribución de las variables numéricas")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se cumple la hipótesis. `RyD_Spend` y `MarketingSpend` se reparten en todo su rango, sin una forma marcada. `Administration` está concentrada entre unos 110.000 y 135.000 dólares, con pocos valores sueltos a los dos lados, y `Profit` está casi toda por debajo de 200.000, pero unos pocos valores aislados estiran el eje hasta 476.000.

# %% [markdown]
# ### 1.3 Valores atípicos
#
# **Hipótesis.** Si hay valores extremos, deberían aparecer como atípicos según la regla del rango intercuartílico. Si además son errores de carga, no deberían seguir la relación con los gastos que tienen las demás empresas.

# %%
# Función de U3_Clustering.ipynb (sección "Remocion de Outliers"), la misma que usamos en el TP1:
# devuelve solo las filas que quedan dentro de los límites del rango intercuartílico
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


# Adaptada del "Conteo de atípicos por columna" de U4_DecisionTreeRegresion.ipynb: cuenta con
# remove_outliers_iqr y devuelve la tabla en lugar de imprimirla
def contar_atipicos(df, columnas):
    """Cantidad y porcentaje de atípicos de cada columna, según la regla del rango intercuartílico."""
    resultados = []
    for col in columnas:
        cantidad = len(df) - len(remove_outliers_iqr(df, col))
        resultados.append({"Variable": col, "Cantidad": cantidad, "Porcentaje": round(100 * cantidad / len(df), 1)})
    return pd.DataFrame(resultados)


contar_atipicos(empresas, NUMERICAS_EMPRESAS)

# %%
# Diagrama de caja de cada variable, como en el notebook de árboles de regresión de la U4 (sección "Valores atípicos")
fig, axes = plt.subplots(1, 4, figsize=(16, 2.8))
for ax, columna in zip(axes, NUMERICAS_EMPRESAS):
    sns.boxplot(data=empresas, x=columna, ax=ax)
fig.suptitle("Valores atípicos (regla del rango intercuartílico)")
plt.tight_layout()
plt.show()

# %%
# Las empresas atípicas en Profit son las que remove_outliers_iqr deja afuera
atipicas_profit = empresas.drop(remove_outliers_iqr(empresas, "Profit").index)
atipicas_profit

# %%
# Profit contra cada gasto, con las empresas atípicas en Profit en rojo
grupo = np.where(empresas.index.isin(atipicas_profit.index), "atípica en Profit", "resto")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, columna in zip(axes, GASTOS):
    sns.scatterplot(data=empresas, x=columna, y="Profit", hue=grupo, hue_order=["resto", "atípica en Profit"],
                    palette=["tab:blue", "tab:red"], s=15, ax=ax)
fig.suptitle("Profit según cada gasto")
plt.tight_layout()
plt.show()

# %% [markdown]
# La regla del rango intercuartílico marca 22 atípicos en `Administration` y 3 en `Profit`. Los de `Profit` son las tres últimas filas del archivo (997 a 999): ganan entre 334.000 y 476.000 dólares, cuando ninguna otra empresa llega a 200.000, y tienen los tres valores más altos de `Administration` (de 242.000 a 322.000).
#
# Se cumplen las dos partes de la hipótesis. En los gráficos de dispersión, casi todas las empresas forman una línea con `RyD_Spend` y con `MarketingSpend`, y las tres atípicas quedan muy por encima de esa línea con los tres gastos. Las tratamos como errores de carga y las sacamos. En `Administration`, la mayoría de las empresas forma una franja angosta y unas pocas quedan dispersas alrededor; las miramos después de limpiar.

# %% [markdown]
# ### 1.4 Limpieza

# %%
# Sacamos la fila duplicada (pesaría el doble en el entrenamiento) y las tres empresas atípicas en Profit
empresas_limpio = empresas.drop_duplicates()
empresas_limpio = empresas_limpio.drop(atipicas_profit.index)
print(f"Quedan {len(empresas_limpio)} de {len(empresas)} empresas")

# %%
# Las empresas atípicas en Administration que quedan, comparadas con el resto: mínimo y máximo de cada variable
atipicas_admin = empresas_limpio.drop(remove_outliers_iqr(empresas_limpio, "Administration").index)
grupo = np.where(empresas_limpio.index.isin(atipicas_admin.index), "atípica en Administration", "resto")
empresas_limpio.groupby(grupo)[NUMERICAS_EMPRESAS].agg(["count", "min", "max"]).round(0)

# %% [markdown]
# Quedan 996 empresas. Las 19 atípicas en `Administration` que quedan gastan de 51.283 a 182.646 dólares en administración, mientras que el resto está entre 99.281 y 147.199. Sus otros gastos y su ganancia están dentro de los rangos del resto. Son montos posibles para un gasto de administración, así que las conservamos.

# %% [markdown]
# ### 1.5 Correlaciones
#
# **Hipótesis.** Esperamos que la ganancia crezca con los tres gastos, y que los gastos estén correlacionados entre sí, porque una empresa más grande gasta más en todo.

# %%
# Matriz de correlación de Pearson, con el mapa de calor de los notebooks de árboles de la U4
correlaciones = empresas_limpio[NUMERICAS_EMPRESAS].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(correlaciones, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
            annot=True, fmt=".2f", square=True)
plt.title("Correlación de Pearson (datos limpios)")
plt.show()

# %%
# Efecto de la limpieza: correlación de Profit con cada gasto, con y sin las tres empresas atípicas en Profit
con_atipicas = empresas.drop_duplicates()[NUMERICAS_EMPRESAS].corr()["Profit"]
sin_atipicas = correlaciones["Profit"]
pd.DataFrame({"con las atípicas": con_atipicas, "sin las atípicas": sin_atipicas}).drop("Profit").round(3)

# %% [markdown]
# Se cumple la hipótesis. `Profit` está casi perfectamente correlacionada con `RyD_Spend` (0,999), muy correlacionada con `MarketingSpend` (0,98) y bastante menos con `Administration` (0,75). Entre los gastos, `RyD_Spend` y `MarketingSpend` también están muy correlacionadas (0,98): cuentan casi lo mismo.
#
# La segunda tabla muestra el efecto de la limpieza. Con las tres atípicas, la correlación de `Profit` con `RyD_Spend` era de 0,945 y con `MarketingSpend`, de 0,917: tres empresas alcanzaban para bajarlas, porque se alejan mucho de la relación del resto.

# %% [markdown]
# ### 1.6 Codificación de la variable categórica

# %%
# OrdinalEncoder con salida en DataFrame, como en los notebooks de Bayes y k-NN de la U5.
# State es nominal: el orden de los códigos no significa nada, pero un árbol puede separar cualquiera
# de los tres estados con a lo sumo dos cortes, así que alcanza con un número por estado
codificador_estado = OrdinalEncoder().set_output(transform="pandas")
empresas_limpio[["State"]] = codificador_estado.fit_transform(empresas_limpio[["State"]])

# Categorías en el orden de sus códigos (0, 1 y 2), como muestra el notebook de Bayes de la U5
codificador_estado.categories_

# %% [markdown]
# `State` queda con California = 0, Florida = 1 y New York = 2.

# %% [markdown]
# ### 1.7 Particiones 80/20 y 70/30 y estandarización
#
# Dividimos con `random_state = 63` y después estandarizamos los gastos con la media y el desvío del entrenamiento de cada partición, como indica el apunte de la Unidad 4 («Estandarización vs separación de datos»): así el test no influye en nada de lo que se aprende. `Profit` no se estandariza, para que los errores queden en dólares.
#
# Cada partición se guarda como `(X_train, X_test, y_train, y_test)`, los cuatro conjuntos que usan los notebooks de clase.

# %%
X_empresas = empresas_limpio.drop(columns="Profit")
y_empresas = empresas_limpio["Profit"]

particiones_a = {}   # nombre de la partición -> (X_train, X_test, y_train, y_test)
escaladores_a = {}   # nombre de la partición -> escalador ajustado con su entrenamiento
for nombre, proporcion_test in PARTICIONES.items():
    X_train, X_test, y_train, y_test = train_test_split(X_empresas, y_empresas, test_size=proporcion_test,
                                                        random_state=SEMILLA)

    # El escalador aprende la media y el desvío solo del entrenamiento y se aplica a los dos conjuntos,
    # como en los notebooks de Bayes y k-NN de la U5
    escalador = StandardScaler().set_output(transform="pandas")
    escalador.fit(X_train[GASTOS])
    X_train[GASTOS] = escalador.transform(X_train[GASTOS])
    X_test[GASTOS] = escalador.transform(X_test[GASTOS])

    particiones_a[nombre] = (X_train, X_test, y_train, y_test)
    escaladores_a[nombre] = escalador
    print(f"Partición {nombre}: {len(X_train)} empresas de entrenamiento y {len(X_test)} de test")

# %%
# Entrenamiento de la partición 80/20 ya preparado
X_train, X_test, y_train, y_test = particiones_a["80/20"]
X_train.describe().round(2)

# %% [markdown]
# En el entrenamiento ya preparado, los tres gastos tienen media 0 y desvío 1. `Administration` va de −8,01 a 6,77 desvíos por las empresas atípicas que conservamos, y `State` queda con sus códigos de 0 a 2.

# %% [markdown]
# ## 2. Árboles de decisión para regresión
#
# En cada partición probamos 864 combinaciones: `max_depth` de 2 a 8, 10, 12, 15, 20 y sin límite; `min_samples_leaf` de 1 a 5, 7, 10, 15 y 20; y `min_samples_split` de 2, 4, 6, 8, 10, 15, 20 y 30. Cada combinación se evalúa con validación cruzada de 5 pliegues sobre el entrenamiento, con `GridSearchCV` como en el notebook de árboles de clasificación de la U4, y elegimos la de menor RMSE promedio, como propone el apunte de la Unidad 4. El test se usa una sola vez, al final.

# %%
GRILLA_REGRESION = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 15, 20],
                    "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30]}

busquedas_a = {}    # nombre de la partición -> GridSearchCV ya ajustado
resultados_a = {}   # nombre de la partición -> tabla con el resultado de cada combinación
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_a[nombre]

    # scoring="neg_root_mean_squared_error" devuelve el RMSE con signo negativo, porque GridSearchCV
    # siempre busca el valor más alto
    busqueda = GridSearchCV(DecisionTreeRegressor(random_state=SEMILLA), GRILLA_REGRESION, cv=5,
                            scoring="neg_root_mean_squared_error")
    busqueda.fit(X_train, y_train)

    resultados = pd.DataFrame(busqueda.cv_results_)
    resultados["RMSE validación"] = -resultados["mean_test_score"]   # de vuelta a positivo, en dólares
    busquedas_a[nombre] = busqueda
    resultados_a[nombre] = resultados
    print(f"Partición {nombre}: {busqueda.best_params_}, RMSE de validación {-busqueda.best_score_:,.0f} dólares")

# %% [markdown]
# **Hipótesis.** Un árbol más profundo aproxima mejor una relación continua como la de `Profit` con los gastos, así que esperamos que el RMSE baje al aumentar `max_depth` hasta estabilizarse. Las hojas muy chicas deberían empezar a ajustar el ruido del entrenamiento, y las muy grandes, a promediar empresas demasiado distintas.

# %%
# Efecto de cada hiperparámetro: para cada valor, el menor RMSE de validación entre todas las
# combinaciones de los otros dos
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for fila, nombre in enumerate(PARTICIONES):
    resultados = resultados_a[nombre].copy()
    resultados["param_max_depth"] = resultados["param_max_depth"].fillna("sin límite")
    for columna, parametro in enumerate(GRILLA_REGRESION):
        mejor_por_valor = resultados.groupby(f"param_{parametro}", sort=False)["RMSE validación"].min()
        ax = axes[fila, columna]
        ax.plot(mejor_por_valor.index.astype(str), mejor_por_valor.values, marker="o")
        ax.set(title=f"Partición {nombre}", xlabel=parametro, ylabel="RMSE de validación (USD)")
        ax.tick_params(axis="x", rotation=45)
fig.suptitle("Menor RMSE de validación para cada valor de cada hiperparámetro")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se cumple la hipótesis. Con `max_depth` de 2 a 4 el error es alto (de unos 10.000 a 3.300 dólares), y desde 6 o 7 niveles se estabiliza cerca de 1.800 (80/20) y 2.100 (70/30). `min_samples_leaf` tiene su mejor valor en el medio, con hojas de 4 empresas (80/20) y de 5 (70/30): con hojas más chicas el árbol ajusta el ruido del entrenamiento, y con hojas más grandes promedia empresas de ganancias distintas. `min_samples_split` no cambia el error hasta 8 o 10; con valores mayores, el árbol deja de dividir nodos chicos y el error sube.

# %%
# Las 5 mejores combinaciones de cada partición, ordenadas por rank_test_score como en el notebook de
# árboles de clasificación de la U4
mostrar = ["rank_test_score", "param_max_depth", "param_min_samples_leaf", "param_min_samples_split",
           "RMSE validación"]
mejores = []
for nombre in PARTICIONES:
    tabla = resultados_a[nombre][mostrar].sort_values("rank_test_score").head(5)
    tabla.insert(0, "Partición", nombre)
    mejores.append(tabla)
pd.concat(mejores).round(0)

# %% [markdown]
# Con 80/20, la configuración elegida es `max_depth = 8`, `min_samples_leaf = 4` y `min_samples_split = 2` (RMSE de validación de 1.777 dólares), y con 70/30, `max_depth = 7`, `min_samples_leaf = 5` y `min_samples_split = 2` (2.077). Las primeras filas de cada partición empatan porque dan el mismo árbol: con hojas de al menos 4 empresas, un nodo con menos de 8 no se puede dividir en dos, así que `min_samples_split` de 2 a 8 no cambia nada (con hojas de 5, de 2 a 10). Ante un empate, `GridSearchCV` se queda con la primera combinación.

# %%
# RMSE del árbol elegido en entrenamiento, validación y test (como la comparación de RMSE del notebook de
# árboles de regresión de la U4), y el MAE y el MSE de test que pide la consigna
filas = []
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_a[nombre]
    arbol = busquedas_a[nombre].best_estimator_   # GridSearchCV ya lo reentrenó con todo el entrenamiento
    prediccion_train = arbol.predict(X_train)
    prediccion_test = arbol.predict(X_test)
    filas.append({"Partición": nombre,
                  "Hojas": arbol.get_n_leaves(),
                  "RMSE entrenamiento": np.sqrt(mean_squared_error(y_train, prediccion_train)),
                  "RMSE validación": -busquedas_a[nombre].best_score_,
                  "RMSE test": np.sqrt(mean_squared_error(y_test, prediccion_test)),
                  "MAE test": mean_absolute_error(y_test, prediccion_test),
                  "MSE test": mean_squared_error(y_test, prediccion_test)})
pd.DataFrame(filas).set_index("Partición").round(0)

# %% [markdown]
# En test, el árbol elegido tiene un RMSE de 3.485 dólares con 80/20 y de 2.806 con 70/30, y un MAE de 1.046 y 940: menos de una décima parte del desvío de `Profit` (42.889, en la tabla de 1.1).
#
# El RMSE de entrenamiento (1.205 y 1.446) es menor que el de validación (1.777 y 2.077): hay algo de sobreajuste, el esperable en árboles de 156 y 99 hojas. El RMSE de test es mayor que el de validación y unas tres veces el MAE. Como el RMSE pesa más los errores grandes, eso indica que unas pocas empresas concentran buena parte del error.

# %% [markdown]
# ### El árbol con la mejor configuración
#
# **Hipótesis.** Por las correlaciones de la preparación, esperamos que el árbol corte primero por `RyD_Spend` y que casi no use `State`.

# %%
# Los tres primeros niveles del árbol de la partición 80/20, con plot_tree y max_depth=3 como en el notebook
# de árboles de regresión de la U4 (el árbol completo tiene demasiadas hojas para leerse)
X_train, X_test, y_train, y_test = particiones_a["80/20"]
arbol = busquedas_a["80/20"].best_estimator_
plt.figure(figsize=(22, 8))
plot_tree(arbol, feature_names=list(X_train.columns), max_depth=3, filled=True, rounded=True,
          impurity=False, fontsize=9, precision=2)
plt.title(f"Árbol de regresión de la partición 80/20 ({arbol.get_n_leaves()} hojas; se muestran los 3 primeros niveles)")
plt.show()

# %%
# Importancia de cada variable, con el gráfico de barras del notebook de árboles de regresión de la U4
importancias = pd.DataFrame({"variable": X_train.columns, "importancia": arbol.feature_importances_})
importancias = importancias.sort_values(by="importancia", ascending=False)

g = sns.barplot(data=importancias, x="importancia", y="variable")
sns.despine(bottom=True, left=True)
g.set_title("Importancia de las variables (árbol de la partición 80/20)")
g.set(xlabel=None, ylabel=None, xticks=[])
for barras in g.containers:
    g.bar_label(barras, padding=2, fmt="%.3f")
plt.show()

# %%
# Los umbrales del gráfico están en desvíos estándar. Para leerlos en dólares: umbral × desvío + media del
# entrenamiento. RyD_Spend es la primera de las columnas que estandarizó el escalador
escalador = escaladores_a["80/20"]
media = escalador.mean_[0]
desvio = escalador.scale_[0]
for umbral in [-0.86, 0.03, 0.85]:   # los cortes de los dos primeros niveles, leídos del gráfico
    print(f"RyD_Spend <= {umbral} desvíos equivale a {umbral * desvio + media:,.0f} dólares")

# %% [markdown]
# Se cumple la hipótesis. El primer corte es por `RyD_Spend` en 0,03 desvíos, unos 82.000 dólares, y el segundo nivel también corta por `RyD_Spend`, en unos 41.000 y 120.000. Los tres primeros niveles usan solo esa variable; `MarketingSpend` y `Administration` aparecen recién en el cuarto. La importancia de `RyD_Spend` es 0,989 y la de `State`, 0, coherente con la correlación de 0,999 entre `Profit` y `RyD_Spend`.
#
# Los umbrales están en desvíos porque los gastos se estandarizaron en la preparación, como pide la consigna. Al árbol no le cambia nada: los árboles no requieren escalado (lo dice el notebook de árboles de clasificación de la U4), y estandarizar solo cambia la escala en la que se leen los umbrales.

# %% [markdown]
# ### Valores reales y predichos
#
# **Hipótesis.** Si el error es de pocos miles de dólares, los puntos deberían quedar cerca de la diagonal. Las empresas atípicas en `Administration`, que conservamos en la limpieza, podrían tener errores más grandes.

# %%
# Profit real contra predicho en cada test, adaptado de "Valores reales vs Predicciones" del notebook de
# árboles de regresión de la U4. En rojo, las empresas atípicas en Administration
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, nombre in zip(axes, PARTICIONES):
    X_train, X_test, y_train, y_test = particiones_a[nombre]
    prediccion = busquedas_a[nombre].best_estimator_.predict(X_test)
    grupo = np.where(y_test.index.isin(atipicas_admin.index), "atípica en Administration", "resto")
    sns.scatterplot(x=y_test, y=prediccion, hue=grupo, hue_order=["resto", "atípica en Administration"],
                    palette=["tab:blue", "tab:red"], s=20, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="gray", linestyle="--",
            label="predicción perfecta")
    ax.set(title=f"Test {nombre}", xlabel="Profit real (USD)", ylabel="Profit predicho (USD)")
    ax.legend()
fig.suptitle("Profit real y predicho por el árbol elegido")
plt.tight_layout()
plt.show()

# %%
# Cuántas empresas del test se predicen con menos de 1.000 dólares de error, y cuánto del error cuadrático
# aportan las atípicas en Administration
filas = []
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_a[nombre]
    error = (y_test - busquedas_a[nombre].best_estimator_.predict(X_test)).abs()
    es_atipica = y_test.index.isin(atipicas_admin.index)
    filas.append({"Partición": nombre,
                  "Empresas en test": len(y_test),
                  "% con error menor a 1.000 USD": round(100 * (error < 1000).mean()),
                  "Atípicas en Administration": es_atipica.sum(),
                  "% del error cuadrático que aportan": round(100 * (error[es_atipica] ** 2).sum() / (error ** 2).sum())})
pd.DataFrame(filas).set_index("Partición")

# %%
# Empresas del test 80/20 que también están en el test 70/30
X_train_80, X_test_80, y_train_80, y_test_80 = particiones_a["80/20"]
X_train_70, X_test_70, y_train_70, y_test_70 = particiones_a["70/30"]
print(f"{y_test_80.index.isin(y_test_70.index).sum()} de las {len(y_test_80)} empresas del test 80/20 "
      "también están en el test 70/30")

# %% [markdown]
# Se cumple la hipótesis. Casi todos los puntos están sobre la diagonal: el error es menor a 1.000 dólares en el 90 % de las empresas del test 80/20 y en el 88 % del 70/30. Las atípicas en `Administration` tienen errores más grandes: son 7 de las 200 empresas del test 80/20 y aportan el 38 % del error cuadrático, y 8 de las 299 del 70/30, con el 41 %. El árbol casi no usa `Administration`, así que predice su ganancia con `RyD_Spend`, y la de esas empresas se aparta de esa relación.
#
# Las 200 empresas del test 80/20 también están en el test 70/30, y en los dos gráficos se ven los mismos puntos alejados de la diagonal. Con 70/30, esos errores grandes se promedian entre 99 empresas más, y por eso su RMSE de test es menor aunque el árbol se entrena con menos empresas.

# %% [markdown]
# ---
# # Parte B – Clasificación (dataset drugType.csv)
#
# ## 3. Preparación de datos
#
# ### 3.1 Estructura, valores faltantes y duplicados

# %%
drogas = pd.read_csv(RUTA_DROGAS, encoding="utf-8-sig")
drogas.head()

# %%
# Tipo de cada columna y cantidad de valores no nulos
drogas.info()

# %%
print(f"Filas duplicadas: {drogas.duplicated().sum()}")

# %%
# Orden natural de cada categórica, de menor a mayor; se usa en los gráficos y en la codificación
ORDEN_CATEGORIAS = {"Sexo": ["F", "M"], "BP": ["Bajo", "Normal", "Alto"], "Colesterol": ["Normal", "Alto"]}
NUMERICAS_DROGAS = ["Edad", "Na_a_K"]

# Cantidad de pacientes de cada categoría
for columna in ORDEN_CATEGORIAS:
    print(columna, drogas[columna].value_counts().to_dict())

# %% [markdown]
# Hay 200 pacientes con 5 atributos: dos numéricos (`Edad` y `Na_a_K`, la relación sodio/potasio en sangre) y tres categóricos (`Sexo`, `BP`, que es la presión arterial, y `Colesterol`). No hay valores faltantes ni filas duplicadas, y las categorías de cada atributo tienen cantidades parecidas.

# %%
# Adaptada de target_balance de U4_DesicionTreeClassifier.ipynb: devuelve la tabla en lugar de imprimirla
def balance_de_clases(y):
    """Cantidad y porcentaje de pacientes de cada droga."""
    return pd.DataFrame({"Cantidad": y.value_counts(),
                         "Porcentaje": (y.value_counts(normalize=True) * 100).round(1)})


balance_de_clases(drogas["Droga"])

# %% [markdown]
# Las clases están desbalanceadas: `DrogaY` es el 45,5 % de los pacientes, y `DrogaB` y `DrogaC`, el 8 % cada una, casi seis veces menos. Por eso las particiones se estratifican por `Droga`, como pide la consigna.

# %% [markdown]
# ### 3.2 Distribuciones por clase
#
# **Hipótesis.** Si alguno de los atributos numéricos influye en la droga que se receta, su distribución debería cambiar de una droga a otra, y alguna droga podría ocupar un rango propio.

# %%
# Histogramas de Edad y Na_a_K separados por droga, como en el notebook de árboles de clasificación de la U4
# (sección "Tipos de distribución")
ORDEN_DROGAS = sorted(drogas["Droga"].unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, columna in zip(axes, NUMERICAS_DROGAS):
    sns.histplot(data=drogas, x=columna, hue="Droga", hue_order=ORDEN_DROGAS, kde=True, element="step", ax=ax)
fig.suptitle("Atributos numéricos según la droga")
plt.tight_layout()
plt.show()

# %%
# Rango de Edad y de Na_a_K en cada droga
drogas.groupby("Droga")[NUMERICAS_DROGAS].agg(["min", "max"])

# %% [markdown]
# Se cumple la hipótesis. En el histograma de `Na_a_K`, `DrogaY` ocupa sola la parte derecha: sus pacientes tienen siempre 15,0 o más, y los de las otras cuatro drogas, 14,6 o menos. `Edad` separa a `DrogaA` de `DrogaB`: todos los pacientes de `DrogaA` tienen 50 años o menos, y todos los de `DrogaB`, 51 o más. En las otras drogas, la edad cubre casi todo el rango.

# %% [markdown]
# **Hipótesis.** La presión y el colesterol son criterios clínicos habituales para elegir un tratamiento, así que esperamos que algunas drogas aparezcan solo con ciertos valores. El sexo no debería influir.

# %%
# Cantidad de pacientes de cada droga para cada valor de las categóricas (tablas de contingencia como mapas de calor)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, columna in zip(axes, ORDEN_CATEGORIAS):
    tabla = pd.crosstab(drogas[columna], drogas["Droga"]).reindex(ORDEN_CATEGORIAS[columna])
    sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{columna} según la droga")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se cumple la hipótesis. Con presión alta solo aparecen `DrogaA`, `DrogaB` y `DrogaY`; `DrogaX` nunca tiene presión alta, y los 16 pacientes de `DrogaC` tienen presión baja y colesterol alto. El sexo se reparte de forma parecida en todas las drogas.

# %% [markdown]
# ### 3.3 Valores atípicos

# %%
contar_atipicos(drogas, NUMERICAS_DROGAS)

# %%
# Los pacientes atípicos en Na_a_K: de qué droga son y entre qué valores están
atipicos_na_k = drogas.drop(remove_outliers_iqr(drogas, "Na_a_K").index)
atipicos_na_k.groupby("Droga")["Na_a_K"].agg(["count", "min", "max"])

# %% [markdown]
# `Edad` no tiene atípicos. `Na_a_K` tiene 8, todos de `DrogaY`, con valores entre 32,9 y 38,2. Los conservamos: son valores posibles y están del lado de `Na_a_K` alto, donde solo hay pacientes de `DrogaY`, así que sacarlos solo quitaría ejemplos de esa droga.

# %% [markdown]
# ### 3.4 Codificación de las variables categóricas
#
# Las tres categóricas tienen un orden natural, así que se codifican con `OrdinalEncoder` respetando ese orden: `Sexo` F = 0 y M = 1, `BP` Bajo = 0, Normal = 1 y Alto = 2, y `Colesterol` Normal = 0 y Alto = 1. La variable objetivo queda como texto.

# %%
# OrdinalEncoder con las categorías en su orden y salida en DataFrame, como en los notebooks de Bayes y
# k-NN de la U5
codificador = OrdinalEncoder(categories=list(ORDEN_CATEGORIAS.values())).set_output(transform="pandas")
drogas_cod = drogas.copy()
drogas_cod[list(ORDEN_CATEGORIAS)] = codificador.fit_transform(drogas[list(ORDEN_CATEGORIAS)])
drogas_cod.head()

# %% [markdown]
# ### 3.5 Correlaciones
#
# **Hipótesis.** No esperamos relaciones fuertes entre los atributos: la edad, el sexo, la presión, el colesterol y la relación sodio/potasio deberían variar de forma independiente en esta muestra.

# %%
# Correlación de Spearman, porque tres de las variables son ordinales (el notebook de árboles de clasificación
# de la U4 la indica para ese caso)
correlaciones = drogas_cod.drop(columns="Droga").corr(method="spearman")
plt.figure(figsize=(6, 5))
sns.heatmap(correlaciones, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
            annot=True, fmt=".2f", square=True)
plt.title("Correlación de Spearman entre los atributos")
plt.show()

# %% [markdown]
# Se cumple la hipótesis: todas las correlaciones están entre −0,15 y 0,10, así que los atributos son casi independientes entre sí. La variable objetivo no entra en la matriz porque es nominal, con cinco clases sin orden; su relación con los atributos la vimos en los gráficos por droga.

# %% [markdown]
# ### 3.6 Particiones estratificadas y estandarización
#
# Igual que en la Parte A, pero estratificando por `Droga`: `Edad` y `Na_a_K` se estandarizan con la media y el desvío del entrenamiento de cada partición.

# %%
X_drogas = drogas_cod.drop(columns="Droga")
y_drogas = drogas_cod["Droga"]

particiones_b = {}   # nombre de la partición -> (X_train, X_test, y_train, y_test)
escaladores_b = {}   # nombre de la partición -> escalador ajustado con su entrenamiento
for nombre, proporcion_test in PARTICIONES.items():
    # stratify mantiene en los dos conjuntos la proporción de cada droga
    X_train, X_test, y_train, y_test = train_test_split(X_drogas, y_drogas, test_size=proporcion_test,
                                                        random_state=SEMILLA, stratify=y_drogas)

    escalador = StandardScaler().set_output(transform="pandas")
    escalador.fit(X_train[NUMERICAS_DROGAS])
    X_train[NUMERICAS_DROGAS] = escalador.transform(X_train[NUMERICAS_DROGAS])
    X_test[NUMERICAS_DROGAS] = escalador.transform(X_test[NUMERICAS_DROGAS])

    particiones_b[nombre] = (X_train, X_test, y_train, y_test)
    escaladores_b[nombre] = escalador
    print(f"Partición {nombre}: {len(X_train)} pacientes de entrenamiento y {len(X_test)} de test")

# %%
# Pacientes de cada droga en el entrenamiento y el test de cada partición, como el balance de clases que el
# notebook de árboles de clasificación de la U4 muestra después de dividir
conteos = pd.DataFrame()
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    conteos[f"entrenamiento {nombre}"] = balance_de_clases(y_train)["Cantidad"]
    conteos[f"test {nombre}"] = balance_de_clases(y_test)["Cantidad"]
conteos

# %%
# Pacientes del test 80/20 que también están en el test 70/30
X_train_80, X_test_80, y_train_80, y_test_80 = particiones_b["80/20"]
X_train_70, X_test_70, y_train_70, y_test_70 = particiones_b["70/30"]
print(f"{y_test_80.index.isin(y_test_70.index).sum()} de los {len(y_test_80)} pacientes del test 80/20 "
      "también están en el test 70/30")

# %%
# Entrenamiento de la partición 80/20 ya preparado
X_train, X_test, y_train, y_test = particiones_b["80/20"]
X_train.describe().round(2)

# %% [markdown]
# La estratificación mantiene las proporciones de las drogas en cada conjunto, y en el entrenamiento `Edad` y `Na_a_K` quedan con media 0 y desvío 1. Los test son chicos: 40 pacientes en la partición 80/20 y 60 en la 70/30, con solo 3 y 5 pacientes de `DrogaB` y de `DrogaC`, así que un solo paciente de esas drogas es entre el 20 % y el 33 % de su clase en el test. Además, los 40 pacientes del test 80/20 también están en el test 70/30: las dos particiones no son independientes.

# %% [markdown]
# ## 4. Árboles de decisión para clasificación
#
# Como en la Parte A, en cada partición probamos muchas combinaciones (504): `max_depth` de 1 a 6, 8, 10 y sin límite; `min_samples_leaf` de 1, 2, 3, 5, 7, 10 y 15; y `min_samples_split` de 2, 4, 6, 8, 10, 15, 20 y 30. Elegimos la de mejor promedio en la validación cruzada de 5 pliegues. Por el desbalance de clases, la métrica es el F1 macro, que pesa igual a las cinco drogas.

# %%
GRILLA_CLASIFICACION = {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, None],
                        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15],
                        "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30]}

busquedas_arbol = {}   # nombre de la partición -> GridSearchCV ya ajustado
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    busqueda = GridSearchCV(DecisionTreeClassifier(random_state=SEMILLA), GRILLA_CLASIFICACION, cv=5,
                            scoring="f1_macro")
    busqueda.fit(X_train, y_train)
    busquedas_arbol[nombre] = busqueda
    print(f"Partición {nombre}: {busqueda.best_params_}, F1 macro de validación {busqueda.best_score_:.3f}")

# %%
# F1 macro de validación según max_depth: el mínimo y el máximo entre las combinaciones de los otros dos
# hiperparámetros. Si son iguales, todas esas combinaciones dan el mismo resultado
f1_por_profundidad = pd.DataFrame()
for nombre in PARTICIONES:
    resultados = pd.DataFrame(busquedas_arbol[nombre].cv_results_)
    resultados["param_max_depth"] = resultados["param_max_depth"].fillna("sin límite")
    por_profundidad = resultados.groupby("param_max_depth", sort=False)["mean_test_score"]
    f1_por_profundidad[f"{nombre} mínimo"] = por_profundidad.min()
    f1_por_profundidad[f"{nombre} máximo"] = por_profundidad.max()
f1_por_profundidad.round(3)

# %% [markdown]
# Con `max_depth` de 1 o 2, el F1 macro no pasa de 0,517: un árbol de 2 niveles tiene a lo sumo 4 hojas, menos que las 5 drogas. Con 3 niveles llega a 0,776 (80/20) y 0,761 (70/30), y desde 4 alcanza 0,966 y 0,965, que no mejoran con más profundidad. El mínimo de cada fila (0,554 y 0,523) sale de las combinaciones que no permiten hojas chicas: con `min_samples_leaf = 15`, por ejemplo, ninguna hoja puede tener solo a los 13 pacientes de `DrogaB` o de `DrogaC` del entrenamiento (11 con 70/30).
#
# La configuración elegida es la misma en las dos particiones: `max_depth = 4`, `min_samples_leaf = 1` y `min_samples_split = 2`, la primera de las que empatan en el mejor F1.

# %% [markdown]
# ### Post-poda (ccp_alpha)
#
# Para el árbol elegido en cada partición calculamos los valores de `ccp_alpha` en los que pierde hojas, con `cost_complexity_pruning_path` (la forma de obtenerlos en scikit-learn), y medimos cada uno con la misma validación cruzada. Elegimos el de mayor F1 macro.

# %%
# Camino de poda de cada partición: para cada ccp_alpha, las hojas que le quedan al árbol y su F1 macro de validación
tablas_poda = {}
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    mejores_parametros = busquedas_arbol[nombre].best_params_
    camino = busquedas_arbol[nombre].best_estimator_.cost_complexity_pruning_path(X_train, y_train)

    filas = []
    for alpha in camino.ccp_alphas:
        podado = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha, **mejores_parametros)
        f1 = cross_val_score(podado, X_train, y_train, cv=5, scoring="f1_macro").mean()
        podado.fit(X_train, y_train)
        filas.append({"ccp_alpha": alpha, "hojas": podado.get_n_leaves(), "F1 macro validación": f1})
    tablas_poda[nombre] = pd.DataFrame(filas)

pd.concat(tablas_poda, names=["Partición", "paso"]).round(4)

# %%
# El árbol final de cada partición usa el ccp_alpha con mayor F1 macro de validación
arboles_b = {}   # nombre de la partición -> árbol final
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    poda = tablas_poda[nombre]
    alpha = poda.loc[poda["F1 macro validación"].idxmax(), "ccp_alpha"]
    arbol = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha, **busquedas_arbol[nombre].best_params_)
    arboles_b[nombre] = arbol.fit(X_train, y_train)
    print(f"Partición {nombre}: ccp_alpha = {alpha:.4f}, {arbol.get_n_leaves()} hojas")

# %% [markdown]
# Elegimos `ccp_alpha = 0`, es decir, no podar, en las dos particiones: es el de mayor F1 macro (0,966 y 0,965). La primera poda (`ccp_alpha` de 0,062 y 0,061) deja 4 hojas y baja el F1 a 0,828 (80/20) y 0,918 (70/30), y las siguientes lo bajan a 0,66 o menos.
#
# La post-poda quita hojas que ajustan el ruido del entrenamiento, y este árbol no tiene ninguna: cada hoja tiene pacientes de una sola droga, como se ve en el gráfico siguiente.

# %% [markdown]
# **Hipótesis.** Por lo que vimos en la preparación, esperamos que el árbol corte primero por `Na_a_K`, después por `BP`, y que use `Edad` y `Colesterol` para separar las drogas que comparten presión. La primera poda debería quitar la hoja que menos pacientes del entrenamiento separa.

# %%
# Árbol de la partición 80/20 antes y después de la primera poda (el segundo ccp_alpha del camino de poda)
X_train, X_test, y_train, y_test = particiones_b["80/20"]
alpha_primera_poda = tablas_poda["80/20"]["ccp_alpha"].iloc[1]
primera_poda = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha_primera_poda,
                                      **busquedas_arbol["80/20"].best_params_)
primera_poda.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [3, 2]})
plot_tree(arboles_b["80/20"], feature_names=list(X_train.columns), class_names=ORDEN_DROGAS,
          filled=True, rounded=True, impurity=False, fontsize=10, precision=2, ax=axes[0])
axes[0].set_title(f"Antes de podar: el árbol elegido, {arboles_b['80/20'].get_n_leaves()} hojas")
plot_tree(primera_poda, feature_names=list(X_train.columns), class_names=ORDEN_DROGAS,
          filled=True, rounded=True, impurity=False, fontsize=10, precision=2, ax=axes[1])
axes[1].set_title(f"Después de la primera poda (ccp_alpha = {alpha_primera_poda:.3f}): {primera_poda.get_n_leaves()} hojas")
fig.suptitle("Árbol de clasificación de la partición 80/20")
plt.show()

# %%
# Los umbrales de Edad y Na_a_K del gráfico están en desvíos estándar. En su unidad: umbral × desvío + media
# del entrenamiento. El escalador estandarizó Edad (posición 0) y Na_a_K (posición 1)
escalador = escaladores_b["80/20"]
print(f"Na_a_K <= -0.19 desvíos equivale a {-0.19 * escalador.scale_[1] + escalador.mean_[1]:.1f}")
print(f"Edad <= 0.32 desvíos equivale a {0.32 * escalador.scale_[0] + escalador.mean_[0]:.1f} años")

# %% [markdown]
# Se cumple la hipótesis sobre los cortes. El árbol son 5 reglas:
#
# - `DrogaY` si `Na_a_K` es mayor a 14,8 (los 73 pacientes del entrenamiento con ese valor son de `DrogaY`);
# - con presión alta (`BP` > 1,5), `DrogaA` hasta los 50 años y `DrogaB` desde los 51 (`Edad` ≤ 50,5);
# - con presión baja y colesterol alto, `DrogaC`;
# - en el resto, `DrogaX`.
#
# No usa `Sexo`, como anticipaba la preparación. Como en la Parte A, los umbrales de `Edad` y `Na_a_K` del gráfico están en desvíos, y los de `BP` y `Colesterol` son sobre los códigos de la codificación.
#
# También se cumple la hipótesis sobre la poda: la primera junta en una sola hoja, que predice `DrogaX`, toda la rama de presión baja o normal, y así quita la hoja de los 13 pacientes de `DrogaC`, una de las dos más chicas.

# %% [markdown]
# **Hipótesis.** Si las reglas del árbol también valen fuera del entrenamiento, esperamos muy pocos errores en el test, y que los que haya sean de pacientes cerca de algún umbral.

# %%
# Adaptada de tableResult de U4_DesicionTreeClassifier.ipynb: dibuja la matriz de confusión con un mapa de
# calor, para las dos particiones lado a lado (las métricas van en otra celda)
def graficar_matrices(modelos, titulo):
    """Matriz de confusión de cada modelo sobre el test de su partición."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, nombre in zip(axes, PARTICIONES):
        X_train, X_test, y_train, y_test = particiones_b[nombre]
        # labels fija el orden de las drogas, así aparecen todas aunque el modelo no prediga alguna
        matriz = confusion_matrix(y_test, modelos[nombre].predict(X_test), labels=ORDEN_DROGAS)
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=ORDEN_DROGAS,
                    yticklabels=ORDEN_DROGAS, cbar=False, ax=ax)
        ax.set(title=f"Test {nombre}", xlabel="Predicción", ylabel="Real")
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()


graficar_matrices(arboles_b, "Árbol de decisión: matrices de confusión")

# %%
# Las tres métricas que pide la consigna, calculadas como en Unidad5.py (precisión y exhaustividad macro)
def metricas_clasificacion(modelos):
    """Exactitud, y precisión y exhaustividad macro, de cada modelo sobre el test de su partición."""
    filas = []
    for nombre in PARTICIONES:
        X_train, X_test, y_train, y_test = particiones_b[nombre]
        prediccion = modelos[nombre].predict(X_test)
        # zero_division=0: si un modelo nunca predice una droga, su precisión se cuenta como 0
        filas.append({"Partición": nombre,
                      "Accuracy": accuracy_score(y_test, prediccion),
                      "Precision macro": precision_score(y_test, prediccion, average="macro", zero_division=0),
                      "Recall macro": recall_score(y_test, prediccion, average="macro", zero_division=0)})
    return pd.DataFrame(filas).set_index("Partición")


metricas_arbol = metricas_clasificacion(arboles_b)
metricas_arbol.round(3)

# %% [markdown]
# Se cumple la hipótesis, sin ningún error: el árbol clasifica bien a todos los pacientes de los dos test, y las tres métricas valen 1. Con test tan chicos, acertar todos no garantiza acertar siempre con pacientes nuevos; el F1 macro de validación cruzada (0,966 y 0,965) es una estimación más prudente.

# %% [markdown]
# ## 5. Bayes ingenuo
#
# Discretizamos `Edad` y `Na_a_K` en intervalos de igual frecuencia (cuantiles) y aplicamos `CategoricalNB` a los cinco atributos, que así quedan todos categóricos. Usamos `CategoricalNB` y no `MultinomialNB` porque los atributos son categorías, no conteos. La consigna pide al menos 3 y 5 intervalos: probamos de 3 a 10 y elegimos, en cada partición, la cantidad con mayor F1 macro de validación.

# %%
modelos_bayes = {}           # (partición, cantidad de intervalos) -> modelo entrenado
f1_bayes = pd.DataFrame()    # F1 macro de validación: una fila por cantidad de intervalos, una columna por partición
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    for n_intervalos in range(3, 11):
        # Discretiza Edad y Na_a_K por cuantiles. Sexo, BP y Colesterol ya son categorías y pasan sin
        # cambios (remainder="passthrough", como el make_column_transformer del apunte de la Unidad 5)
        discretizador = ColumnTransformer(
            [("intervalos", KBinsDiscretizer(n_bins=n_intervalos, encode="ordinal", strategy="quantile"),
              NUMERICAS_DROGAS)],
            remainder="passthrough")

        # Con el discretizador dentro de un pipeline (como el make_pipeline de Unidad4.py), en la validación
        # cruzada los cortes se calculan solo con los pliegues de entrenamiento
        modelo = make_pipeline(discretizador, CategoricalNB())
        f1 = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1_macro").mean()

        f1_bayes.loc[n_intervalos, nombre] = f1
        modelos_bayes[(nombre, n_intervalos)] = modelo.fit(X_train, y_train)

f1_bayes.index.name = "intervalos"
f1_bayes.round(3)

# %%
# En cada partición, la cantidad de intervalos con mayor F1 macro de validación
mejores_bayes = {}         # nombre de la partición -> modelo elegido
intervalos_elegidos = {}   # nombre de la partición -> cantidad de intervalos
for nombre in PARTICIONES:
    intervalos_elegidos[nombre] = int(f1_bayes[nombre].idxmax())
    mejores_bayes[nombre] = modelos_bayes[(nombre, intervalos_elegidos[nombre])]
    print(f"Partición {nombre}: {intervalos_elegidos[nombre]} intervalos")

# %%
# Cortes de Edad y Na_a_K para cada cantidad de intervalos, en la partición 80/20, pasados de desvíos
# estándar a su unidad (corte × desvío + media del entrenamiento)
escalador = escaladores_b["80/20"]   # estandarizó Edad (posición 0) y Na_a_K (posición 1)
for n_intervalos in range(3, 11):
    # El primer paso del pipeline es el ColumnTransformer; bin_edges_ tiene los cortes de cada columna
    intervalos = modelos_bayes[("80/20", n_intervalos)][0].named_transformers_["intervalos"]
    cortes_edad = intervalos.bin_edges_[0] * escalador.scale_[0] + escalador.mean_[0]
    cortes_na_k = intervalos.bin_edges_[1] * escalador.scale_[1] + escalador.mean_[1]
    print(f"{n_intervalos:2d} intervalos | Edad: {np.round(cortes_edad, 1).tolist()}")
    print(f"             | Na_a_K: {np.round(cortes_na_k, 1).tolist()}")

# %% [markdown]
# El F1 macro no mejora de forma pareja con más intervalos: sube y baja. Con 80/20 el mejor es con 4 intervalos (0,894) y con 70/30, con 9 (0,882); con 3 es el peor en las dos particiones (0,691 y 0,639).
#
# Los cortes explican en parte ese zigzag. El F1 depende de que algún corte caiga cerca de las dos fronteras que vimos en la preparación: `Na_a_K` alrededor de 15 (`DrogaY` contra el resto) y `Edad` entre 50 y 51 (`DrogaA` contra `DrogaB`). Con 3 intervalos, el del medio de `Na_a_K` va de 11,4 a 18,3 y mezcla a `DrogaY` con las demás drogas. Con 4 hay un corte de `Na_a_K` en 14,0, cerca de la frontera, aunque los de `Edad` caen en 46,5 y 59. Con 9 hay un corte de `Na_a_K` justo en 15,0. La discretización por igual frecuencia no conoce las drogas, así que acertar una frontera depende de la cantidad de intervalos. Además, con más intervalos cada uno tiene menos pacientes (con 10, unos 16 del entrenamiento), y las probabilidades del modelo se apoyan en menos casos.

# %% [markdown]
# **Hipótesis.** Esperamos errores en los pacientes cuyos valores caen en un intervalo que mezcla drogas. Con los 4 intervalos de la partición 80/20, el de 46,5 a 59 años junta pacientes de `DrogaA` (hasta 50) y de `DrogaB` (desde 51), y el de `Na_a_K` de 14,0 a 19,7 junta a `DrogaY` con pacientes de otras drogas.

# %%
graficar_matrices(mejores_bayes, "Bayes ingenuo: matrices de confusión")

# %%
metricas_bayes = metricas_clasificacion(mejores_bayes)
metricas_bayes.round(3)

# %%
def mal_clasificados(modelos):
    """Pacientes del test que cada modelo clasifica mal, con sus datos sin codificar."""
    filas = []
    for nombre in PARTICIONES:
        X_train, X_test, y_train, y_test = particiones_b[nombre]
        prediccion = pd.Series(modelos[nombre].predict(X_test), index=X_test.index)
        errores = prediccion[prediccion != y_test]
        for indice, droga_predicha in errores.items():
            fila = {"Partición": nombre, "Droga predicha": droga_predicha}
            fila.update(drogas.loc[indice].to_dict())   # los datos originales del paciente
            filas.append(fila)
    return pd.DataFrame(filas)


mal_clasificados(mejores_bayes)

# %% [markdown]
# Se cumple la hipótesis en tres de los cuatro errores del test 80/20. Dos pacientes de `DrogaB`, de 53 y 57 años, caen en el intervalo de 46,5 a 59 y se clasifican como `DrogaA`, y uno de `DrogaA` con `Na_a_K` de 13,97, en el borde del corte de 14,0, como `DrogaY`. Con 70/30 hay un solo error: el mismo paciente de `DrogaB` de 53 años.
#
# El cuarto error del test 80/20 muestra el supuesto de independencia del modelo: un paciente de `DrogaC`, con presión baja y colesterol alto, se clasifica como `DrogaX`. Bayes ingenuo mira la presión y el colesterol por separado, y `DrogaX` tiene muchos pacientes con presión baja (18) y muchos con colesterol alto (20), aunque esa combinación sea la de `DrogaC` en el árbol. Además, `DrogaX` es más de tres veces más frecuente que `DrogaC`.
#
# Con 80/20, la exhaustividad macro (0,760) queda muy por debajo de la precisión macro (0,906): `DrogaB` tiene 3 pacientes en el test y 2 mal clasificados, así que su exhaustividad es de 1 de 3. Con 70/30, las tres métricas son de 0,96 o más.

# %% [markdown]
# ## 6. k-NN
#
# Probamos los k impares de 1 a 31 (como el `range(1, 15, 2)` de `Unidad5.py`, pero más amplio) con las distancias euclidiana y Manhattan, y elegimos la combinación con mayor F1 macro de validación, con `GridSearchCV` y `best_estimator_` como en el notebook de k-NN de la U5. `Edad` y `Na_a_K` ya están estandarizadas, y las categóricas codificadas van de 0 a 1 o de 0 a 2, así que todos los atributos tienen escalas comparables, como pide el apunte de la Unidad 5.
#
# **Hipótesis.** Con k chico, el vecino más cercano puede ser un paciente de otra droga por un detalle que no importa (por ejemplo, el sexo). Con k grande, las drogas con pocos pacientes quedan en minoría entre los vecinos. Esperamos el mejor F1 en un k intermedio.

# %%
GRILLA_KNN = {"n_neighbors": list(range(1, 32, 2)), "metric": ["euclidean", "manhattan"]}

busquedas_knn = {}   # nombre de la partición -> GridSearchCV ya ajustado
mejores_knn = {}     # nombre de la partición -> modelo elegido
for nombre in PARTICIONES:
    X_train, X_test, y_train, y_test = particiones_b[nombre]
    busqueda = GridSearchCV(KNeighborsClassifier(), GRILLA_KNN, cv=5, scoring="f1_macro")
    busqueda.fit(X_train, y_train)
    busquedas_knn[nombre] = busqueda
    mejores_knn[nombre] = busqueda.best_estimator_
    print(f"Partición {nombre}: {busqueda.best_params_}, F1 macro de validación {busqueda.best_score_:.3f}")

# %%
# F1 macro de validación según k, una línea por distancia
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, nombre in zip(axes, PARTICIONES):
    resultados = pd.DataFrame(busquedas_knn[nombre].cv_results_)
    sns.lineplot(x=resultados["param_n_neighbors"].astype(int), y=resultados["mean_test_score"],
                 hue=resultados["param_metric"], marker="o", ax=ax)
    ax.set(title=f"Partición {nombre}", xlabel="k (n_neighbors)", ylabel="F1 macro de validación",
           xticks=GRILLA_KNN["n_neighbors"])
    ax.legend(title="distancia")
fig.suptitle("k-NN: F1 macro de validación cruzada según k y la distancia")
plt.tight_layout()
plt.show()

# %% [markdown]
# La hipótesis se cumple en parte. Con k grande el F1 cae en las dos particiones, como esperábamos, porque las drogas con pocos pacientes quedan en minoría entre los vecinos: con k = 31 queda entre 0,52 y 0,69. El máximo, en cambio, no siempre está en un k intermedio: con 70/30 sí (k = 11, F1 de 0,852), pero con 80/20 el mejor es k = 1 (0,874), apenas por encima de k = 13. Para k chicos las curvas van en zigzag: con 13 pacientes o menos de `DrogaB` y de `DrogaC` en el entrenamiento, cada pliegue de validación tiene 2 o 3 de cada una, y un solo error mueve mucho el F1 macro.
#
# La distancia Manhattan queda por encima de la euclidiana en casi todos los k, y es la elegida en las dos particiones.

# %% [markdown]
# **Hipótesis.** Esperamos errores cerca de la frontera de `Na_a_K` (alrededor de 15), donde los vecinos de un paciente pueden ser de otras drogas con la misma presión y colesterol.

# %%
graficar_matrices(mejores_knn, "k-NN: matrices de confusión")

# %%
metricas_knn = metricas_clasificacion(mejores_knn)
metricas_knn.round(3)

# %%
mal_clasificados(mejores_knn)

# %% [markdown]
# Se cumple la hipótesis en casi todos los errores. k-NN se equivoca con 5 pacientes en el test 80/20 (exactitud de 0,875, con k = 1) y con 6 en el 70/30 (0,900, con k = 11). Los pacientes de `DrogaY` mal clasificados tienen casi todos `Na_a_K` entre 15,16 y 16,59, cerca de la frontera, y reciben la droga que corresponde a su presión (`DrogaX` con presión normal y `DrogaA` con presión alta). En sentido contrario, uno de `DrogaX` con 12,86 se clasifica como `DrogaY` (con k = 1), y uno de `DrogaC` con presión baja y colesterol alto, como `DrogaX` (con k = 11).
#
# El único error lejos de la frontera es un paciente de `DrogaY` con 19,80, clasificado como `DrogaC` con k = 1: tiene presión baja y colesterol alto, el perfil de `DrogaC`, y con un solo vecino alcanza con que el más cercano sea de esa droga.

# %% [markdown]
# ## 7. Comparación de modelos

# %%
# Tabla comparativa con el mejor modelo de cada algoritmo, como la tabla de métricas del notebook de Bayes de la U5
filas = []
for nombre in PARTICIONES:
    arbol = arboles_b[nombre]
    knn = mejores_knn[nombre]
    modelos = {f"Árbol de decisión ({arbol.get_n_leaves()} hojas)": metricas_arbol,
               f"Bayes ingenuo ({intervalos_elegidos[nombre]} intervalos)": metricas_bayes,
               f"k-NN ({knn.metric}, k = {knn.n_neighbors})": metricas_knn}
    for modelo, metricas in modelos.items():
        filas.append({"Partición": nombre,
                      "Modelo": modelo,
                      "Accuracy": metricas.loc[nombre, "Accuracy"],
                      "Precision macro": metricas.loc[nombre, "Precision macro"],
                      "Recall macro": metricas.loc[nombre, "Recall macro"]})
pd.DataFrame(filas).set_index(["Partición", "Modelo"]).round(3)

# %% [markdown]
# El árbol de decisión es el mejor en las dos particiones: vale 1 en las tres métricas. Con 70/30, Bayes ingenuo queda muy cerca, con un solo error (exactitud de 0,983), y supera a k-NN (0,900). Con 80/20, Bayes y k-NN quedan más lejos (0,900 y 0,875), y Bayes tiene la exhaustividad macro más baja (0,760) por los dos pacientes de `DrogaB` que clasifica mal. Parte de la diferencia entre particiones viene de que cada una eligió otra cantidad de intervalos y otro k, y de que los test son chicos.
#
# El árbol es además el más fácil de interpretar: son 5 reglas con umbrales que se pueden leer y comprobar, como «relación sodio/potasio mayor a 14,8» o «presión alta y hasta 50 años». Bayes da probabilidades por intervalo, más difíciles de leer, y k-NN no da ninguna regla: solo puede justificar cada predicción con los pacientes parecidos.
#
# El árbol gana porque en estos datos cada droga se define con umbrales sobre un atributo a la vez, que es la forma de las fronteras que arma un árbol. Bayes depende de que los cortes por cuantiles caigan cerca de esas fronteras y mira cada atributo por separado, y k-NN mezcla todos los atributos en una distancia, incluido `Sexo`, que no ayuda a distinguir las drogas.
#
# Elegimos el árbol de decisión (`max_depth = 4`, sin post-poda): tiene las mejores métricas y es el que mejor explica sus decisiones.

# %% [markdown]
# ---
# # Conclusiones
#
# ### Parte A – Regresión
#
# La ganancia depende casi solo del gasto en I+D: su correlación con `RyD_Spend` es de 0,999, y el árbol le da a esa variable una importancia de 0,989. Sacamos una fila duplicada y las 3 empresas con ganancias de hasta 476.000 dólares, que no seguían la relación con los gastos, y conservamos las atípicas en `Administration`, que tienen montos posibles.
#
# El árbol con menor RMSE de validación tiene 156 hojas con 80/20 y 99 con 70/30. En test tiene un RMSE de 3.485 y 2.806 dólares, y un MAE de 1.046 y 940. Casi todas las empresas se predicen con menos de 1.000 dólares de error; las atípicas en `Administration` concentran cerca del 40 % del error cuadrático, porque el árbol predice casi solo con `RyD_Spend`.
#
# ### Parte B – Clasificación
#
# Las drogas siguen reglas simples: `DrogaY` si la relación sodio/potasio supera 14,8, y las demás según la presión, la edad (hasta o desde los 50 años) y el colesterol. `Sexo` no influye. Las clases están desbalanceadas (del 45,5 % al 8 %), así que estratificamos las particiones y elegimos con el F1 macro.
#
# El árbol (6 hojas, sin poda) clasifica bien todo el test en las dos particiones, y cualquier poda baja su F1 de validación. Bayes ingenuo (4 y 9 intervalos) y k-NN (Manhattan, k = 1 y 11) llegan a una exactitud de entre 0,875 y 0,983, y fallan sobre todo con pacientes cerca de las fronteras de `Na_a_K` y de `Edad`. Elegimos el árbol, por sus métricas y porque sus reglas se pueden leer y verificar.
#
# ### Las dos particiones
#
# En las dos partes, el test 80/20 está contenido en el 70/30, así que los casos difíciles son los mismos y las diferencias entre particiones dependen también de cuántos casos fáciles se suman. En la Parte B, con test de 40 y 60 pacientes, un solo error mueve varios puntos las métricas macro; la validación cruzada da una estimación más estable.
