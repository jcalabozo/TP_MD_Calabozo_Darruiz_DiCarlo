# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trabajo Práctico N.º 2 — Árboles de decisión, Bayes ingenuo y k-NN
#
# **Minería de Datos · TUIA · UNR · Año 2026**
#
# **Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo
#
# **Datos de la cátedra:** `data/1000_Companies.csv` (1000 empresas, para la Parte A) y `data/drugType.csv` (200 pacientes, para la Parte B).
#
# El notebook sigue las siete actividades del enunciado, con sus mismos títulos:
#
# | Parte | Actividad | Modelo |
# |---|---|---|
# | A. Regresión | 1. Preparación de datos | — |
# | | 2. Árboles de decisión para regresión | `DecisionTreeRegressor` |
# | B. Clasificación | 3. Preparación de datos | — |
# | | 4. Árboles de decisión para clasificación | `DecisionTreeClassifier` con post-poda |
# | | 5. Bayes ingenuo | `CategoricalNB` sobre atributos discretizados |
# | | 6. k-NN | `KNeighborsClassifier` |
# | | 7. Comparación de modelos | — |
#
# **Cómo ejecutarlo:** ver el [README](../README.md).

# %% [markdown]
# ---
# # Preparación

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Semilla que pide la consigna para las particiones y los modelos: con ella, cada corrida da lo mismo
SEMILLA = 63

# Las dos particiones entrenamiento/test que pide la consigna: nombre → proporción de test
PARTICIONES = {"80/20": 0.2, "70/30": 0.3}

# Rutas relativas a tp2/notebooks/, la carpeta del notebook (ver el README)
RUTA_EMPRESAS = "../data/1000_Companies.csv"
RUTA_DROGAS = "../data/drugType.csv"
RUTA_FIGURAS = "../docs/figuras"   # figuras que usa el informe (docs/informe.md)
os.makedirs(RUTA_FIGURAS, exist_ok=True)


def guardar_figura(nombre):
    """Guarda la figura actual en docs/figuras/ para usarla en el informe."""
    plt.savefig(f"{RUTA_FIGURAS}/{nombre}.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ---
# # Parte A — Regresión (dataset 1000_Companies.csv)
#
# ## 1. Preparación de datos
#
# ### Estructura, valores faltantes y duplicados

# %%
# El archivo empieza con una marca BOM: con "utf-8-sig" la primera columna se lee como "RyD_Spend"
empresas = pd.read_csv(RUTA_EMPRESAS, encoding="utf-8-sig")
empresas.info()
print(f"\nFilas duplicadas: {empresas.duplicated().sum()}")
print(empresas[empresas.duplicated(keep=False)])

# %%
NUMERICAS_EMPRESAS = ["RyD_Spend", "Administration", "MarketingSpend", "Profit"]

print(empresas["State"].value_counts(), "\n")
gasto_cero = (empresas[["RyD_Spend", "MarketingSpend"]] == 0).any(axis=1)
print(f"Filas con gasto 0 en RyD_Spend o MarketingSpend: {empresas.index[gasto_cero].tolist()}")
empresas[NUMERICAS_EMPRESAS].describe().round(0)

# %% [markdown]
# **Qué muestra**
#
# - 1000 empresas, 4 variables numéricas (`RyD_Spend`, `Administration`, `MarketingSpend` y el objetivo `Profit`) y una categórica, `State`, con tres estados en proporciones parecidas (344, 334 y 322).
# - **No hay valores faltantes.** Hay **una fila duplicada** (la 755 repite la 335), que se elimina en la limpieza.
# - Hay 4 empresas con gasto 0 en I+D o en marketing (filas 19, 47, 48 y 49). Son valores posibles (una empresa puede no invertir en eso), así que se conservan.
# - El máximo de `Profit` (476.485) y el de `Administration` (321.652) están muy lejos del tercer cuartil. Lo analizamos en los atípicos.

# %% [markdown]
# ### Distribuciones
#
# **Hipótesis.** Los tres gastos y la ganancia son montos de empresas de distintos tamaños: esperamos distribuciones amplias, sin una forma marcada, y quizás algunos valores extremos en `Profit`.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, columna in zip(axes, NUMERICAS_EMPRESAS):
    sns.histplot(data=empresas, x=columna, kde=True, element="step", ax=ax)
fig.suptitle("Distribución de las variables numéricas")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Qué muestra**
#
# - `RyD_Spend` y `MarketingSpend` se reparten de forma bastante pareja en todo su rango, como esperábamos.
# - `Administration`, en cambio, está muy concentrada entre 110.000 y 135.000, con unos pocos valores sueltos a los dos lados.
# - `Profit` está casi toda entre 40.000 y 190.000, pero tiene unos pocos valores aislados mucho más altos, que estiran el eje hasta 476.000. Es lo que anticipaba la hipótesis.

# %% [markdown]
# ### Valores atípicos
#
# **Hipótesis.** Si los valores extremos de `Profit` son errores, deberían aparecer como atípicos en el boxplot y no seguir la relación con los gastos que tiene el resto.

# %%
def limites_iqr(serie):
    """Límites inferior y superior de la regla de 1,5 veces el rango intercuartílico."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def filas_atipicas(df, columna):
    """Índices de las filas que quedan fuera de los límites IQR de la columna."""
    inferior, superior = limites_iqr(df[columna])
    return df.index[(df[columna] < inferior) | (df[columna] > superior)]


fig, axes = plt.subplots(1, 4, figsize=(16, 2.8))
for ax, columna in zip(axes, NUMERICAS_EMPRESAS):
    sns.boxplot(data=empresas, x=columna, ax=ax)
fig.suptitle("Valores atípicos (regla del rango intercuartílico)")
plt.tight_layout()
plt.show()

for columna in NUMERICAS_EMPRESAS:
    print(f"{columna}: {len(filas_atipicas(empresas, columna))} atípicos")

atipicos_profit = filas_atipicas(empresas, "Profit")
empresas.loc[atipicos_profit]

# %% [markdown]
# **Qué muestra**
#
# - **`Profit` tiene 3 atípicos: las últimas tres filas del archivo (997 a 999).** Su ganancia va de 334.000 a 476.000, hasta 2,5 veces la mayor del resto (192.000). Además, tienen los tres valores más altos de `Administration` (de 242.000 a 322.000), muy por encima de cualquier otra empresa.
# - No siguen la relación con los gastos que muestran las demás: la fila 999 gasta en I+D menos que la fila 0 y gana 2,5 veces más. Los tratamos como errores de carga y **los eliminamos**. Se confirma la hipótesis.
# - `Administration` tiene 22 atípicos. Tres son esas mismas filas; los otros 19 los miramos después de limpiar.

# %% [markdown]
# ### Limpieza

# %%
# Una empresa repetida no aporta información nueva y pesaría el doble en el entrenamiento
empresas_limpio = empresas.drop_duplicates()
empresas_limpio = empresas_limpio.drop(atipicos_profit)
print(f"{len(empresas)} filas → {len(empresas_limpio)} filas "
      f"(1 duplicada y {len(atipicos_profit)} atípicas de Profit)")

# ¿Dónde quedan los atípicos de Administration? Comparamos las primeras 50 filas del archivo con el resto
atipicos_admin = filas_atipicas(empresas_limpio, "Administration")
print(f"\nAtípicos de Administration después de limpiar: {len(atipicos_admin)}, en las filas {atipicos_admin.tolist()}")
grupo = np.where(empresas_limpio.index < 50, "Filas 0 a 49", "Filas 50 en adelante")
empresas_limpio.groupby(grupo)["Administration"].describe().round(0)

# %% [markdown]
# **Qué muestra**
#
# - Quedan **996 empresas**.
# - **Los 19 atípicos de `Administration` que quedan están todos entre las primeras 50 filas del archivo.** En esas filas la variable varía mucho más: su desvío es de 28.000, contra 6.900 en el resto, y su mínimo es de 51.000, contra 111.000. Las primeras 50 filas parecen venir de otra fuente, más heterogénea (también son las que tienen los gastos en 0).
# - Son montos posibles para un gasto de administración, así que **los conservamos**. Además, a un árbol de decisión no lo desvían los valores extremos de las variables de entrada: solo cambian en qué lado de un corte cae cada empresa.

# %% [markdown]
# ### Correlaciones
#
# **Hipótesis.** Esperamos que la ganancia crezca con los tres gastos, y que `RyD_Spend` y `MarketingSpend` estén correlacionados entre sí, porque una empresa más grande gasta más en todo.

# %%
correlaciones = empresas_limpio[NUMERICAS_EMPRESAS].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(correlaciones, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
            annot=True, fmt=".2f", square=True)
plt.title("Correlación de Pearson (datos limpios)")
plt.show()

# %% [markdown]
# **Qué muestra**
#
# - **`Profit` está casi perfectamente correlacionada con `RyD_Spend` (0,999)**, muy correlacionada con `MarketingSpend` (0,98) y bastante menos con `Administration` (0,75). La ganancia crece con los tres gastos, como esperábamos, pero prácticamente queda determinada por el gasto en I+D.
# - `RyD_Spend` y `MarketingSpend` también están muy correlacionadas entre sí (0,98): se cumple la segunda hipótesis. Las dos variables cuentan casi lo mismo, así que es esperable que el árbol use sobre todo una.
# - Calculamos la correlación después de limpiar porque los 3 atípicos la distorsionaban: con ellos, la de `Profit` con `RyD_Spend` daba 0,945.

# %% [markdown]
# ### Codificación de la variable categórica

# %%
# State es nominal. Un árbol puede aislar cualquiera de los tres estados con a lo sumo dos cortes,
# así que alcanza con un código numérico por estado
codificador_estado = OrdinalEncoder()
empresas_limpio[["State"]] = codificador_estado.fit_transform(empresas_limpio[["State"]])
print("Códigos de State:", dict(enumerate(codificador_estado.categories_[0])))
empresas_limpio.head()


# %% [markdown]
# ### Particiones 80/20 y 70/30 y estandarización
#
# Las numéricas se estandarizan **después** de dividir: el escalador se ajusta solo con el entrenamiento de cada partición, para que el test no influya en nada de lo que aprende el modelo. `Profit` no se estandariza, para que los errores queden en dólares.

# %%
def particionar(X, y, proporcion_test, numericas, estratificar):
    """Divide en entrenamiento y test, y estandariza las numéricas con la media y el desvío del entrenamiento."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=proporcion_test, random_state=SEMILLA, stratify=y if estratificar else None)
    escalador = StandardScaler().fit(X_train[numericas])
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[numericas] = escalador.transform(X_train[numericas])
    X_test[numericas] = escalador.transform(X_test[numericas])
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "escalador": escalador}


GASTOS = ["RyD_Spend", "Administration", "MarketingSpend"]
X_empresas = empresas_limpio.drop(columns="Profit")
y_empresas = empresas_limpio["Profit"]

particiones_a = {nombre: particionar(X_empresas, y_empresas, proporcion, GASTOS, estratificar=False)
                 for nombre, proporcion in PARTICIONES.items()}
for nombre, p in particiones_a.items():
    print(f"Partición {nombre}: {len(p['X_train'])} empresas de entrenamiento y {len(p['X_test'])} de test")

# %% [markdown]
# ## 2. Árboles de decisión para regresión
#
# **Cómo elegimos la configuración.** En cada partición probamos las 36 combinaciones de la grilla con validación cruzada de 5 particiones **sobre el entrenamiento**; el test se usa una sola vez, al final, para reportar. Medimos el RMSE de validación, que está en dólares.
#
# Si varias configuraciones quedan a menos del 1 % del mejor RMSE (unos 22 dólares, frente a un desvío de `Profit` de 41.000), las consideramos empatadas y nos quedamos con la de **menos hojas**: rinde lo mismo y se ajusta menos al ruido del entrenamiento.

# %%
GRILLA_REGRESION = {"max_depth": [3, 5, 10, None],
                    "min_samples_leaf": [1, 5, 10],
                    "min_samples_split": [2, 10, 20]}


def buscar_arbol_regresion(p):
    """RMSE de validación cruzada (5 particiones del entrenamiento) y hojas de cada configuración de la grilla."""
    busqueda = GridSearchCV(DecisionTreeRegressor(random_state=SEMILLA), GRILLA_REGRESION, cv=5,
                            scoring="neg_mean_squared_error", return_train_score=True)
    busqueda.fit(p["X_train"], p["y_train"])
    resultados = pd.DataFrame({"params": busqueda.cv_results_["params"]})
    for parametro in GRILLA_REGRESION:
        resultados[parametro] = busqueda.cv_results_[f"param_{parametro}"]
    # GridSearchCV devuelve el MSE con signo negativo (abs lo vuelve positivo); su raíz es el RMSE, en dólares
    resultados["RMSE validación"] = np.sqrt(np.abs(busqueda.cv_results_["mean_test_score"]))
    resultados["RMSE entrenamiento"] = np.sqrt(np.abs(busqueda.cv_results_["mean_train_score"]))
    resultados["hojas"] = [DecisionTreeRegressor(random_state=SEMILLA, **params)
                           .fit(p["X_train"], p["y_train"]).get_n_leaves()
                           for params in resultados["params"]]
    return resultados


def elegir_configuracion(resultados):
    """Entre las configuraciones a menos del 1 % del mejor RMSE de validación, la de menos hojas."""
    mejor_rmse = resultados["RMSE validación"].min()
    empatadas = resultados[resultados["RMSE validación"] <= 1.01 * mejor_rmse]
    return empatadas.sort_values(["hojas", "RMSE validación"]).iloc[0]


resultados_a, elegidas_a = {}, {}
for nombre, p in particiones_a.items():
    resultados_a[nombre] = buscar_arbol_regresion(p)
    elegidas_a[nombre] = elegir_configuracion(resultados_a[nombre])
    print(f"Partición {nombre}: las 6 configuraciones con menor RMSE de validación")
    display(resultados_a[nombre].drop(columns="params").sort_values("RMSE validación").head(6).round(0))
    print(f"Elegida: {elegidas_a[nombre]['params']}\n")

# %% [markdown]
# **Qué muestra**
#
# - **En las dos particiones ganan los árboles profundos** (`max_depth` de 10 o sin límite).
# - **Con 80/20, el menor RMSE de validación (2.176) es de un árbol con `min_samples_leaf = 1` y 601 hojas**, que casi memoriza el entrenamiento: su RMSE de entrenamiento es de 31 dólares. Con `min_samples_leaf = 5`, el RMSE de validación es de 2.179, apenas 3 dólares más, con 130 hojas. Están empatados, y elegimos el más simple.
# - Con 70/30 gana directamente `min_samples_leaf = 5` (2.261). **La configuración elegida es la misma en las dos particiones:** `max_depth = 10`, `min_samples_leaf = 5` y `min_samples_split = 2`.
# - Las configuraciones aparecen empatadas de a cuatro porque dan el mismo árbol. Con hojas de al menos 5 empresas, el árbol nunca llega a profundidad 10, así que `max_depth` de 10 o sin límite da lo mismo. Y `min_samples_split` de 2 o de 10 tampoco cambia nada, porque un nodo con menos de 10 empresas no se puede dividir en dos hojas de 5.

# %% [markdown]
# **Hipótesis.** Un árbol más profundo aproxima mejor una relación continua como la de `Profit` con los gastos, así que esperamos que el RMSE baje con `max_depth` hasta estabilizarse. Hojas muy chicas (`min_samples_leaf = 1`) deberían empezar a ajustar el ruido.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (nombre, resultados) in zip(axes, resultados_a.items()):
    # Para cada profundidad y tamaño mínimo de hoja, el mejor valor de min_samples_split
    tabla = (resultados.assign(max_depth=resultados["max_depth"].fillna("sin límite").astype(str))
             .groupby(["max_depth", "min_samples_leaf"], sort=False)["RMSE validación"].min().unstack())
    tabla.plot(ax=ax, marker="o")
    ax.set_xticks(range(len(tabla)), tabla.index)
    ax.set(title=f"Partición {nombre}", xlabel="max_depth", ylabel="RMSE de validación (USD)")
    ax.legend(title="min_samples_leaf")
fig.suptitle("Error de validación cruzada según la profundidad y el tamaño mínimo de hoja")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Qué muestra**
#
# - **Se cumple la hipótesis sobre la profundidad:** el RMSE baja de unos 5.400 a 5.600 dólares con `max_depth = 3` a unos 2.200 a 2.500 con 5 o más niveles, y desde 10 ya no cambia.
# - **Las hojas de una sola empresa empeoran el error, sobre todo con 70/30:** ahí el RMSE con `min_samples_leaf = 1` queda cerca de 3.000, contra 2.261 con 5. Hojas de 10 empresas también son un poco peores que de 5, porque cada hoja promedia empresas con ganancias más distintas.

# %%
def metricas_regresion(y_real, y_pred):
    """MAE, MSE y RMSE, las métricas que pide la consigna."""
    mse = mean_squared_error(y_real, y_pred)
    return {"MAE": mean_absolute_error(y_real, y_pred), "MSE": mse, "RMSE": np.sqrt(mse)}


arboles_a, filas = {}, []
for nombre, p in particiones_a.items():
    params = elegidas_a[nombre]["params"]
    arbol = DecisionTreeRegressor(random_state=SEMILLA, **params).fit(p["X_train"], p["y_train"])
    arboles_a[nombre] = arbol
    filas.append({"Partición": nombre, **params, "Hojas": arbol.get_n_leaves(),
                  "RMSE validación": elegidas_a[nombre]["RMSE validación"],
                  "RMSE entrenamiento": metricas_regresion(p["y_train"], arbol.predict(p["X_train"]))["RMSE"],
                  **metricas_regresion(p["y_test"], arbol.predict(p["X_test"]))})

print("Mejor árbol de cada partición. MAE, MSE y RMSE sobre su test (USD y USD²):")
metricas_a = pd.DataFrame(filas).set_index("Partición")
metricas_a.style.format("{:,.0f}", subset=["RMSE validación", "RMSE entrenamiento", "MAE", "MSE", "RMSE"])


# %% [markdown]
# **Qué muestra**
#
# - **En test, el mejor árbol tiene un RMSE de 3.492 dólares con 80/20 y de 2.766 con 70/30**, y un MAE de 1.100 y 925. Son menos de una décima parte del desvío de `Profit` (unos 41.000).
# - El RMSE de entrenamiento (1.315 y 1.439) es menor que el de validación (2.179 y 2.261) y el de test. Hay algo de sobreajuste, el esperable en un árbol de más de 100 hojas, pero mucho menos que con hojas de una empresa.
# - **El RMSE es unas tres veces el MAE.** Como el RMSE pesa más los errores grandes, eso indica que unas pocas empresas concentran casi todo el error. Lo vemos en el gráfico de valores reales y predichos.
# - 70/30 tiene menos error en test aunque entrena con menos empresas. Más abajo vemos que la diferencia depende de qué empresas caen en cada test.

# %% [markdown]
# ### El árbol con la mejor configuración
#
# **Hipótesis.** Por las correlaciones, esperamos que el primer corte sea por `RyD_Spend`.

# %%
def umbrales_originales(arbol, p, numericas, niveles):
    """Umbrales de los primeros niveles del árbol, pasados de desvíos estándar a la unidad original."""
    filas = []
    pendientes = [(0, 0)]   # (nodo, nivel), empezando por la raíz
    while pendientes:
        nodo, nivel = pendientes.pop(0)
        indice_variable = arbol.tree_.feature[nodo]
        if indice_variable < 0 or nivel >= niveles:   # es una hoja o está más abajo de lo pedido
            continue
        variable = p["X_train"].columns[indice_variable]
        umbral = arbol.tree_.threshold[nodo]
        if variable in numericas:
            j = numericas.index(variable)
            umbral = umbral * p["escalador"].scale_[j] + p["escalador"].mean_[j]
        filas.append({"nivel": nivel, "variable": variable, "umbral": umbral})
        pendientes += [(arbol.tree_.children_left[nodo], nivel + 1), (arbol.tree_.children_right[nodo], nivel + 1)]
    return pd.DataFrame(filas)


arbol, p = arboles_a["80/20"], particiones_a["80/20"]
plt.figure(figsize=(22, 8))
plot_tree(arbol, feature_names=list(p["X_train"].columns), max_depth=3, filled=True, rounded=True,
          fontsize=9, precision=2)
plt.title(f"Árbol de regresión, partición 80/20 ({arbol.get_n_leaves()} hojas; se muestran los 3 primeros niveles)")
guardar_figura("arbol_regresion")
plt.show()

print("Importancia de cada variable:")
print(pd.Series(arbol.feature_importances_, index=p["X_train"].columns).round(3).to_string())
print("\nLos umbrales del gráfico están en desvíos estándar; en dólares, los de los primeros tres niveles son:")
umbrales_originales(arbol, p, GASTOS, niveles=3).style.format({"umbral": "{:,.0f}"})

# %% [markdown]
# **Qué muestra**
#
# - **Se cumple la hipótesis: el primer corte es por `RyD_Spend`**, en 0,03 desvíos, que son unos **82.000 dólares**. Los tres primeros niveles cortan solo por `RyD_Spend`, en 41.000 y 120.000 y después en 21.000, 61.000, 99.000 y 139.000 dólares, y dividen el rango del gasto en I+D en 8 tramos parecidos. `MarketingSpend` y `Administration` recién aparecen en el cuarto nivel.
# - **La importancia de `RyD_Spend` es 0,99.** `MarketingSpend` y `Administration` aportan menos de 0,01 cada una, y `State` no se usa. Es coherente con la correlación de 0,999 entre `Profit` y `RyD_Spend`.
# - Cada hoja predice el promedio de `Profit` de sus empresas, así que el árbol aproxima una relación continua con escalones. Por eso necesita 130 hojas para bajar el error a unos pocos miles de dólares.

# %% [markdown]
# ### Valores reales y predichos
#
# **Hipótesis.** Con un RMSE de unos pocos miles de dólares, los puntos deberían quedar casi sobre la diagonal. Como el RMSE pesa más los errores grandes y es mucho mayor que el MAE, esperamos que unas pocas empresas concentren casi todo el error.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nombre, p) in zip(axes, particiones_a.items()):
    prediccion = arboles_a[nombre].predict(p["X_test"])
    primeras_50 = p["y_test"].index < 50
    ax.scatter(p["y_test"][~primeras_50], prediccion[~primeras_50], s=12, alpha=0.6, label="filas 50 en adelante")
    ax.scatter(p["y_test"][primeras_50], prediccion[primeras_50], s=25, color="tab:red", label="filas 0 a 49")
    extremos = [p["y_test"].min(), p["y_test"].max()]
    ax.plot(extremos, extremos, color="gray", linestyle="--", label="predicción perfecta")
    ax.set(title=f"Test {nombre}", xlabel="Profit real (USD)", ylabel="Profit predicho (USD)")
    ax.legend()
fig.suptitle("Profit real y predicho por el mejor árbol")
plt.tight_layout()
plt.show()

for nombre, p in particiones_a.items():
    error = (p["y_test"] - arboles_a[nombre].predict(p["X_test"])).abs()
    primeras_50 = p["y_test"].index < 50
    peso = (error[primeras_50] ** 2).sum() / (error ** 2).sum()
    print(f"Test {nombre}: {primeras_50.sum()} empresas de las filas 0 a 49 de {len(error)}, "
          f"que aportan el {peso:.0%} del error cuadrático. "
          f"Mayores errores en las filas {error.nlargest(5).index.tolist()}. "
          f"Error menor a 1.000 USD en el {(error < 1000).mean():.0%} de las empresas.")

# %% [markdown]
# **Qué muestra**
#
# - **Se cumple la hipótesis:** casi todos los puntos están sobre la diagonal. En el 88 % de las empresas de los dos test, el error es menor a 1.000 dólares.
# - **Los errores grandes son todos de empresas de las filas 0 a 49 (en rojo).** En el test 80/20 hay 14 de esas empresas, y aportan el 98 % del error cuadrático; en el 70/30 hay 16, y aportan el 96 %. Los cinco mayores errores son de las filas 45, 38, 14, 3 y 7.
# - En las filas 50 en adelante, la ganancia queda prácticamente determinada por los gastos, y el árbol la reproduce casi exacta. Las primeras 50 son más heterogéneas, como ya vimos en `Administration`. Además, el entrenamiento tiene solo unas 35 de ellas, que no alcanzan para aprender sus desvíos.
# - Por eso **la diferencia entre las dos particiones depende de cuáles de esas 50 empresas caen en cada test**, y no tanto de cuántas empresas hay en el entrenamiento.

# %% [markdown]
# ---
# # Parte B — Clasificación (dataset drugType.csv)
#
# ## 3. Preparación de datos
#
# ### Estructura, valores faltantes y duplicados

# %%
drogas = pd.read_csv(RUTA_DROGAS, encoding="utf-8-sig")
drogas.info()
print(f"\nFilas duplicadas: {drogas.duplicated().sum()}")
drogas.head()

# %%
# Orden natural de cada categórica, de menor a mayor; se usa en los gráficos y en la codificación
ORDEN_CATEGORIAS = {"Sexo": ["F", "M"], "BP": ["Bajo", "Normal", "Alto"], "Colesterol": ["Normal", "Alto"]}
NUMERICAS_DROGAS = ["Edad", "Na_a_K"]
ORDEN_DROGAS = sorted(drogas["Droga"].unique())

for columna in ORDEN_CATEGORIAS:
    print(drogas[columna].value_counts().to_dict())

print("\nDistribución de la variable objetivo:")
pd.DataFrame({"Cantidad": drogas["Droga"].value_counts(),
              "Porcentaje": drogas["Droga"].value_counts(normalize=True).mul(100).round(1)})

# %% [markdown]
# **Qué muestra**
#
# - 200 pacientes con 5 atributos: dos numéricos (`Edad` y `Na_a_K`, la relación sodio/potasio en sangre) y tres categóricos (`Sexo`, `BP`, la presión arterial, y `Colesterol`).
# - **No hay valores faltantes ni filas duplicadas.**
# - **Las clases están desbalanceadas:** `DrogaY` es el 45,5 % de los pacientes y `DrogaB` y `DrogaC`, el 8 % cada una (5,7 veces menos). Por eso las particiones se estratifican por `Droga`, y elegimos los hiperparámetros con el F1 macro, que pesa igual a todas las clases.

# %% [markdown]
# ### Distribuciones por clase
#
# **Hipótesis.** Si alguna droga se receta según un análisis de sangre, la relación sodio/potasio debería ser distinta en esa clase. La edad podría separar drogas indicadas para pacientes jóvenes o mayores.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, columna in zip(axes, NUMERICAS_DROGAS):
    sns.boxplot(data=drogas, x="Droga", y=columna, order=ORDEN_DROGAS, ax=ax)
fig.suptitle("Atributos numéricos según la droga")
plt.tight_layout()
plt.show()

drogas.groupby("Droga")[NUMERICAS_DROGAS].agg(["min", "max"])

# %% [markdown]
# **Qué muestra**
#
# - **`Na_a_K` separa a `DrogaY` de todas las demás:** sus pacientes tienen siempre un valor de 15,0 o más, y los de las otras cuatro drogas, de 14,6 o menos. Se confirma la hipótesis.
# - **`Edad` separa a `DrogaA` de `DrogaB`:** todos los pacientes de `DrogaA` tienen 50 años o menos, y todos los de `DrogaB`, 51 o más. En las otras drogas la edad cubre todo el rango.

# %% [markdown]
# **Hipótesis.** La presión y el colesterol son criterios clínicos habituales para elegir una droga, así que esperamos que algunas clases aparezcan solo con ciertos valores. El sexo no debería influir.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, columna in zip(axes, ORDEN_CATEGORIAS):
    tabla = pd.crosstab(drogas[columna], drogas["Droga"]).reindex(ORDEN_CATEGORIAS[columna])
    sns.heatmap(tabla, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{columna} según la droga")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Qué muestra**
#
# - **La presión arterial ordena casi todo:** con presión alta solo aparecen `DrogaA`, `DrogaB` y `DrogaY`; `DrogaX` nunca tiene presión alta, y `DrogaC` solo aparece con presión baja.
# - **`DrogaC` aparece solo con colesterol alto** (y presión baja), así que esa combinación la distingue de `DrogaX`.
# - **El sexo se reparte de forma parecida en todas las drogas.** Se cumple la hipótesis: no debería ayudar a clasificar.

# %% [markdown]
# ### Valores atípicos

# %%
for columna in NUMERICAS_DROGAS:
    filas = filas_atipicas(drogas, columna)
    print(f"{columna}: {len(filas)} atípicos", drogas.loc[filas, "Droga"].value_counts().to_dict())
    if len(filas) > 0:
        print(f"   valores entre {drogas.loc[filas, columna].min()} y {drogas.loc[filas, columna].max()}")

# %% [markdown]
# **Qué muestra**
#
# - `Edad` no tiene atípicos. `Na_a_K` tiene 8, todos de `DrogaY`, con valores entre 32,9 y 38,2.
# - **Los conservamos:** son valores posibles y caen del lado de `Na_a_K` alto que identifica a `DrogaY`. Sacarlos solo quitaría ejemplos de esa clase.

# %% [markdown]
# ### Codificación de las variables categóricas
#
# Las tres categóricas tienen un orden natural, así que se codifican con ese orden: `Sexo` F = 0 y M = 1, `BP` Bajo = 0, Normal = 1 y Alto = 2, y `Colesterol` Normal = 0 y Alto = 1. La variable objetivo queda como texto.

# %%
codificador = OrdinalEncoder(categories=list(ORDEN_CATEGORIAS.values()))
drogas_cod = drogas.copy()
drogas_cod[list(ORDEN_CATEGORIAS)] = codificador.fit_transform(drogas[list(ORDEN_CATEGORIAS)])
drogas_cod.head()

# %% [markdown]
# ### Correlaciones
#
# **Hipótesis.** No esperamos relaciones fuertes entre los atributos: la edad, el sexo, la presión, el colesterol y la relación sodio/potasio deberían variar de forma independiente en esta muestra.

# %%
# Spearman, porque tres de las variables son ordinales
correlaciones = drogas_cod.drop(columns="Droga").corr(method="spearman")
plt.figure(figsize=(6, 5))
sns.heatmap(correlaciones, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
            annot=True, fmt=".2f", square=True)
plt.title("Correlación de Spearman entre los atributos")
plt.show()

# %% [markdown]
# **Qué muestra**
#
# - **Todas las correlaciones están entre −0,15 y 0,10:** los atributos son prácticamente independientes entre sí, como esperábamos. Es un buen escenario para Bayes ingenuo, que supone justamente eso.
# - La variable objetivo no entra en la matriz porque es nominal, con 5 clases sin orden. Su relación con los atributos ya la vimos en los gráficos por clase.

# %% [markdown]
# ### Particiones estratificadas y estandarización
#
# Igual que en la Parte A: se divide con `random_state = 63`, ahora estratificando por `Droga`, y se estandarizan `Edad` y `Na_a_K` con la media y el desvío del entrenamiento.

# %%
X_drogas = drogas_cod.drop(columns="Droga")
y_drogas = drogas_cod["Droga"]

particiones_b = {nombre: particionar(X_drogas, y_drogas, proporcion, NUMERICAS_DROGAS, estratificar=True)
                 for nombre, proporcion in PARTICIONES.items()}

print("Pacientes de cada droga en el entrenamiento y el test de cada partición:")
pd.DataFrame({f"{parte} {nombre}": p[f"y_{parte}"].value_counts()
              for nombre, p in particiones_b.items() for parte in ["train", "test"]})


# %% [markdown]
# **Qué muestra**
#
# - La estratificación mantiene las proporciones de las clases en cada conjunto.
# - **Los test son chicos:** 40 pacientes en la partición 80/20 y 60 en la 70/30, con solo 3 y 5 pacientes de `DrogaB` y de `DrogaC`. Un solo error en una de esas clases baja su exhaustividad entre 20 y 33 puntos, y eso mueve bastante los promedios macro.

# %%
def metricas_clasificacion(y_real, y_pred):
    """Accuracy, y Precision y Recall macro, las métricas que pide la consigna."""
    # zero_division=0: si un modelo nunca predice una clase, su precisión se cuenta como 0
    return {"Accuracy": accuracy_score(y_real, y_pred),
            "Precision macro": precision_score(y_real, y_pred, average="macro", zero_division=0),
            "Recall macro": recall_score(y_real, y_pred, average="macro", zero_division=0)}


def tabla_metricas(modelos):
    """Métricas de cada modelo sobre el test de su partición."""
    return pd.DataFrame({nombre: metricas_clasificacion(particiones_b[nombre]["y_test"],
                                                        modelo.predict(particiones_b[nombre]["X_test"]))
                         for nombre, modelo in modelos.items()}).T


def graficar_matrices(modelos, titulo):
    """Matrices de confusión de los modelos de cada partición sobre su test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        p = particiones_b[nombre]
        ConfusionMatrixDisplay.from_predictions(p["y_test"], modelo.predict(p["X_test"]), labels=ORDEN_DROGAS,
                                                cmap="Blues", colorbar=False, ax=ax)
        ax.set(title=f"Test {nombre}", xlabel="Droga predicha", ylabel="Droga real")
    fig.suptitle(titulo)
    plt.tight_layout()


def mal_clasificados(modelos):
    """Pacientes del test que cada modelo clasifica mal, con sus datos originales."""
    filas = []
    for nombre, modelo in modelos.items():
        p = particiones_b[nombre]
        prediccion = pd.Series(modelo.predict(p["X_test"]), index=p["X_test"].index)
        for indice, droga_predicha in prediccion[prediccion != p["y_test"]].items():
            filas.append({"Partición": nombre, "Droga predicha": droga_predicha, **drogas.loc[indice].to_dict()})
    return pd.DataFrame(filas)


# %% [markdown]
# ## 4. Árboles de decisión para clasificación
#
# Como en la Parte A, la configuración se elige con validación cruzada de 5 particiones sobre el entrenamiento. La métrica es el **F1 macro**, que combina la precisión y la exhaustividad macro que pide la consigna y no deja que `DrogaY` domine el promedio.

# %%
GRILLA_CLASIFICACION = {"max_depth": [2, 4, 6, None],
                        "min_samples_leaf": [1, 3, 5],
                        "min_samples_split": [2, 5, 10]}

parametros_arbol = {}
for nombre, p in particiones_b.items():
    busqueda = GridSearchCV(DecisionTreeClassifier(random_state=SEMILLA), GRILLA_CLASIFICACION, cv=5,
                            scoring="f1_macro")
    busqueda.fit(p["X_train"], p["y_train"])
    parametros_arbol[nombre] = busqueda.best_params_

    resultados = pd.DataFrame({parametro: busqueda.cv_results_[f"param_{parametro}"] for parametro in GRILLA_CLASIFICACION})
    resultados["F1 macro validación"] = busqueda.cv_results_["mean_test_score"]
    resultados["hojas"] = [DecisionTreeClassifier(random_state=SEMILLA, **params)
                           .fit(p["X_train"], p["y_train"]).get_n_leaves()
                           for params in busqueda.cv_results_["params"]]
    print(f"Partición {nombre}: F1 macro de validación y hojas según max_depth "
          f"(mínimo y máximo entre las 9 combinaciones de min_samples_leaf y min_samples_split)")
    display(resultados.assign(max_depth=resultados["max_depth"].fillna("sin límite").astype(str))
            .groupby("max_depth", sort=False)[["F1 macro validación", "hojas"]].agg(["min", "max"]).round(3))
    print(f"Elegida: {busqueda.best_params_}\n")


# %% [markdown]
# **Qué muestra**
#
# - **Con `max_depth = 2`, el árbol tiene solo 3 hojas para 5 clases**, y el F1 macro de validación queda en 0,52.
# - **Desde `max_depth = 4`, las 27 combinaciones dan el mismo árbol de 6 hojas**, con un F1 macro de validación de 0,966 (80/20) y 0,965 (70/30). `min_samples_leaf` y `min_samples_split` no cambian nada porque las clases se separan del todo con hojas de 13 pacientes o más.
# - Como todas empatan, nos quedamos con la más simple, `max_depth = 4`, que es la que devuelve `GridSearchCV` ante un empate. La configuración elegida es la misma en las dos particiones.

# %% [markdown]
# ### Post-poda (ccp_alpha)
#
# Para el mejor árbol de cada partición calculamos los valores de `ccp_alpha` en los que cambia el árbol podado, y medimos cada uno con la misma validación cruzada. Elegimos el de mayor F1 macro y, si hay empate, el mayor `ccp_alpha` (el árbol más chico).

# %%
def camino_de_poda(p, params):
    """Hojas y F1 macro de validación del árbol podado con cada ccp_alpha de su camino de poda."""
    arbol = DecisionTreeClassifier(random_state=SEMILLA, **params).fit(p["X_train"], p["y_train"])
    alphas = arbol.cost_complexity_pruning_path(p["X_train"], p["y_train"]).ccp_alphas
    filas = []
    for alpha in alphas:
        modelo = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha, **params)
        f1 = cross_val_score(modelo, p["X_train"], p["y_train"], cv=5, scoring="f1_macro").mean()
        filas.append({"ccp_alpha": alpha, "hojas": modelo.fit(p["X_train"], p["y_train"]).get_n_leaves(),
                      "F1 macro validación": f1})
    return pd.DataFrame(filas)


podas, alpha_elegido, arboles_b = {}, {}, {}
for nombre, p in particiones_b.items():
    podas[nombre] = camino_de_poda(p, parametros_arbol[nombre])
    poda = podas[nombre]
    mejores = poda[poda["F1 macro validación"] >= poda["F1 macro validación"].max() - 1e-9]
    alpha_elegido[nombre] = mejores["ccp_alpha"].max()
    arboles_b[nombre] = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha_elegido[nombre],
                                               **parametros_arbol[nombre]).fit(p["X_train"], p["y_train"])
    print(f"Partición {nombre}: ccp_alpha elegido = {alpha_elegido[nombre]:.4f}")
    display(poda.round(4))

# %% [markdown]
# **Qué muestra**
#
# - **Elegimos `ccp_alpha = 0` en las dos particiones, es decir, no podar.** Sin poda, el árbol conserva sus 6 hojas y un F1 macro de validación de 0,966 (80/20) y 0,965 (70/30). La primera poda (`ccp_alpha` ≈ 0,06) lo deja en 4 hojas, y el F1 baja a 0,83 (80/20) y 0,92 (70/30). Las siguientes lo bajan a menos de 0,7, y con una sola hoja, a menos de 0,3.
# - La post-poda sirve cuando el árbol tiene hojas que ajustan ruido del entrenamiento. Este no tiene ninguna: cada hoja separa una clase, así que cualquier poda pierde una clase entera. `max_depth = 4` ya lo había limitado, y en realidad ni siquiera hacía falta: sin límite de profundidad, el árbol también tiene 6 hojas.

# %% [markdown]
# **Hipótesis.** Por lo que vimos en la preparación, esperamos que el árbol corte primero por `Na_a_K` (para separar `DrogaY`), después por `BP`, y que use `Edad` para separar `DrogaA` de `DrogaB` y `Colesterol` para `DrogaC`. La poda debería quitar primero la hoja que menos pacientes del entrenamiento separa.

# %%
p = particiones_b["80/20"]
params = parametros_arbol["80/20"]
alpha_primera_poda = podas["80/20"]["ccp_alpha"].iloc[1]
arbol_podado = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha_primera_poda,
                                      **params).fit(p["X_train"], p["y_train"])

fig, axes = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [3, 2]})
for ax, modelo, titulo in [
        (axes[0], arboles_b["80/20"], f"Antes de la poda: ccp_alpha = {alpha_elegido['80/20']:.3f} (elegido)"),
        (axes[1], arbol_podado, f"Después de la primera poda: ccp_alpha = {alpha_primera_poda:.3f}")]:
    plot_tree(modelo, feature_names=list(p["X_train"].columns), class_names=list(modelo.classes_),
              filled=True, rounded=True, impurity=False, fontsize=10, precision=2, ax=ax)
    ax.set_title(f"{titulo} · {modelo.get_n_leaves()} hojas")
fig.suptitle("Árbol de clasificación de la partición 80/20")
guardar_figura("arbol_clasificacion_poda")
plt.show()

print("Umbrales de Edad y Na_a_K en su unidad original (los de BP y Colesterol son los códigos de la codificación):")
umbrales_originales(arboles_b["80/20"], p, NUMERICAS_DROGAS, niveles=4)

# %% [markdown]
# **Qué muestra**
#
# - **Se cumple la hipótesis sobre los cortes.** El primero es por `Na_a_K` en −0,19 desvíos, que es una relación sodio/potasio de **14,83**: los 73 pacientes de arriba de ese valor son todos de `DrogaY`. Después corta por `BP`:
#     - con presión alta (`BP` > 1,5), `Edad` ≤ 50,5 separa a `DrogaA` de `DrogaB`;
#     - con presión baja o normal, la presión baja y el colesterol alto separan a `DrogaC` de `DrogaX`.
# - **El árbol son 5 reglas:** `DrogaY` si `Na_a_K` > 14,83; si no, `DrogaA` con presión alta y hasta 50 años, `DrogaB` con presión alta y más de 50, `DrogaC` con presión baja y colesterol alto, y `DrogaX` en el resto. No usa `Sexo`, como anticipaba el análisis exploratorio.
# - **También se cumple la hipótesis sobre la poda.** La primera poda junta en una sola hoja, que predice `DrogaX`, las tres hojas de la rama de presión baja o normal. Los 13 pacientes de `DrogaC` del entrenamiento quedarían mal clasificados, y por eso el F1 macro cae tanto.

# %%
graficar_matrices(arboles_b, "Árbol de decisión: matrices de confusión")
plt.show()
metricas_arbol = tabla_metricas(arboles_b)
metricas_arbol.round(3)


# %% [markdown]
# **Qué muestra**
#
# - **El árbol clasifica bien a todos los pacientes de los dos test:** la exactitud, la precisión macro y la exhaustividad macro valen 1. Las reglas que aprendió con el entrenamiento se cumplen también en el test.
# - Los test son chicos (40 y 60 pacientes): acertar todos no garantiza acertar siempre con pacientes nuevos. El F1 macro de validación cruzada (0,966 y 0,965) es una estimación más prudente.

# %% [markdown]
# ## 5. Bayes ingenuo
#
# Discretizamos `Edad` y `Na_a_K` en intervalos de igual frecuencia (cuantiles), con 3 y con 5 intervalos, y aplicamos `CategoricalNB` a los cinco atributos, ya todos categóricos. Usamos `CategoricalNB` y no `MultinomialNB` porque los atributos son categorías, no conteos.
#
# El discretizador va dentro de un pipeline: así los cortes se calculan solo con el entrenamiento (y, en la validación cruzada, solo con las particiones de entrenamiento de cada vuelta). La cantidad de intervalos se elige con el mismo F1 macro de validación.

# %%
def crear_bayes(n_intervalos):
    """Pipeline que discretiza Edad y Na_a_K por cuantiles y aplica CategoricalNB a todos los atributos."""
    discretizador = ColumnTransformer(
        [("intervalos", KBinsDiscretizer(n_bins=n_intervalos, encode="ordinal", strategy="quantile"), NUMERICAS_DROGAS)],
        remainder="passthrough")   # Sexo, BP y Colesterol ya son categorías codificadas
    return make_pipeline(discretizador, CategoricalNB())


modelos_bayes, filas = {}, []
for nombre, p in particiones_b.items():
    for n_intervalos in [3, 5]:
        modelo = crear_bayes(n_intervalos)
        f1 = cross_val_score(modelo, p["X_train"], p["y_train"], cv=5, scoring="f1_macro").mean()
        modelo.fit(p["X_train"], p["y_train"])
        modelos_bayes[(nombre, n_intervalos)] = modelo
        filas.append({"Partición": nombre, "Intervalos": n_intervalos, "F1 macro validación": f1,
                      **metricas_clasificacion(p["y_test"], modelo.predict(p["X_test"]))})

print("F1 macro de validación, y métricas sobre el test de cada partición:")
resultados_bayes = pd.DataFrame(filas).set_index(["Partición", "Intervalos"])
resultados_bayes.round(3)

# %%
# Cortes de los intervalos en la partición 80/20, pasados de desvíos estándar a la unidad original
p = particiones_b["80/20"]
for n_intervalos in [3, 5]:
    intervalos = modelos_bayes[("80/20", n_intervalos)][0].named_transformers_["intervalos"]
    for j, columna in enumerate(NUMERICAS_DROGAS):
        cortes = intervalos.bin_edges_[j] * p["escalador"].scale_[j] + p["escalador"].mean_[j]
        print(f"{n_intervalos} intervalos, {columna}: {np.round(cortes, 1).tolist()}")

# %% [markdown]
# **Qué muestra**
#
# - **5 intervalos funcionan mejor que 3**, tanto en validación (F1 macro de 0,81 contra 0,69 con 80/20, y de 0,76 contra 0,64 con 70/30) como en test (exactitud de 0,925 contra 0,825, y de 0,917 contra 0,850).
# - **La diferencia está en dónde caen los cortes:**
#     - Con 3 intervalos, el del medio de `Na_a_K` va de 11,4 a 18,3, y mezcla a los pacientes de `DrogaY` (15,0 o más) con los de las otras drogas (14,6 o menos). Con 5 intervalos, el intervalo mezclado se achica a 12,8–16,1.
#     - En `Edad`, con 3 intervalos el corte en 54 junta a pacientes de `DrogaA` (hasta 50 años) y de `DrogaB` (desde 51) en el intervalo de 37 a 54. Con 5, uno de los cortes cae justo en 50.
# - La discretización por igual frecuencia no conoce las clases, así que no pone los cortes en las fronteras de 14,83 y 50,5 que encuentra el árbol. Esa información se pierde antes de entrenar.

# %%
mejor_n_intervalos = {nombre: int(resultados_bayes.loc[nombre, "F1 macro validación"].idxmax())
                      for nombre in PARTICIONES}
mejores_bayes = {nombre: modelos_bayes[(nombre, n)] for nombre, n in mejor_n_intervalos.items()}
print("Cantidad de intervalos elegida:", mejor_n_intervalos)

graficar_matrices(mejores_bayes, "Bayes ingenuo: matrices de confusión")
plt.show()
metricas_bayes = tabla_metricas(mejores_bayes)
display(metricas_bayes.round(3))
print("Pacientes mal clasificados:")
mal_clasificados(mejores_bayes)

# %% [markdown]
# **Qué muestra**
#
# - **Todos los errores son pacientes de `DrogaY` clasificados con otra droga:** 3 de 18 en el test 80/20 y 5 de 27 en el 70/30.
# - **Todos tienen una relación `Na_a_K` de entre 15,2 y 15,8**, justo arriba de la frontera de 14,83. Caen en el intervalo de 12,8 a 16,1, donde la mayoría de los pacientes son de otras drogas. Para ellos, el modelo termina decidiendo por la presión, el colesterol y la edad, y elige la droga que corresponde a esos valores: `DrogaA` con presión alta y menos de 50 años, `DrogaX` con presión normal, o `DrogaC` con presión baja y colesterol alto.
# - Por eso la exhaustividad macro (0,967 y 0,963) es mayor que la precisión macro (0,926 y 0,900). Las clases chicas reciben pacientes de `DrogaY`, lo que baja su precisión, mientras que la exhaustividad de `DrogaY` baja a 0,83 y 0,81.

# %% [markdown]
# ## 6. k-NN
#
# Probamos k = 1, 3, 5, 7 y 9 con las distancias euclidiana y Manhattan, y elegimos con el mismo F1 macro de validación. `Edad` y `Na_a_K` ya están estandarizadas, y las categóricas codificadas van de 0 a 1 o de 0 a 2, así que todos los atributos tienen escalas comparables.

# %%
GRILLA_KNN = {"n_neighbors": [1, 3, 5, 7, 9], "metric": ["euclidean", "manhattan"]}

busquedas_knn = {}
for nombre, p in particiones_b.items():
    busquedas_knn[nombre] = GridSearchCV(KNeighborsClassifier(), GRILLA_KNN, cv=5, scoring="f1_macro")
    busquedas_knn[nombre].fit(p["X_train"], p["y_train"])
    print(f"Partición {nombre}: elegida {busquedas_knn[nombre].best_params_}, "
          f"F1 macro de validación {busquedas_knn[nombre].best_score_:.3f}")

# %% [markdown]
# **Hipótesis.** Con k chico, el vecino más cercano puede ser un paciente de otra clase por un detalle irrelevante (por ejemplo, el sexo). Con k grande, las clases chicas (`DrogaB` y `DrogaC`, 13 pacientes en el entrenamiento) quedan en minoría entre los vecinos. Esperamos un máximo en un k intermedio.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (nombre, busqueda) in zip(axes, busquedas_knn.items()):
    resultados = pd.DataFrame({"k": busqueda.cv_results_["param_n_neighbors"],
                               "distancia": busqueda.cv_results_["param_metric"],
                               "F1 macro validación": busqueda.cv_results_["mean_test_score"]})
    sns.lineplot(data=resultados, x="k", y="F1 macro validación", hue="distancia", marker="o", ax=ax)
    ax.set(title=f"Partición {nombre}", xticks=GRILLA_KNN["n_neighbors"])
fig.suptitle("k-NN: F1 macro de validación cruzada según k y la distancia")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Qué muestra**
#
# - **No se cumple la hipótesis de un máximo en un k intermedio.** Las curvas van en zigzag: con 80/20 el mejor es k = 1, y con 70/30, k = 9, con diferencias de hasta 0,1 entre valores vecinos de k. Con 13 pacientes o menos de `DrogaB` y de `DrogaC` en el entrenamiento, cada pliegue de validación tiene 2 o 3 de cada una. Un solo error mueve mucho el F1 macro, así que la elección de k es inestable.
# - **La distancia Manhattan le gana a la euclidiana en 8 de las 10 comparaciones**, y es la elegida en las dos particiones.

# %%
mejores_knn = {nombre: busqueda.best_estimator_ for nombre, busqueda in busquedas_knn.items()}
graficar_matrices(mejores_knn, "k-NN: matrices de confusión")
plt.show()
metricas_knn = tabla_metricas(mejores_knn)
display(metricas_knn.round(3))
print("Pacientes mal clasificados:")
mal_clasificados(mejores_knn)


# %% [markdown]
# **Qué muestra**
#
# - **k-NN se equivoca con 5 pacientes en cada test**, una exactitud de 0,875 (80/20) y 0,917 (70/30).
# - **Igual que Bayes, falla cerca de la frontera de `Na_a_K`:**
#     - pacientes de `DrogaY` con valores de 15,2 a 16,6, clasificados como `DrogaX` o `DrogaA`;
#     - en sentido contrario, uno de `DrogaX` con 12,9 y uno de `DrogaC` con 13,1, clasificados como `DrogaY`.
# - La distancia mezcla `Na_a_K` con los demás atributos. Por eso, un paciente de `DrogaY` con 15,2 puede quedar más cerca de pacientes de `DrogaX` con 14 y la misma presión y colesterol que de los de `DrogaY`, que en general tienen valores mucho más altos. `Sexo` también entra en la distancia, aunque no ayuda a distinguir las drogas.
# - Otro paciente de `DrogaY`, con 19,8, se clasifica como `DrogaC` porque tiene presión baja y colesterol alto, el perfil de `DrogaC`.

# %% [markdown]
# ## 7. Comparación de modelos
#
# El mejor modelo de cada algoritmo en cada partición, con las métricas sobre su test.

# %%
def describir_knn(nombre):
    """Configuración elegida de k-NN, para la tabla."""
    params = busquedas_knn[nombre].best_params_
    return f"k-NN ({params['metric']}, k = {params['n_neighbors']})"


filas = []
for nombre in PARTICIONES:
    filas.append({"Partición": nombre, "Modelo": f"Árbol (max_depth = {parametros_arbol[nombre]['max_depth']}, "
                                                f"{arboles_b[nombre].get_n_leaves()} hojas)",
                  **metricas_arbol.loc[nombre]})
    filas.append({"Partición": nombre, "Modelo": f"Bayes ingenuo ({mejor_n_intervalos[nombre]} intervalos)",
                  **metricas_bayes.loc[nombre]})
    filas.append({"Partición": nombre, "Modelo": describir_knn(nombre), **metricas_knn.loc[nombre]})

comparacion = pd.DataFrame(filas).set_index(["Partición", "Modelo"])
comparacion.round(3)

# %% [markdown]
# **Qué muestra**
#
# - **El árbol de decisión es el mejor en las dos particiones:** vale 1 en las tres métricas. Le siguen Bayes ingenuo (exactitud de 0,925 y 0,917) y k-NN (0,875 y 0,917). Con 70/30 empatan en exactitud, y k-NN tiene más precisión macro y Bayes, más exhaustividad macro.
# - **El árbol también es el más interpretable:** son 5 reglas con umbrales que un médico puede leer y comprobar, como "relación sodio/potasio mayor a 14,83" o "presión alta y hasta 50 años". Bayes da probabilidades por intervalo, más difíciles de leer, y k-NN no da ninguna regla: solo puede justificar cada predicción mostrando los pacientes parecidos.
# - **Por qué gana el árbol:** en estos datos, cada droga se define por umbrales sobre un atributo a la vez, que es justamente la forma de las fronteras que construye un árbol. Bayes pierde información al discretizar con cortes que no coinciden con esas fronteras. k-NN mezcla todos los atributos en una distancia, incluido `Sexo`, que no aporta nada.
#
# **Elegimos el árbol de decisión** (`max_depth = 4`, sin post-poda): tiene las mejores métricas y es el que mejor explica sus decisiones.

# %% [markdown]
# ---
# # Conclusiones
#
# **Parte A: regresión**
#
# - **`Profit` depende casi solo del gasto en I+D:** su correlación con `RyD_Spend` es de 0,999, y el árbol le da a esa variable una importancia de 0,99.
# - **La limpieza importó.** Eliminamos una fila duplicada y las 3 empresas con ganancias de hasta 476.000 dólares, que no seguían la relación con los gastos. Conservamos los atípicos de `Administration`, que son valores posibles.
# - **El mejor árbol** (`max_depth = 10`, `min_samples_leaf = 5`, `min_samples_split = 2`, la misma configuración en las dos particiones) tiene en test un RMSE de 3.492 (80/20) y 2.766 dólares (70/30), y un MAE de 1.100 y 925. Elegir, entre las configuraciones empatadas, la de menos hojas evitó un árbol de 601 hojas que memorizaba el entrenamiento.
# - **Casi todo el error viene de las primeras 50 filas del archivo**, más heterogéneas: aportan entre el 96 % y el 98 % del error cuadrático. En las demás empresas, el error es menor a 1.000 dólares.
#
# **Parte B: clasificación**
#
# - **Las drogas siguen reglas simples:** `DrogaY` si la relación sodio/potasio supera 14,83, y las demás según la presión, la edad (hasta o desde 50 años) y el colesterol. `Sexo` no influye. Las clases están desbalanceadas (de 45,5 % a 8 %), así que estratificamos las particiones y medimos con promedios macro.
# - **El árbol (6 hojas, sin poda) clasifica bien todo el test** en las dos particiones. La post-poda no mejora: cada hoja separa una clase, y la primera poda ya pierde `DrogaC`.
# - **Bayes ingenuo** (5 intervalos) y **k-NN** (Manhattan) llegan a una exactitud de entre 0,875 y 0,925. **Fallan con los mismos pacientes:** los de `DrogaY` con una relación sodio/potasio apenas mayor a 14,83.
# - **El modelo elegido es el árbol de decisión,** por sus métricas y porque sus reglas se pueden leer y verificar.
#
# **Las dos particiones** dieron resultados parecidos, y las diferencias dependen de qué casos difíciles caen en cada test, no de la cantidad de datos de entrenamiento:
#
# - en la Parte A, las empresas de las primeras 50 filas;
# - en la Parte B, los pacientes cerca de la frontera de `Na_a_K`.
#
# Con test tan chicos (40 y 60 pacientes en la Parte B), un solo error mueve varios puntos las métricas macro.
