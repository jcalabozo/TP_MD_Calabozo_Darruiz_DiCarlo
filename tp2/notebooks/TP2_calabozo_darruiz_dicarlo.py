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
# - Parte B – Clasificación: 3. Preparación de datos, 4. Árboles de decisión para clasificación, 5. Bayes ingenuo, 6. k-NN y 7. Comparación de modelos.
# - 8. Conclusiones.
#
# Las dos particiones que pide la consigna (80/20 y 70/30) se guardan con los sufijos `_80` y `_70`, y cada paso se resuelve con una función que se aplica a las dos. Cuando reutilizamos o adaptamos código de los notebooks de clase, el comentario de la celda dice de cuál sale.

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
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, root_mean_squared_error)

# random_state que pide la consigna para las particiones y los modelos
SEMILLA = 63

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
# Hay 1000 empresas con tres gastos (`RyD_Spend`, `Administration` y `MarketingSpend`), el estado (`State`, la única categórica) y la ganancia, `Profit`, que es la variable a estimar. Ninguna columna tiene valores faltantes.

# %%
# Filas repetidas: keep=False muestra la fila original y su copia
empresas[empresas.duplicated(keep=False)]

# %% [markdown]
# La fila 755 repite la 335: la sacamos en la limpieza, porque pesaría el doble en el entrenamiento.

# %%
GASTOS = ["RyD_Spend", "Administration", "MarketingSpend"]
NUMERICAS_EMPRESAS = GASTOS + ["Profit"]

# Resumen de las variables numéricas
empresas[NUMERICAS_EMPRESAS].describe().round(0)

# %%
# Gastos en 0 (podrían ser faltantes cargados como cero) y cantidad de empresas por estado
print("Empresas con gasto 0:", (empresas[GASTOS] == 0).sum().to_dict())
print("Empresas por estado:", empresas["State"].value_counts().to_dict())

# %% [markdown]
# Unas pocas empresas no gastan nada en I+D o en marketing. Es algo posible y las conservamos. Los tres estados tienen cantidades parecidas. Lo que llama la atención son los máximos de `Profit` y de `Administration`, muy lejos de su tercer cuartil: los revisamos en 1.3.

# %% [markdown]
# ### 1.2 Distribuciones
#
# Si las empresas fueran de tamaños muy distintos, esperaríamos distribuciones asimétricas: muchas empresas que gastan y ganan poco, y pocas que gastan y ganan mucho.

# %%
# Histograma de cada variable numérica, como en el notebook de árboles de regresión de la U4 (sección "Distribuciones")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, columna in zip(axes, NUMERICAS_EMPRESAS):
    sns.histplot(data=empresas, x=columna, kde=True, element="step", ax=ax)
fig.suptitle("Distribución de las variables numéricas")
plt.tight_layout()
plt.show()


# %% [markdown]
# No se cumple. `RyD_Spend` y `MarketingSpend` se reparten en todo su rango, sin acumularse en los valores bajos, y `Administration` se concentra en una franja angosta. La única cola larga es la de `Profit`: casi todas las empresas ganan menos de 200.000 dólares, pero unos pocos valores aislados estiran el eje hasta 476.000.

# %% [markdown]
# ### 1.3 Valores atípicos
#
# Si esos valores aislados son errores de carga, deberían aparecer como atípicos y, además, no seguir la relación con los gastos que tienen las demás empresas.

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
# Se confirma. La regla del rango intercuartílico marca 3 atípicos en `Profit`, y son las tres últimas filas del archivo: ganan entre 334.000 y 476.000 dólares, cuando ninguna otra empresa llega a 200.000. En los gráficos de dispersión, las demás empresas forman una línea con `RyD_Spend` y con `MarketingSpend`, y esas tres quedan muy por encima. Las tratamos como errores de carga y las sacamos antes de dividir, porque la consigna pide limpiar el conjunto antes de generar las particiones.
#
# `Administration` tiene además 22 atípicos, que quedan dispersos alrededor de la franja del resto. Los miramos después de limpiar.

# %% [markdown]
# ### 1.4 Limpieza

# %%
# Sacamos la fila duplicada y las tres empresas atípicas en Profit
empresas_limpio = empresas.drop_duplicates()
empresas_limpio = empresas_limpio.drop(atipicas_profit.index)
print(f"Quedan {len(empresas_limpio)} de {len(empresas)} empresas")

# %%
# Las empresas atípicas en Administration que quedan, comparadas con el resto: mínimo y máximo de cada variable
atipicas_admin = empresas_limpio.drop(remove_outliers_iqr(empresas_limpio, "Administration").index)
grupo = np.where(empresas_limpio.index.isin(atipicas_admin.index), "atípicas en Administration", "resto")
print(f"Empresas atípicas en Administration: {len(atipicas_admin)}")
empresas_limpio.groupby(grupo)[NUMERICAS_EMPRESAS].agg(["min", "max"]).T.round(0)

# %% [markdown]
# Las 19 atípicas en `Administration` que quedan gastan en administración más o menos que el resto, pero sus otros gastos y su ganancia están dentro de los rangos de las demás. Como son montos posibles, **conservamos esas 19 y sacamos solo la fila duplicada y las 3 empresas con errores de carga**: quedan 996.

# %% [markdown]
# ### 1.5 Correlaciones
#
# Esperamos que la ganancia crezca con los tres gastos y que los gastos estén correlacionados entre sí, porque una empresa más grande gasta más en todo.

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
pd.DataFrame({"con las atípicas": con_atipicas, "sin las atípicas": correlaciones["Profit"]}).drop("Profit").round(3)

# %% [markdown]
# Se cumple, y más de lo esperado: `Profit` está casi perfectamente correlacionada con `RyD_Spend` (0,999), muy correlacionada con `MarketingSpend` y bastante menos con `Administration`. Entre los gastos, `RyD_Spend` y `MarketingSpend` cuentan casi lo mismo. La segunda tabla muestra que las tres empresas que sacamos alcanzaban para bajar esas correlaciones, porque se alejan mucho de la relación del resto.

# %% [markdown]
# `State` no entra en la matriz porque es nominal. Si la ganancia depende de los gastos, el estado no debería cambiarla: esperamos ganancias parecidas en los tres estados, o diferencias que acompañen a las del gasto.

# %%
# Por estado: cantidad de empresas, mediana de la ganancia y del gasto en I+D, y ganancia por dólar de I+D
por_estado = empresas_limpio.groupby("State")[["Profit", "RyD_Spend"]].median()
por_estado.insert(0, "Empresas", empresas_limpio["State"].value_counts())
por_estado["Profit por dólar de I+D"] = por_estado["Profit"] / por_estado["RyD_Spend"]
por_estado.round(2)

# %% [markdown]
# Se cumple. La ganancia mediana va de unos 114.000 a 121.000 dólares según el estado, y Florida, el que más gana, es también el que más gasta en I+D. Por cada dólar de I+D, las empresas de los tres estados ganan casi lo mismo (entre 1,44 y 1,48 dólares): el estado no parece agregar nada a lo que explican los gastos. Lo vamos a comprobar con el árbol en 2.3.

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
# ### 1.7 Particiones 80/20 y 70/30 y estandarización
#
# Dividimos con `random_state = 63` y después estandarizamos los gastos con la media y el desvío del entrenamiento de cada partición, para que el test no influya en nada de lo que se aprende. `Profit` no se estandariza, para que los errores queden en dólares. La misma función sirve para la Parte B.

# %%
# Divide y estandariza como en los notebooks de Bayes y k-NN de la U5: el escalador aprende la media y el desvío
# solo del entrenamiento y se aplica a los dos conjuntos
def dividir_y_estandarizar(X, y, proporcion_test, columnas, estratificar_por=None):
    """Divide con random_state=63 y estandariza las columnas con la media y el desvío del entrenamiento."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=proporcion_test, random_state=SEMILLA,
                                                        stratify=estratificar_por)
    escalador = StandardScaler().set_output(transform="pandas")
    escalador.fit(X_train[columnas])
    X_train[columnas] = escalador.transform(X_train[columnas])
    X_test[columnas] = escalador.transform(X_test[columnas])
    return X_train, X_test, y_train, y_test, escalador


X_empresas = empresas_limpio.drop(columns="Profit")
y_empresas = empresas_limpio["Profit"]
X_train_80, X_test_80, y_train_80, y_test_80, escalador_80 = dividir_y_estandarizar(
    X_empresas, y_empresas, 0.2, GASTOS)
X_train_70, X_test_70, y_train_70, y_test_70, escalador_70 = dividir_y_estandarizar(
    X_empresas, y_empresas, 0.3, GASTOS)
print(f"80/20: {len(X_train_80)} empresas de entrenamiento y {len(X_test_80)} de test")
print(f"70/30: {len(X_train_70)} empresas de entrenamiento y {len(X_test_70)} de test")
print(f"Empresas del test 80/20 que también están en el test 70/30: {y_test_80.index.isin(y_test_70.index).sum()}")

# %%
# Entrenamiento de la partición 80/20 ya preparado
X_train_80.describe().round(2)

# %% [markdown]
# Los tres gastos quedan con media 0 y desvío 1, y `Administration` llega a unos 8 desvíos de la media por las empresas atípicas que conservamos. Algo para tener en cuenta al comparar particiones: con la misma semilla, las 200 empresas del test 80/20 también están en el test 70/30: los dos test comparten los casos difíciles.

# %% [markdown]
# ## 2. Árboles de decisión para regresión
#
# Por la preparación, esperamos que el árbol prediga casi solo con `RyD_Spend` y que sus errores más grandes estén en las empresas atípicas en `Administration`, que conservamos.
#
# ### 2.1 Búsqueda de hiperparámetros
#
# Probamos 864 combinaciones de `max_depth`, `min_samples_leaf` y `min_samples_split` (la grilla está en el código). Cada una se evalúa con validación cruzada de 5 pliegues sobre el entrenamiento, y elegimos la de menor RMSE promedio; el test se usa una sola vez, al final.
#
# Como un árbol más profundo aproxima mejor una relación continua, esperamos que el error baje con `max_depth` hasta estabilizarse. Las hojas muy chicas deberían ajustar el ruido del entrenamiento, y las muy grandes, promediar empresas demasiado distintas.

# %%
GRILLA_REGRESION = {"max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None],
                    "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 15, 20],
                    "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30]}


# GridSearchCV con cv=5, como en el notebook de árboles de clasificación de la U4. La usamos también en la Parte B
def buscar_hiperparametros(modelo, grilla, metrica, X_train, y_train):
    """Prueba la grilla con validación cruzada de 5 pliegues y se queda con la combinación de mejor promedio."""
    busqueda = GridSearchCV(modelo, grilla, cv=5, scoring=metrica)
    return busqueda.fit(X_train, y_train)


# "neg_root_mean_squared_error" es el RMSE con signo negativo, porque GridSearchCV busca el valor más alto
arbol_regresion = DecisionTreeRegressor(random_state=SEMILLA)
METRICA_REGRESION = "neg_root_mean_squared_error"
busqueda_80 = buscar_hiperparametros(arbol_regresion, GRILLA_REGRESION, METRICA_REGRESION, X_train_80, y_train_80)
busqueda_70 = buscar_hiperparametros(arbol_regresion, GRILLA_REGRESION, METRICA_REGRESION, X_train_70, y_train_70)

# Configuración elegida en cada partición, con su RMSE de validación en dólares
elegidos = pd.DataFrame([busqueda_80.best_params_, busqueda_70.best_params_], index=["80/20", "70/30"])
elegidos["RMSE validación"] = [-busqueda_80.best_score_, -busqueda_70.best_score_]
elegidos.round(0)


# %%
def resultados_validacion(busqueda_80, busqueda_70):
    """Resultados de la validación cruzada de las dos particiones en una sola tabla, con la columna Partición."""
    resultados_80 = pd.DataFrame(busqueda_80.cv_results_).assign(Partición="80/20")
    resultados_70 = pd.DataFrame(busqueda_70.cv_results_).assign(Partición="70/30")
    return pd.concat([resultados_80, resultados_70])


resultados = resultados_validacion(busqueda_80, busqueda_70)
resultados["RMSE validación"] = -resultados["mean_test_score"]
resultados["param_max_depth"] = resultados["param_max_depth"].fillna("sin límite")

# Efecto de cada hiperparámetro: para cada valor, el menor RMSE entre todas las combinaciones de los otros dos
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, parametro in zip(axes, GRILLA_REGRESION):
    menor_rmse = resultados.groupby(["Partición", f"param_{parametro}"], sort=False)["RMSE validación"].min()
    sns.pointplot(data=menor_rmse.reset_index(), x=f"param_{parametro}", y="RMSE validación", hue="Partición", ax=ax)
    ax.set(xlabel=parametro, ylabel="Menor RMSE de validación (USD)")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Menor RMSE de validación para cada valor de cada hiperparámetro")
plt.tight_layout()
plt.show()

# %% [markdown]
# Se cumple en las dos particiones. Con 2 a 4 niveles el error es alto, y desde 6 o 7 se estabiliza. `min_samples_leaf` tiene su mejor valor en el medio: con hojas más chicas el árbol ajusta el ruido del entrenamiento, y con hojas más grandes promedia empresas de ganancias distintas. `min_samples_split` no cambia el error hasta 8 o 10, y con valores mayores el árbol deja de dividir nodos chicos y el error sube.

# %%
# Las 5 mejores combinaciones de cada partición, ordenadas por rank_test_score como en el notebook de árboles de
# clasificación de la U4
columnas = ["Partición", "rank_test_score", "param_max_depth", "param_min_samples_leaf", "param_min_samples_split",
            "RMSE validación"]
mejores = resultados.sort_values(["Partición", "rank_test_score"], ascending=[False, True])
mejores.groupby("Partición", sort=False).head(5)[columnas].round(0)


# %% [markdown]
# Elegimos **`max_depth = 8`, `min_samples_leaf = 4` y `min_samples_split = 2` con 80/20, y `max_depth = 7`, `min_samples_leaf = 5` y `min_samples_split = 2` con 70/30**. Las primeras filas de cada partición empatan porque dan el mismo árbol: mientras `min_samples_split` no supere el doble de `min_samples_leaf`, no cambia nada, ya que un nodo más chico no se puede dividir en dos hojas permitidas. Ante un empate, `GridSearchCV` se queda con la primera combinación.

# %% [markdown]
# ### 2.2 Resultados en test

# %%
# Las tres métricas que pide la consigna, como en Unidad4.py
def metricas_regresion(y_real, y_pred):
    """MAE, MSE y RMSE de unas predicciones."""
    return {"MAE": mean_absolute_error(y_real, y_pred),
            "MSE": mean_squared_error(y_real, y_pred),
            "RMSE": root_mean_squared_error(y_real, y_pred)}


# El árbol elegido en cada partición (GridSearchCV ya lo reentrenó con todo el entrenamiento) y sus predicciones
arbol_80 = busqueda_80.best_estimator_
arbol_70 = busqueda_70.best_estimator_
y_pred_80 = arbol_80.predict(X_test_80)
y_pred_70 = arbol_70.predict(X_test_70)

pd.DataFrame({"80/20": metricas_regresion(y_test_80, y_pred_80),
              "70/30": metricas_regresion(y_test_70, y_pred_70)}).T.rename_axis("Test").round(0).astype(int)

# %%
# Ajuste del árbol elegido: RMSE en entrenamiento, validación y test, como la comparación de RMSE del notebook de
# árboles de regresión de la U4
pd.DataFrame({"Hojas": [arbol_80.get_n_leaves(), arbol_70.get_n_leaves()],
              "RMSE entrenamiento": [root_mean_squared_error(y_train_80, arbol_80.predict(X_train_80)),
                                     root_mean_squared_error(y_train_70, arbol_70.predict(X_train_70))],
              "RMSE validación": [-busqueda_80.best_score_, -busqueda_70.best_score_],
              "RMSE test": [root_mean_squared_error(y_test_80, y_pred_80),
                            root_mean_squared_error(y_test_70, y_pred_70)]},
             index=["80/20", "70/30"]).rename_axis("Partición").round(0).astype(int)

# %% [markdown]
# El árbol se equivoca en promedio en unos mil dólares (MAE), menos de la décima parte del desvío de `Profit` (42.889 dólares, en la tabla de 1.1). Rinde algo mejor en entrenamiento que en validación, un sobreajuste esperable en árboles con tantas hojas (156 y 99). Que el RMSE de test sea unas tres veces el MAE avisa que unas pocas empresas tienen errores mucho más grandes que el resto; en 2.4 vemos cuáles.

# %% [markdown]
# ### 2.3 El árbol con la mejor configuración
#
# Por la correlación de 0,999 entre `Profit` y `RyD_Spend`, esperamos que el árbol corte primero por esa variable y que casi no use `State`.

# %%
# Los 4 primeros niveles del árbol de la partición 80/20 (max_depth=3 cuenta desde la raíz, que es el nivel 0),
# con plot_tree como en el notebook de árboles de regresión de la U4 (el árbol completo no se puede leer)
plt.figure(figsize=(22, 8))
plot_tree(arbol_80, feature_names=list(X_train_80.columns), max_depth=3, filled=True, rounded=True,
          impurity=False, fontsize=9, precision=2)
plt.title(f"Árbol de regresión de la partición 80/20 ({arbol_80.get_n_leaves()} hojas; "
          "se muestran los 4 primeros niveles)")
plt.show()

# %%
# Importancia de cada variable, con el gráfico de barras del notebook de árboles de regresión de la U4
importancias = pd.DataFrame({"variable": X_train_80.columns, "importancia": arbol_80.feature_importances_})
importancias = importancias.sort_values(by="importancia", ascending=False)

g = sns.barplot(data=importancias, x="importancia", y="variable")
sns.despine(bottom=True, left=True)
g.set_title("Importancia de las variables (árbol de la partición 80/20)")
g.set(xlabel=None, ylabel=None, xticks=[])
for barras in g.containers:
    g.bar_label(barras, padding=2, fmt="%.3f")
plt.show()

# %%
# El primer corte del árbol (el de la raíz, nodo 0) en desvíos estándar y en dólares: umbral × desvío + media
# del entrenamiento. tree_.feature dice qué columna usa el nodo, y tree_.threshold, su umbral
columna = X_train_80.columns[arbol_80.tree_.feature[0]]
umbral = arbol_80.tree_.threshold[0]
posicion = GASTOS.index(columna)   # posición de la columna en el escalador
en_dolares = umbral * escalador_80.scale_[posicion] + escalador_80.mean_[posicion]
print(f"{columna} <= {umbral:.2f} desvíos equivale a {en_dolares:,.0f} dólares")


# %% [markdown]
# Se confirma. El primer corte separa a las empresas que gastan más o menos de unos 82.000 dólares en I+D, y los dos niveles siguientes también cortan por `RyD_Spend`; `MarketingSpend` y `Administration` aparecen recién en el cuarto. En la importancia, `RyD_Spend` se lleva casi todo (0,989) y `State` no aporta nada, como anticipaba la comparación por estado de 1.5.
#
# Los umbrales están en desvíos porque la consigna pide estandarizar. Al árbol no le cambia nada: compara cada variable con un umbral, y estandarizar mueve los umbrales sin cambiar qué empresas quedan de cada lado.

# %% [markdown]
# ### 2.4 Valores reales y predichos
#
# Si el error es de pocos miles de dólares, los puntos deberían quedar cerca de la diagonal, salvo, como anticipamos, las empresas atípicas en `Administration`.

# %%
# Profit real contra predicho en un test, adaptado de "Valores reales vs Predicciones" del notebook de árboles de
# regresión de la U4. En rojo, las empresas atípicas en Administration
def graficar_reales_predichos(ax, y_test, y_pred, titulo):
    """Profit real contra predicho, con la diagonal de la predicción perfecta."""
    grupo = np.where(y_test.index.isin(atipicas_admin.index), "atípica en Administration", "resto")
    sns.scatterplot(x=y_test, y=y_pred, hue=grupo, hue_order=["resto", "atípica en Administration"],
                    palette=["tab:blue", "tab:red"], s=20, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="gray", linestyle="--",
            label="predicción perfecta")
    ax.set(title=titulo, xlabel="Profit real (USD)", ylabel="Profit predicho (USD)")
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
graficar_reales_predichos(axes[0], y_test_80, y_pred_80, "Test 80/20")
graficar_reales_predichos(axes[1], y_test_70, y_pred_70, "Test 70/30")
fig.suptitle("Profit real y predicho por el árbol elegido")
plt.tight_layout()
plt.show()


# %%
def concentracion_del_error(y_test, y_pred):
    """Empresas con error menor a 1.000 USD y parte del error cuadrático que aportan las atípicas en Administration."""
    error = (y_test - y_pred).abs()
    es_atipica = y_test.index.isin(atipicas_admin.index)
    return {"Empresas en test": len(y_test),
            "% con error menor a 1.000 USD": round(100 * (error < 1000).mean()),
            "Atípicas en Administration": es_atipica.sum(),
            "% del error cuadrático que aportan": round(100 * (error[es_atipica] ** 2).sum() / (error ** 2).sum())}


pd.DataFrame({"80/20": concentracion_del_error(y_test_80, y_pred_80),
              "70/30": concentracion_del_error(y_test_70, y_pred_70)}).T.rename_axis("Test")

# %% [markdown]
# Se cumple. Nueve de cada diez empresas se predicen con menos de 1.000 dólares de error, y las atípicas en `Administration`, que son menos del 4 % del test, aportan cerca del 40 % del error cuadrático. El árbol predice su ganancia con `RyD_Spend`, y la de esas empresas se aparta de esa relación.
#
# Como los dos test comparten las empresas (1.7), los errores grandes son los mismos en los dos gráficos. Con 70/30 se promedian entre 99 empresas más, y por eso su RMSE de test es menor aunque el árbol se entrena con menos datos.

# %% [markdown]
# ---
# # Parte B – Clasificación (dataset drugType.csv)
#
# ## 3. Preparación de datos
#
# ### 3.1 Estructura, valores faltantes, duplicados y clases

# %%
drogas = pd.read_csv(RUTA_DROGAS, encoding="utf-8-sig")
drogas.head()

# %%
# Tipo de cada columna y cantidad de valores no nulos
drogas.info()

# %%
# Orden natural de cada categórica, de menor a mayor; se usa en los gráficos y en la codificación
ORDEN_CATEGORIAS = {"Sexo": ["F", "M"], "BP": ["Bajo", "Normal", "Alto"], "Colesterol": ["Normal", "Alto"]}
NUMERICAS_DROGAS = ["Edad", "Na_a_K"]

print(f"Filas duplicadas: {drogas.duplicated().sum()}")
for columna in ORDEN_CATEGORIAS:
    print(columna, drogas[columna].value_counts().to_dict())

# %% [markdown]
# Hay 200 pacientes con dos atributos numéricos (`Edad` y `Na_a_K`, la relación sodio/potasio en sangre) y tres categóricos (`Sexo`, `BP`, que es la presión arterial, y `Colesterol`). No hay faltantes ni duplicados, y las categorías de cada atributo tienen cantidades parecidas.

# %%
# Cantidad y porcentaje de pacientes de cada droga, como target_balance de U4_DesicionTreeClassifier.ipynb
pd.DataFrame({"Cantidad": drogas["Droga"].value_counts(),
              "Porcentaje": (drogas["Droga"].value_counts(normalize=True) * 100).round(1)})

# %% [markdown]
# Las clases están desbalanceadas: `DrogaY` es casi la mitad de los pacientes, y `DrogaB` y `DrogaC`, el 8 % cada una. Por eso estratificamos las particiones, como pide la consigna, y elegimos los modelos con el F1 macro, que pesa igual a las cinco drogas.

# %% [markdown]
# ### 3.2 Distribuciones por clase
#
# Si algún atributo numérico influye en la droga que se receta, su distribución debería cambiar de una droga a otra, y alguna droga podría ocupar un rango propio.

# %%
# Edad y Na_a_K de cada droga: diagrama de caja con cada paciente como un punto, como el boxplot con swarmplot de
# U3_Clustering.ipynb, separado por droga (los puntos ya muestran los atípicos, así que la caja no los repite)
ORDEN_DROGAS = sorted(drogas["Droga"].unique())
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, columna in zip(axes, NUMERICAS_DROGAS):
    sns.boxplot(data=drogas, x="Droga", y=columna, order=ORDEN_DROGAS, color="white", fliersize=0, ax=ax)
    sns.swarmplot(data=drogas, x="Droga", y=columna, order=ORDEN_DROGAS, size=3, ax=ax)
    ax.set(title=columna, xlabel="")
fig.suptitle("Atributos numéricos según la droga")
plt.tight_layout()
plt.show()

# %%
# Rango de Edad y de Na_a_K en cada droga
drogas.groupby("Droga")[NUMERICAS_DROGAS].agg(["min", "max"])

# %% [markdown]
# Se cumple, con dos fronteras muy claras. En `Na_a_K`, todos los pacientes de `DrogaY` tienen 15,0 o más, y los de las otras cuatro drogas, 14,6 o menos. En `Edad`, `DrogaA` y `DrogaB` no se superponen: todos los de `DrogaA` tienen 50 años o menos, y todos los de `DrogaB`, 51 o más.
#
# En los categóricos, esperamos que la presión y el colesterol, que son criterios clínicos habituales, definan algunas drogas, y que el sexo no influya.

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
# También se cumple. Con presión alta solo aparecen `DrogaA`, `DrogaB` y `DrogaY`, y los 16 pacientes de `DrogaC` tienen presión baja y colesterol alto. El sexo se reparte de forma parecida en todas las drogas.
#
# Este análisis anticipa cómo les va a ir a los modelos. Las drogas se separan con umbrales sobre un atributo a la vez: `Na_a_K` alrededor de 15 para `DrogaY`, la edad entre 50 y 51 para `DrogaA` y `DrogaB` con presión alta, y la combinación de presión baja y colesterol alto para `DrogaC`. Un árbol arma justamente ese tipo de fronteras, y por eso esperamos que acierte casi todo. Bayes ingenuo y k-NN no cortan por umbrales, y esperamos que sus errores se concentren cerca de esas fronteras. Lo vamos a verificar en cada modelo.

# %% [markdown]
# ### 3.3 Valores atípicos

# %%
# Atípicos de cada atributo numérico y, para Na_a_K, de qué droga son y entre qué valores están
display(contar_atipicos(drogas, NUMERICAS_DROGAS))
atipicos_na_k = drogas.drop(remove_outliers_iqr(drogas, "Na_a_K").index)
atipicos_na_k.groupby("Droga")["Na_a_K"].agg(["count", "min", "max"])

# %% [markdown]
# `Edad` no tiene atípicos, y los 8 de `Na_a_K` son todos de `DrogaY`, con valores entre 32,9 y 38,2. Los conservamos: son valores posibles y están del lado de `Na_a_K` alto, donde solo hay pacientes de `DrogaY`: sacarlos solo quitaría ejemplos de esa droga.

# %% [markdown]
# ### 3.4 Codificación de las variables categóricas
#
# Las tres categóricas tienen un orden natural, y se codifican con `OrdinalEncoder` respetándolo: `Sexo` F = 0 y M = 1, `BP` Bajo = 0, Normal = 1 y Alto = 2, y `Colesterol` Normal = 0 y Alto = 1. La variable objetivo queda como texto.

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
# No esperamos relaciones fuertes entre los atributos: la edad, el sexo, la presión, el colesterol y la relación sodio/potasio deberían variar de forma independiente en esta muestra.

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
# Se cumple: todas las correlaciones están entre −0,15 y 0,10, y los atributos son casi independientes entre sí. La variable objetivo no entra en la matriz porque es nominal, con cinco clases sin orden; su relación con los atributos la vimos en 3.2.

# %% [markdown]
# ### 3.6 Particiones estratificadas y estandarización
#
# Con la misma función de la Parte A, estratificando por `Droga` y estandarizando `Edad` y `Na_a_K`.

# %%
X_drogas = drogas_cod.drop(columns="Droga")
y_drogas = drogas_cod["Droga"]
X_train_80, X_test_80, y_train_80, y_test_80, escalador_80 = dividir_y_estandarizar(
    X_drogas, y_drogas, 0.2, NUMERICAS_DROGAS, estratificar_por=y_drogas)
X_train_70, X_test_70, y_train_70, y_test_70, escalador_70 = dividir_y_estandarizar(
    X_drogas, y_drogas, 0.3, NUMERICAS_DROGAS, estratificar_por=y_drogas)
print(f"80/20: {len(X_train_80)} pacientes de entrenamiento y {len(X_test_80)} de test")
print(f"70/30: {len(X_train_70)} pacientes de entrenamiento y {len(X_test_70)} de test")
print(f"Pacientes del test 80/20 que también están en el test 70/30: {y_test_80.index.isin(y_test_70.index).sum()}")

# %%
# Pacientes de cada droga en cada conjunto, como el balance de clases que el notebook de árboles de clasificación
# de la U4 muestra después de dividir
pd.DataFrame({"entrenamiento 80/20": y_train_80.value_counts(), "test 80/20": y_test_80.value_counts(),
              "entrenamiento 70/30": y_train_70.value_counts(), "test 70/30": y_test_70.value_counts()})

# %% [markdown]
# La estratificación mantiene las proporciones de las drogas en cada conjunto. Los test son chicos, con solo 3 o 5 pacientes de `DrogaB` y de `DrogaC`, y un solo error con esas drogas mueve mucho las métricas macro. Como en la Parte A, el test 80/20 está dentro del 70/30.

# %% [markdown]
# ## 4. Árboles de decisión para clasificación
#
# ### 4.1 Búsqueda de hiperparámetros
#
# Como en la Parte A, probamos muchas combinaciones de los tres hiperparámetros (504 en cada partición) con validación cruzada de 5 pliegues, ahora con el F1 macro. Con cinco drogas, un árbol necesita al menos cinco hojas para predecirlas todas: esperamos un F1 bajo con 1 o 2 niveles.

# %%
GRILLA_CLASIFICACION = {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, None],
                        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15],
                        "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30]}

arbol_clasificacion = DecisionTreeClassifier(random_state=SEMILLA)
busqueda_arbol_80 = buscar_hiperparametros(arbol_clasificacion, GRILLA_CLASIFICACION, "f1_macro",
                                           X_train_80, y_train_80)
busqueda_arbol_70 = buscar_hiperparametros(arbol_clasificacion, GRILLA_CLASIFICACION, "f1_macro",
                                           X_train_70, y_train_70)

# Configuración elegida en cada partición, con su F1 macro de validación
elegidos = pd.DataFrame([busqueda_arbol_80.best_params_, busqueda_arbol_70.best_params_], index=["80/20", "70/30"])
elegidos["F1 macro validación"] = [busqueda_arbol_80.best_score_, busqueda_arbol_70.best_score_]
elegidos.round(3)

# %%
# F1 macro de validación según max_depth: el mínimo y el máximo entre las combinaciones de los otros dos
# hiperparámetros. Si son iguales, todas esas combinaciones dan el mismo resultado
resultados_arbol = resultados_validacion(busqueda_arbol_80, busqueda_arbol_70)
resultados_arbol["param_max_depth"] = resultados_arbol["param_max_depth"].fillna("sin límite")
resultados_arbol.pivot_table(index="param_max_depth", columns="Partición", values="mean_test_score",
                             aggfunc=["min", "max"], sort=False).round(3)


# %% [markdown]
# Se cumple: con 1 o 2 niveles el F1 macro no pasa de 0,52, porque un árbol de 2 niveles tiene a lo sumo 4 hojas. Desde 4 niveles alcanza su máximo, y más profundidad no lo mejora. El mínimo de cada fila sale de las combinaciones que no permiten hojas chicas: con `min_samples_leaf = 15`, por ejemplo, ninguna hoja puede tener solo a los 13 pacientes de `DrogaB` o de `DrogaC` del entrenamiento.
#
# En las dos particiones elegimos **`max_depth = 4`, `min_samples_leaf = 1` y `min_samples_split = 2`**, la primera de las que empatan en el mejor F1.

# %% [markdown]
# ### 4.2 Post-poda
#
# Para el árbol elegido calculamos los valores de `ccp_alpha` en los que pierde hojas, con `cost_complexity_pruning_path` (la forma de obtenerlos en scikit-learn), y medimos cada uno con la misma validación cruzada.

# %%
def arbol_podado(busqueda, alpha, X_train, y_train):
    """Árbol con la mejor configuración de la búsqueda y el ccp_alpha indicado, entrenado con todo el entrenamiento."""
    arbol = DecisionTreeClassifier(random_state=SEMILLA, ccp_alpha=alpha, **busqueda.best_params_)
    return arbol.fit(X_train, y_train)


def camino_de_poda(busqueda, X_train, y_train):
    """Hojas y F1 macro de validación del árbol elegido con cada ccp_alpha de su camino de poda."""
    camino = busqueda.best_estimator_.cost_complexity_pruning_path(X_train, y_train)
    filas = []
    for alpha in camino.ccp_alphas:
        arbol = arbol_podado(busqueda, alpha, X_train, y_train)
        f1 = cross_val_score(arbol, X_train, y_train, cv=5, scoring="f1_macro").mean()
        filas.append({"ccp_alpha": alpha, "Hojas": arbol.get_n_leaves(), "F1 macro validación": f1})
    return pd.DataFrame(filas)


poda_80 = camino_de_poda(busqueda_arbol_80, X_train_80, y_train_80)
poda_70 = camino_de_poda(busqueda_arbol_70, X_train_70, y_train_70)
pd.concat({"80/20": poda_80, "70/30": poda_70}, names=["Partición", "Paso"]).round(4)

# %%
# El árbol final de cada partición usa el ccp_alpha con mayor F1 macro de validación
alpha_80 = poda_80.loc[poda_80["F1 macro validación"].idxmax(), "ccp_alpha"]
alpha_70 = poda_70.loc[poda_70["F1 macro validación"].idxmax(), "ccp_alpha"]
arbol_80 = arbol_podado(busqueda_arbol_80, alpha_80, X_train_80, y_train_80)
arbol_70 = arbol_podado(busqueda_arbol_70, alpha_70, X_train_70, y_train_70)
print(f"80/20: ccp_alpha = {alpha_80:.4f}, {arbol_80.get_n_leaves()} hojas")
print(f"70/30: ccp_alpha = {alpha_70:.4f}, {arbol_70.get_n_leaves()} hojas")

# %% [markdown]
# Elegimos **`ccp_alpha = 0`, es decir, no podar**, en las dos particiones: es el de mayor F1 macro. Ya la primera poda deja 4 hojas y baja el F1, y las siguientes lo bajan mucho más. Ninguna poda mejora este árbol porque no le sobran hojas: cada hoja tiene pacientes de una sola droga, como se ve en el gráfico siguiente, y sacar cualquiera junta drogas distintas.

# %% [markdown]
# ### 4.3 El árbol antes y después de la poda
#
# Por las fronteras de 3.2, esperamos que el árbol corte primero por `Na_a_K`, después por `BP`, y que use `Edad` y `Colesterol` para separar las drogas que comparten presión. La primera poda debería quitar una de las hojas que menos pacientes separan.

# %%
# Árbol de la partición 80/20 antes y después de la primera poda (el segundo ccp_alpha del camino de poda)
alpha_primera_poda = poda_80["ccp_alpha"].iloc[1]
primera_poda = arbol_podado(busqueda_arbol_80, alpha_primera_poda, X_train_80, y_train_80)

fig, axes = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={"width_ratios": [3, 2]})
plot_tree(arbol_80, feature_names=list(X_train_80.columns), class_names=ORDEN_DROGAS,
          filled=True, rounded=True, impurity=False, fontsize=9, precision=2, ax=axes[0])
axes[0].set_title(f"Antes de podar: el árbol elegido, {arbol_80.get_n_leaves()} hojas")
plot_tree(primera_poda, feature_names=list(X_train_80.columns), class_names=ORDEN_DROGAS,
          filled=True, rounded=True, impurity=False, fontsize=9, precision=2, ax=axes[1])
axes[1].set_title(f"Después de la primera poda (ccp_alpha = {alpha_primera_poda:.3f}): "
                  f"{primera_poda.get_n_leaves()} hojas")
fig.suptitle("Árbol de clasificación de la partición 80/20")
plt.show()

# %%
# Umbrales de Edad y Na_a_K del árbol de la partición 80/20, en desvíos estándar y en su unidad (umbral × desvío +
# media del entrenamiento). tree_.feature dice qué columna usa cada nodo, y tree_.threshold, su umbral
for posicion, columna in enumerate(NUMERICAS_DROGAS):   # posición de la columna en el escalador
    nodos = arbol_80.tree_.feature == list(X_train_80.columns).index(columna)
    for umbral in arbol_80.tree_.threshold[nodos]:
        en_unidad = umbral * escalador_80.scale_[posicion] + escalador_80.mean_[posicion]
        print(f"{columna} <= {umbral:.2f} desvíos equivale a {en_unidad:.1f}")


# %% [markdown]
# Se confirma. El árbol se resume en las mismas reglas que encontramos en 3.2:
#
# - `DrogaY` si `Na_a_K` es mayor a 14,8;
# - con presión alta, `DrogaA` hasta los 50 años y `DrogaB` desde los 51;
# - con presión baja y colesterol alto, `DrogaC`;
# - en el resto, `DrogaX`.
#
# No usa `Sexo`. Los umbrales de `BP` y `Colesterol` (1,5 y 0,5) son sobre los códigos de la codificación. La primera poda junta en una sola hoja, que predice `DrogaX`, toda la rama de presión baja o normal, y así pierde la hoja de los 13 pacientes de `DrogaC`, una de las dos más chicas.

# %% [markdown]
# ### 4.4 Resultados en test
#
# Si las reglas del árbol también valen fuera del entrenamiento, esperamos muy pocos errores en los dos test.

# %%
# Adaptada de tableResult de U4_DesicionTreeClassifier.ipynb: dibuja la matriz de confusión (con
# ConfusionMatrixDisplay, como en los notebooks de Bayes y k-NN de la U5) y devuelve las métricas de la consigna
def evaluar(y_test, y_pred, ax, titulo):
    """Matriz de confusión en el ax indicado, y exactitud, y precisión y exhaustividad macro, de unas predicciones."""
    # labels fija el orden de las drogas, así aparecen todas aunque el modelo no prediga alguna
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=ORDEN_DROGAS, cmap=plt.cm.Blues,
                                            colorbar=False, ax=ax)
    ax.set(title=titulo, xlabel="Predicción", ylabel="Real")
    # zero_division=0: si un modelo nunca predice una droga, su precisión se cuenta como 0
    return {"Accuracy": accuracy_score(y_test, y_pred),
            "Precision macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall macro": recall_score(y_test, y_pred, average="macro", zero_division=0)}


y_pred_arbol_80 = arbol_80.predict(X_test_80)
y_pred_arbol_70 = arbol_70.predict(X_test_70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metricas_arbol = pd.DataFrame({"80/20": evaluar(y_test_80, y_pred_arbol_80, axes[0], "Test 80/20"),
                               "70/30": evaluar(y_test_70, y_pred_arbol_70, axes[1], "Test 70/30")})
metricas_arbol = metricas_arbol.T.rename_axis("Test")
fig.suptitle("Árbol de decisión: matrices de confusión")
plt.tight_layout()
plt.show()
metricas_arbol.round(3)


# %% [markdown]
# Se cumple, sin ningún error: el árbol clasifica bien a todos los pacientes de los dos test, y las tres métricas valen 1. Con test tan chicos, acertar todos no garantiza acertar siempre con pacientes nuevos; el F1 macro de validación cruzada, de 0,97, es una estimación más prudente.

# %% [markdown]
# ## 5. Bayes ingenuo
#
# ### 5.1 Cantidad de intervalos
#
# Discretizamos `Edad` y `Na_a_K` en intervalos de igual frecuencia (cuantiles) y aplicamos `CategoricalNB` a los cinco atributos, que así quedan todos categóricos; `MultinomialNB` está pensado para conteos, y estos atributos son categorías. La consigna pide al menos 3 y 5 intervalos: probamos de 3 a 10 y elegimos, en cada partición, la cantidad con mayor F1 macro de validación.
#
# Por lo visto en 3.2, el resultado debería depender de que algún corte caiga cerca de las fronteras de `Na_a_K` (15) y de `Edad` (50 y 51).

# %%
def crear_bayes(n_intervalos):
    """Pipeline que discretiza Edad y Na_a_K por cuantiles en n_intervalos y aplica CategoricalNB."""
    cuantiles = KBinsDiscretizer(n_bins=n_intervalos, encode="ordinal", strategy="quantile")
    # Sexo, BP y Colesterol ya son categorías y pasan sin cambios (remainder="passthrough", como el
    # make_column_transformer del apunte de la Unidad 5)
    discretizador = ColumnTransformer([("intervalos", cuantiles, NUMERICAS_DROGAS)], remainder="passthrough")
    # Dentro de un pipeline (como el make_pipeline de Unidad4.py), en la validación cruzada los cortes se
    # calculan solo con los pliegues de entrenamiento
    return make_pipeline(discretizador, CategoricalNB())


def f1_por_intervalos(X_train, y_train):
    """F1 macro de validación cruzada de Bayes ingenuo con 3 a 10 intervalos."""
    f1 = {}
    for n_intervalos in range(3, 11):
        modelo = crear_bayes(n_intervalos)
        f1[n_intervalos] = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1_macro").mean()
    return pd.Series(f1)


f1_bayes = pd.DataFrame({"80/20": f1_por_intervalos(X_train_80, y_train_80),
                         "70/30": f1_por_intervalos(X_train_70, y_train_70)})
f1_bayes.index.name = "intervalos"
f1_bayes.round(3)

# %%
# En cada partición, la cantidad de intervalos con mayor F1 macro de validación
intervalos_80 = f1_bayes["80/20"].idxmax()
intervalos_70 = f1_bayes["70/30"].idxmax()
bayes_80 = crear_bayes(intervalos_80).fit(X_train_80, y_train_80)
bayes_70 = crear_bayes(intervalos_70).fit(X_train_70, y_train_70)

pd.DataFrame({"Intervalos": [intervalos_80, intervalos_70],
              "F1 macro validación": [f1_bayes["80/20"].max(), f1_bayes["70/30"].max()]},
             index=["80/20", "70/30"]).round(3)

# %%
# Cortes de igual frecuencia sobre los pacientes del entrenamiento 80/20. Los calculamos con los valores originales
# para leerlos en su unidad: los cuantiles no cambian al estandarizar, así que son los mismos que usa el modelo
originales_80 = drogas.loc[X_train_80.index]


def graficar_cortes(ax, atributo, n_intervalos):
    """Pacientes según su droga y su valor del atributo, con los cortes de n_intervalos de igual frecuencia."""
    cuantiles = KBinsDiscretizer(n_bins=n_intervalos, encode="ordinal", strategy="quantile")
    cortes = cuantiles.fit(originales_80[[atributo]]).bin_edges_[0][1:-1]   # sin el mínimo y el máximo
    sns.stripplot(data=originales_80, x=atributo, y="Droga", order=ORDEN_DROGAS, size=3, ax=ax)
    for corte in cortes:
        ax.axvline(corte, color="gray", linestyle="--")
    ax.set(title=f"{atributo}, {n_intervalos} intervalos: cortes en {np.round(cortes, 1).tolist()}", ylabel="")


fig, axes = plt.subplots(2, 2, figsize=(15, 7))
graficar_cortes(axes[0, 0], "Na_a_K", 3)
graficar_cortes(axes[0, 1], "Na_a_K", 4)
graficar_cortes(axes[1, 0], "Na_a_K", 9)
graficar_cortes(axes[1, 1], "Edad", 4)
fig.suptitle("Cortes de igual frecuencia sobre los pacientes del entrenamiento 80/20")
plt.tight_layout()
plt.show()

# %% [markdown]
# El F1 sube y baja con la cantidad de intervalos, y elegimos **4 intervalos con 80/20 y 9 con 70/30**, los de mayor F1 en cada partición, con el mismo criterio que en los otros modelos. Con pliegues de unos 30 pacientes, la diferencia con 7 o 9 intervalos en la partición 80/20 puede ser ruido.
#
# El gráfico muestra por qué, como anticipamos. Con 3 intervalos, el del medio de `Na_a_K` va de 11,4 a 18,3 y mezcla a `DrogaY` con las demás drogas: es la peor cantidad en las dos particiones. Con 4 hay un corte en 14,0, cerca de la frontera de 15, y con 9, uno justo en 15,0. La discretización por igual frecuencia no conoce las drogas, y acertar la frontera depende de la cantidad de intervalos.

# %% [markdown]
# ### 5.2 Resultados en test
#
# Esperamos errores en los pacientes cuyos valores caen en un intervalo que mezcla drogas. Con los 4 intervalos de la partición 80/20, el de 46,5 a 59 años junta pacientes de `DrogaA` (hasta 50) y de `DrogaB` (desde 51), y el de `Na_a_K` de 14,0 a 19,7 junta a `DrogaY` con pacientes de otras drogas.

# %%
y_pred_bayes_80 = bayes_80.predict(X_test_80)
y_pred_bayes_70 = bayes_70.predict(X_test_70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metricas_bayes = pd.DataFrame({"80/20": evaluar(y_test_80, y_pred_bayes_80, axes[0], "Test 80/20"),
                               "70/30": evaluar(y_test_70, y_pred_bayes_70, axes[1], "Test 70/30")})
metricas_bayes = metricas_bayes.T.rename_axis("Test")
fig.suptitle("Bayes ingenuo: matrices de confusión")
plt.tight_layout()
plt.show()
metricas_bayes.round(3)


# %%
def mal_clasificados(y_test, y_pred):
    """Pacientes del test mal clasificados, con la droga predicha y sus datos sin codificar."""
    prediccion = pd.Series(y_pred, index=y_test.index, name="Droga predicha")
    errores = prediccion[prediccion != y_test]
    return pd.concat([errores, drogas.loc[errores.index]], axis=1)


pd.concat({"80/20": mal_clasificados(y_test_80, y_pred_bayes_80),
           "70/30": mal_clasificados(y_test_70, y_pred_bayes_70)}, names=["Test", "Paciente"])

# %% [markdown]
# Tres de los cuatro errores del test 80/20 son los que esperábamos: dos pacientes de `DrogaB`, de 53 y 57 años, caen en el intervalo de 46,5 a 59 y se clasifican como `DrogaA`, y uno de `DrogaA`, con `Na_a_K` de 13,97, justo en el borde del corte de 14,0, como `DrogaY`. Con 70/30 hay un solo error, el mismo paciente de `DrogaB` de 53 años.
#
# El cuarto error muestra el supuesto de independencia del modelo. Un paciente de `DrogaC`, con presión baja y colesterol alto, se clasifica como `DrogaX`: Bayes ingenuo mira la presión y el colesterol por separado, y `DrogaX` tiene muchos pacientes con presión baja y muchos con colesterol alto, aunque esa combinación sea la de `DrogaC`. Con 80/20, esos dos pacientes de `DrogaB` bajan mucho la exhaustividad macro, porque son 2 de los 3 de esa droga en el test.

# %% [markdown]
# ## 6. k-NN
#
# ### 6.1 Cantidad de vecinos y distancia
#
# Probamos los k impares de 1 a 31 con las distancias euclidiana y Manhattan, y elegimos la combinación con mayor F1 macro de validación. `Edad` y `Na_a_K` ya están estandarizadas, y las categóricas codificadas van de 0 a 1 o de 0 a 2: ningún atributo pesa más en la distancia solo por su escala.
#
# Con k chico, el vecino más cercano puede ser un paciente de otra droga por un detalle que no importa, como el sexo. Con k grande, las drogas con pocos pacientes quedan en minoría entre los vecinos. Esperamos el mejor F1 en un k intermedio.

# %%
# k impares, como el range(1, 15, 2) de Unidad5.py pero más amplio, y GridSearchCV con best_estimator_ como en
# el notebook de k-NN de la U5
GRILLA_KNN = {"n_neighbors": list(range(1, 32, 2)), "metric": ["euclidean", "manhattan"]}

busqueda_knn_80 = buscar_hiperparametros(KNeighborsClassifier(), GRILLA_KNN, "f1_macro", X_train_80, y_train_80)
busqueda_knn_70 = buscar_hiperparametros(KNeighborsClassifier(), GRILLA_KNN, "f1_macro", X_train_70, y_train_70)
knn_80 = busqueda_knn_80.best_estimator_
knn_70 = busqueda_knn_70.best_estimator_

# Configuración elegida en cada partición, con su F1 macro de validación
elegidos = pd.DataFrame([busqueda_knn_80.best_params_, busqueda_knn_70.best_params_], index=["80/20", "70/30"])
elegidos["F1 macro validación"] = [busqueda_knn_80.best_score_, busqueda_knn_70.best_score_]
elegidos.round(3)

# %%
# F1 macro de validación según k, una línea por distancia y un panel por partición
resultados_knn = resultados_validacion(busqueda_knn_80, busqueda_knn_70)
resultados_knn = resultados_knn.rename(columns={"param_n_neighbors": "k", "param_metric": "Distancia",
                                                "mean_test_score": "F1 macro de validación"})
resultados_knn["k"] = resultados_knn["k"].astype(int)
grafico = sns.relplot(data=resultados_knn, x="k", y="F1 macro de validación", hue="Distancia", col="Partición",
                      kind="line", marker="o", height=4, aspect=1.5)
grafico.set(xticks=GRILLA_KNN["n_neighbors"])
grafico.figure.suptitle("k-NN: F1 macro de validación cruzada según k y la distancia", y=1.03)
plt.show()

# %% [markdown]
# Se cumple en parte. Con k grande el F1 cae en las dos particiones, porque las drogas con pocos pacientes quedan en minoría entre los vecinos. Pero el máximo no siempre está en un k intermedio: con 70/30 sí (k = 11), y con 80/20 es k = 1. Para k chicos las curvas van en zigzag, ya que cada pliegue de validación tiene solo 2 o 3 pacientes de `DrogaB` y de `DrogaC`, y un solo error mueve mucho el F1 macro. La distancia Manhattan queda por encima de la euclidiana en casi todos los k.
#
# Elegimos **Manhattan, con k = 1 en 80/20 y k = 11 en 70/30**, con el mismo criterio que en los otros modelos: el mayor F1 promedio de validación. En 80/20 la diferencia con k = 13 es tan chica que puede ser ruido, y k = 1 es el más sensible a un único vecino de otra droga.

# %% [markdown]
# ### 6.2 Resultados en test
#
# Esperamos errores cerca de la frontera de `Na_a_K` (alrededor de 15), donde los vecinos de un paciente pueden ser de otras drogas con la misma presión y colesterol. Con k = 1, además, puede haber algún error lejos de esa frontera, si el único vecino es de otra droga.

# %%
y_pred_knn_80 = knn_80.predict(X_test_80)
y_pred_knn_70 = knn_70.predict(X_test_70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metricas_knn = pd.DataFrame({"80/20": evaluar(y_test_80, y_pred_knn_80, axes[0], "Test 80/20"),
                             "70/30": evaluar(y_test_70, y_pred_knn_70, axes[1], "Test 70/30")})
metricas_knn = metricas_knn.T.rename_axis("Test")
fig.suptitle("k-NN: matrices de confusión")
plt.tight_layout()
plt.show()
metricas_knn.round(3)

# %%
pd.concat({"80/20": mal_clasificados(y_test_80, y_pred_knn_80),
           "70/30": mal_clasificados(y_test_70, y_pred_knn_70)}, names=["Test", "Paciente"])

# %% [markdown]
# Se cumple. Casi todos los pacientes de `DrogaY` mal clasificados tienen `Na_a_K` entre 15,2 y 16,6, pegados a la frontera, y reciben la droga que corresponde a su presión. El único error lejos de la frontera es el que anticipábamos para k = 1: un paciente de `DrogaY` con 19,8, con presión baja y colesterol alto, el perfil de `DrogaC`, toma la droga de su único vecino.

# %% [markdown]
# ## 7. Comparación de modelos
#
# Comparamos los tres modelos en los dos conjuntos de test, cada uno con su mejor configuración. Son las mismas métricas de 4.4, 5.2 y 6.2, juntas en una tabla.

# %%
# Métricas en los dos test del mejor modelo de cada algoritmo, como la tabla de métricas del notebook de Bayes de
# la U5
pd.concat({"Árbol de decisión": metricas_arbol, "Bayes ingenuo": metricas_bayes, "k-NN": metricas_knn},
          names=["Modelo", "Test"]).round(3)

# %% [markdown]
# En los dos test, el árbol de decisión es el mejor, con 1 en las tres métricas. Bayes ingenuo queda muy cerca con 70/30 (un solo error) y más lejos con 80/20, donde los dos pacientes de `DrogaB` mal clasificados le bajan la exhaustividad macro. k-NN queda por debajo de los dos en exactitud en las dos particiones.
#
# El resultado es el que anticipaba el análisis de 3.2. Cada droga se define con umbrales sobre un atributo a la vez, que es la forma de las fronteras que arma un árbol. Bayes depende de que los cortes por cuantiles caigan cerca de esas fronteras, y k-NN suma todos los atributos en una sola distancia, de modo que un paciente pegado a una frontera puede tener vecinos de otra droga.
#
# El árbol es además el más fácil de interpretar: son cuatro reglas que se pueden leer y comprobar, como «relación sodio/potasio mayor a 14,8» o «presión alta y hasta 50 años». Bayes da probabilidades por intervalo, y k-NN no da reglas, solo los pacientes parecidos. Elegimos **el árbol de decisión (`max_depth = 4`, sin post-poda)**, por sus métricas y por su interpretabilidad.

# %% [markdown]
# ---
# ## 8. Conclusiones
#
# 1. **Datos de empresas:** sacamos una fila duplicada y 3 empresas con errores de carga, que ganaban hasta 476.000 dólares sin seguir la relación con los gastos, y conservamos las atípicas en `Administration`, que tienen montos posibles. La ganancia depende casi solo del gasto en I+D (correlación de 0,999).
# 2. **Árbol de regresión:** se equivoca en promedio en unos mil dólares, menos de la décima parte del desvío de la ganancia, y usa casi solo `RyD_Spend`, como esperábamos. Por eso sus errores grandes se concentran en las empresas atípicas en `Administration`, que aportan cerca del 40 % del error cuadrático siendo menos del 4 % del test.
# 3. **Datos de pacientes:** como las clases están desbalanceadas, estratificamos las particiones y elegimos con el F1 macro. El análisis exploratorio encontró reglas casi exactas: `DrogaY` con relación sodio/potasio de 15 o más, y las demás según la presión, la edad (hasta o desde los 50 años) y el colesterol. `Sexo` no influye.
# 4. **Modelos de clasificación:** el árbol (6 hojas, sin poda) aprende esas mismas reglas y clasifica bien todo el test en las dos particiones. Bayes ingenuo y k-NN fallan justo donde anticipábamos: Bayes, cuando un intervalo mezcla drogas, y k-NN, con pacientes pegados a la frontera de `Na_a_K`. Elegimos el árbol, por sus métricas y porque sus reglas se pueden leer y verificar.
# 5. **Las dos particiones:** con la misma semilla, el test 80/20 está dentro del 70/30, y los casos difíciles son los mismos. En la Parte B, con test de 40 y 60 pacientes, un solo error mueve varios puntos las métricas macro; la validación cruzada da una estimación más estable.
# 6. **Lo que anticipó el análisis exploratorio:** se confirmó en las dos partes. La dependencia de `RyD_Spend` y las fronteras de `Na_a_K` y de `Edad`, encontradas antes de modelar, explicaron qué variables usa cada modelo y dónde se equivoca.
