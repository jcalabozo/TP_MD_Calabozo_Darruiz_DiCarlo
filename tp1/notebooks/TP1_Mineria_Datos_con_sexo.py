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
# # Trabajo Práctico N° 1 — Minería de Datos (versión con `Sexo`)
#
# | | |
# |---|---|
# | **Año** | 2026 |
# | **Materia** | Minería de Datos — Tecnicatura Universitaria en Inteligencia Artificial |
# | **Integrantes** | Josías Calabozo · Ismael Darruiz · Sebastian Di Carlo |
#
# **Objetivo:** integrar los contenidos de las unidades 2 (reducción de la dimensionalidad) y 3 (aprendizaje no supervisado) sobre el conjunto de datos de pingüinos del archipiélago Palmer (`penguins_size.csv`). La variable **Especie** es nuestro objetivo.
#
# **Esta versión** repite el análisis de `TP1_Mineria_Datos`, pero usa `Sexo` como característica en lugar de excluirla. El objetivo sigue siendo separar las especies: se busca ver cómo incide `Sexo` en esa separación. Hasta la sección 1.8 el análisis es el mismo, y la sección 7 compara directamente los resultados con y sin `Sexo`.
#
# **Contenido**
# 1. Análisis exploratorio y preparación de los datos
# 2. PCA
# 3. Isomap
# 4. t-SNE
# 5. K-means
# 6. Clustering jerárquico
# 7. Efecto de `Sexo` en la separación por especie
# 8. Conclusiones

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
from IPython.display import display

# Avisos de Isomap que no afectan los resultados (el del grafo
# desconectado se analiza en la sección 3)
warnings.filterwarnings("ignore", message="Changing the sparsity structure")
warnings.filterwarnings("ignore",
                        message="The number of connected components")

SEMILLA = 42
sns.set_theme(style="whitegrid")

# Mismo color para cada especie en todos los gráficos
PALETA_ESPECIES = {"Adelie": "tab:blue",
                   "Chinstrap": "tab:orange",
                   "Gentoo": "tab:green"}
# Colores del sexo, distintos de los de las especies
PALETA_SEXO = {"Hembra": "tab:purple", "Macho": "tab:gray"}

# %% [markdown]
# ## 1. Análisis exploratorio y preparación de los datos
#
# ### 1.1 Carga del conjunto de datos
#
# El archivo usa `;` como separador y cada fila termina en `;;`, lo que genera dos columnas vacías que se descartan al cargar.

# %%
datos = pd.read_csv("../data/penguins_size.csv", sep=";")
# Columnas vacías generadas por el ';;' final de cada fila
datos = datos.dropna(axis=1, how="all")

columnas_numericas = ["Longitud Culmen (mm)", "Profundidad Culmen (mm)",
                      "Longitud Aleta (mm)", "Masa corporal (g)"]

print("Dimensiones:", datos.shape)
datos.head()

# %%
datos.describe().round(2)

# %% [markdown]
# 347 registros con 4 medidas numéricas (longitud y profundidad del culmen, que es la parte superior del pico, longitud de la aleta y masa corporal), `Sexo` y la variable objetivo `Especie`. Las escalas son muy distintas (la masa está en miles de gramos y la profundidad del culmen ronda los 17 mm), así que habrá que estandarizar.

# %% [markdown]
# ### 1.2 Valores faltantes, valores inválidos y duplicados

# %%
print("Valores faltantes por columna:")
print(datos.isna().sum())
print("\nValores de la columna 'Sexo':")
print(datos["Sexo"].value_counts(dropna=False))
print("\nFilas duplicadas:", datos.duplicated().sum())

# %%
# Filas con medidas faltantes
datos[datos[columnas_numericas].isna().any(axis=1)]

# %% [markdown]
# - **2 filas sin medidas:** se eliminan, porque imputarlas sería inventar el individuo completo.
# - **3 filas duplicadas:** se eliminan.
# - **`Sexo`:** tiene 10 faltantes y un valor inválido `"."`, que pasa a faltante. Como las 2 filas sin medidas tampoco tienen sexo, después de eliminarlas quedan 9 pingüinos sin sexo, todos con sus cuatro medidas completas. Los valores se traducen (`"MALE"` → `"Macho"`, `"FEMALE"` → `"Hembra"`). Qué hacer con esta variable y con sus faltantes se analiza en 1.8.
# - Se acortan los nombres de las especies (`"Adelie Penguin"` → `"Adelie"`).

# %%
datos = datos.drop_duplicates()
datos = datos.dropna(subset=columnas_numericas)
datos["Sexo"] = datos["Sexo"].replace({".": np.nan, "MALE": "Macho",
                                       "FEMALE": "Hembra"})
datos["Especie"] = datos["Especie"].str.split().str[0]
datos = datos.reset_index(drop=True)

print("Dimensiones luego de la limpieza:", datos.shape)
print(datos.isna().sum())

# %% [markdown]
# ### 1.3 Distribución de la variable objetivo

# %%
conteo = datos["Especie"].value_counts()
pd.DataFrame({"Cantidad": conteo,
              "Proporción (%)": (100 * conteo / len(datos)).round(1)})

# %% [markdown]
# Hay un desbalance moderado: *Chinstrap* es solo el 20% de los datos, por lo que podría quedar absorbida por otro grupo al hacer clustering. En principio esperamos encontrar tres grupos, uno por especie.

# %% [markdown]
# ### 1.4 Diferencias entre especies
#
# Promedio y desvío estándar de cada medida por especie:

# %%
datos.groupby("Especie")[columnas_numericas].agg(["mean", "std"]).round(1).T

# %% [markdown]
# *Gentoo* se diferencia de las otras dos especies en la aleta (entre 21 y 27 mm más larga), la masa (entre 1.3 y 1.4 kg más) y la profundidad del culmen (más de 3 mm menos). *Adelie* y *Chinstrap* son casi iguales en esas tres variables y solo se distinguen por la longitud del culmen, unos 10 mm mayor en *Chinstrap*.

# %% [markdown]
# ### 1.5 Distribuciones y relaciones entre variables
#
# Por la tabla anterior, esperamos ver a *Gentoo* separada en los gráficos con profundidad del culmen, aleta o masa, y a *Adelie* y *Chinstrap* separadas solo cuando interviene la longitud del culmen.

# %%
sns.pairplot(datos, vars=columnas_numericas, hue="Especie",
             palette=PALETA_ESPECIES, diag_kind="kde", corner=True,
             plot_kws={"s": 18, "alpha": 0.7})
plt.suptitle("Distribuciones y relaciones entre variables por especie",
             y=1.02)
plt.show()


# %% [markdown]
# Coincide con lo esperado. *Gentoo* aparece separada, o apenas en contacto con el resto, en todos los gráficos con profundidad del culmen, aleta o masa. La primera columna (longitud del culmen contra cada una de las otras variables) muestra tres nubes separadas; en los demás gráficos *Adelie* y *Chinstrap* se superponen. Algunas distribuciones tienen dos picos (por ejemplo, la masa de *Gentoo*), lo que sugiere subgrupos dentro de cada especie; en 1.8 se verifica si se deben al sexo.

# %% [markdown]
# ### 1.6 Valores atípicos (outliers)
#
# Como las especies tienen promedios muy distintos, se buscan atípicos en todo el conjunto (como en la U2) y también dentro de cada especie (como en la U3).

# %%
# Función de U3_Clustering.ipynb (sección "Remoción de Outliers"), con
# los nombres de las variables en minúscula según PEP 8. Conserva las
# filas dentro de [Q1 - 1.5·IQR, Q3 + 1.5·IQR] en la columna indicada.
def remove_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


# Atípicos sobre todo el conjunto, como en el notebook de la U2
for col in columnas_numericas:
    n_atipicos = len(datos) - len(remove_outliers_iqr(datos, col))
    print(f"Atípicos en {col}: {n_atipicos}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, col in zip(axes, columnas_numericas):
    sns.boxplot(data=datos, x="Especie", y=col, hue="Especie",
                palette=PALETA_ESPECIES, legend=False, ax=ax)
    ax.set(title=col, xlabel="", ylabel="")
fig.suptitle("Distribución de cada variable por especie")
plt.tight_layout()
plt.show()

# %% [markdown]
# No hay atípicos en el conjunto completo, pero sí algunos dentro de cada especie (por ejemplo, un *Gentoo* con culmen de 59.6 mm o un *Chinstrap* de 2700 g). Siguiendo el criterio de la U3, se eliminan:

# %%
# Como en la U3, los atípicos se buscan dentro de cada especie
# (aquí, en las cuatro variables)
grupos_limpios = []
for _, grupo in datos.groupby("Especie"):
    for col in columnas_numericas:
        grupo = remove_outliers_iqr(grupo, col)
    grupos_limpios.append(grupo)
datos_limpios = pd.concat(grupos_limpios)

print("Filas eliminadas:", len(datos) - len(datos_limpios))
display(datos.drop(datos_limpios.index))

datos = datos_limpios.sort_index().reset_index(drop=True)
print("Dimensiones finales:", datos.shape)

# %% [markdown]
# ### 1.7 Correlaciones
#
# En el pairplot, aleta y masa muestran una relación casi lineal, así que esperamos una correlación positiva fuerte entre ellas.

# %%
plt.figure(figsize=(6.5, 5))
sns.heatmap(datos[columnas_numericas].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, square=True)
plt.title("Matriz de correlación")
plt.show()

# %% [markdown]
# Se confirma: aleta y masa tienen una correlación de 0.87, y ambas se relacionan con la longitud del culmen (0.65 y 0.59). La profundidad del culmen es negativa con el resto por efecto de *Gentoo* (aleta larga, mayor masa y culmen poco profundo); dentro de cada especie, en cambio, la relación es positiva. Esta redundancia sugiere que PCA podrá resumir los datos en pocas componentes.

# %% [markdown]
# ### 1.8 La variable `Sexo`
#
# Además de `Especie` (objetivo), queda una sola variable categórica: `Sexo`. Antes de decidir si se usa y qué hacer con sus faltantes, se analiza cómo se distribuye y qué relación tiene con la especie y con las medidas.
#
# #### Distribución del sexo por especie
#
# En una población natural esperamos una proporción cercana a 50/50 de machos y hembras en cada especie. Si es así, el sexo no aporta información para distinguir especies.

# %%
# Cantidad de pingüinos por especie y sexo ("Sin dato": sexo faltante)
display(pd.crosstab(datos["Especie"], datos["Sexo"].fillna("Sin dato"),
                    margins=True, margins_name="Total"))

# Proporción de cada sexo dentro de cada especie (solo sexo conocido)
pd.crosstab(datos["Especie"], datos["Sexo"], normalize="index").round(3)

# %% [markdown]
# Coincide con lo esperado. Entre los pingüinos de sexo conocido hay 163 hembras y 164 machos, y en cada especie la diferencia entre ambos sexos es de 0 a 2 individuos (entre 49% y 51% de cada sexo). Es decir, saber el sexo de un pingüino no dice nada sobre su especie.
#
# Los 9 faltantes (2.7% de los datos) están solo en *Adelie* (5) y *Gentoo* (4); todos los *Chinstrap* tienen el sexo registrado.
#
# #### Medidas por especie y sexo
#
# Aunque el sexo no distinga especies, puede influir en las medidas. Si los machos son más grandes que las hembras, eso explicaría los dos picos vistos en algunas distribuciones del pairplot (1.5).

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col in zip(axes, columnas_numericas):
    sns.violinplot(data=datos, x="Especie", y=col, hue="Sexo",
                   hue_order=["Hembra", "Macho"], palette=PALETA_SEXO,
                   split=True, inner="quart",
                   legend=(col == columnas_numericas[0]), ax=ax)
    ax.set(title=col, xlabel="", ylabel="")
fig.suptitle("Distribución de cada medida por especie y sexo "
             "(las líneas marcan los cuartiles)")
plt.tight_layout()
plt.show()

# %%
# Diferencia de medias macho − hembra dentro de cada especie, junto a la
# diferencia entre las dos especies más parecidas (Chinstrap − Adelie)
medias_sexo = datos.groupby(["Especie", "Sexo"])[columnas_numericas].mean()
medias_especie = datos.groupby("Especie")[columnas_numericas].mean()

diferencias = pd.DataFrame(
    {f"Macho − hembra ({especie})":
         medias_sexo.loc[(especie, "Macho")]
         - medias_sexo.loc[(especie, "Hembra")]
     for especie in ["Adelie", "Chinstrap", "Gentoo"]})
diferencias["Chinstrap − Adelie"] = (medias_especie.loc["Chinstrap"]
                                     - medias_especie.loc["Adelie"])
diferencias.round(1)

# %% [markdown]
# Coincide con lo esperado: en las tres especies los machos superan a las hembras en las cuatro medidas. La diferencia más marcada es la masa (entre 360 y 800 g); en *Gentoo* los picos de machos y hembras están separados por unos 800 g y se superponen poco, y esa es la causa de los dos picos del pairplot.
#
# La tabla muestra algo importante para el resto del trabajo. Entre *Adelie* y *Chinstrap*, la diferencia por sexo es **mayor que la diferencia por especie** en la profundidad del culmen (alrededor de 1.5 mm contra 0.1 mm) y en la masa (360 a 670 g contra 34 g), y comparable en la aleta (4 a 8 mm contra 6 mm). Solo la longitud del culmen separa más a las especies (10 mm) que a los sexos (3 a 4.5 mm). Por lo tanto, al agrupar por distancias es posible que *Adelie* y *Chinstrap* se dividan por tamaño (sexo) antes que por especie; esto se revisa en las secciones 5 y 6.
#
# #### `Sexo` como característica
#
# En esta versión `Sexo` se usa como característica, codificada como 0/1 en una columna `Macho` (1 = macho).
#
# Sus 9 faltantes se eliminan. Con un reparto 50/50, la moda (global o por especie, como en la U2) se decide por uno o dos individuos, y la categoría "Desconocido" crea un valor que no existe en el fenómeno estudiado. Como ahora `Sexo` entra en todos los métodos, tampoco se pueden dejar como faltantes.
#
# **Hipótesis:** al estandarizar, `Macho` pesa lo mismo que cada medida y, a diferencia de ellas, divide los datos en dos grupos perfectamente separados. Como *Adelie* y *Chinstrap* se diferencian más por sexo que por especie en tres medidas, esperamos que incluir `Sexo` **empeore la separación por especie**: que los métodos dividan a cada especie en machos y hembras, y que agrupen a *Adelie* y *Chinstrap* por sexo antes que por especie.

# %%
# Se eliminan los pingüinos sin sexo y se codifica Sexo como 0/1
datos = datos.dropna(subset=["Sexo"]).reset_index(drop=True)
datos["Macho"] = datos["Sexo"].map({"Hembra": 0, "Macho": 1})
caracteristicas = columnas_numericas + ["Macho"]

print("Dimensiones finales:", datos.shape)
datos["Especie"].value_counts()

# %% [markdown]
# Quedan 327 pingüinos: se pierden 5 *Adelie* y 4 *Gentoo*, y *Chinstrap* queda completa.

# %% [markdown]
# ### 1.9 Estandarización
#
# `Especie` se guarda en `y` y solo se usa para colorear los gráficos y comparar resultados. Las cuatro medidas y `Macho` se estandarizan.

# %%
X = datos[caracteristicas]
y = datos["Especie"]

escalador = StandardScaler()
X_esc = escalador.fit_transform(X)

# %% [markdown]
# ## 2. PCA
#
# Esperamos que la primera componente siga siendo de tamaño (donde se separa *Gentoo*), como en la versión principal, y que `Macho` ocupe una componente propia que desplace a la del culmen (donde se separan *Adelie* y *Chinstrap*).

# %%
pca = PCA()
pca_componentes = pca.fit_transform(X_esc)
var_exp = pca.explained_variance_ratio_
var_cum = np.cumsum(var_exp)
componentes = list(range(1, len(var_exp) + 1))

pd.DataFrame({"Autovalor": pca.explained_variance_,
              "Varianza explicada (%)": 100 * var_exp,
              "Varianza acumulada (%)": 100 * var_cum},
             index=[f"PC{c}" for c in componentes]).round(2)

# %% [markdown]
# Se aplican los tres criterios de la U2: varianza acumulada (umbral del 80%), Kaiser y codo.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

# Varianza explicada (barras) y acumulada (línea), como en la U2
axes[0].bar(componentes, var_exp, alpha=0.8, label="Varianza explicada")
axes[0].plot(componentes, var_cum, color="red", marker="o",
             label="Varianza acumulada")
axes[0].axhline(0.8, color="gray", linestyle="--", label="Umbral 80%")
for x, v in zip(componentes, var_cum):
    axes[0].text(x, v + 0.02, f"{v:.2f}", color="red", ha="center")
axes[0].set(title="Varianza explicada y acumulada",
            xlabel="Componente principal", ylabel="Proporción de varianza",
            xticks=componentes, ylim=(0, 1.12))
axes[0].legend(loc="center right")

# Gráfico del codo con los autovalores y el criterio de Kaiser
axes[1].plot(componentes, pca.explained_variance_, "o-", linewidth=2)
axes[1].axhline(1, color="gray", linestyle="--",
                label="Criterio de Kaiser (autovalor = 1)")
axes[1].set(title="Gráfico del codo", xlabel="Componente principal",
            ylabel="Autovalor", xticks=componentes)
axes[1].legend()

plt.tight_layout()
plt.show()

print("Varianza acumulada ≥ 80%:", np.argmax(var_cum >= 0.8) + 1,
      "componentes")
print("Kaiser (autovalor > 1):", np.sum(pca.explained_variance_ > 1),
      "componentes")
print("Codo: 3 componentes (se lee en el gráfico)")

# %% [markdown]
# Varianza acumulada y Kaiser indican **2 componentes** (84.9% de la varianza; PC1 y PC2 tienen autovalor mayor a 1). El codo es menos claro: la curva baja fuerte hasta PC3 y recién después se aplana, lo que sugiere 3 componentes (94.6%). Elegimos el criterio de **varianza acumulada**, que coincide con Kaiser, permite graficar en 2D y usa el mismo número de componentes que la versión principal, así que las dos se pueden comparar en las mismas condiciones.

# %%
# Cargas: peso de cada variable original en las tres primeras componentes
pd.DataFrame(pca.components_[:3].T, index=caracteristicas,
             columns=["PC1", "PC2", "PC3"]).round(2)

# %% [markdown]
# PC1 combina aleta, masa y longitud del culmen, con signo opuesto a la profundidad del culmen: sigue representando el **tamaño corporal**. PC2 depende sobre todo de `Macho` (0.74) y de la profundidad del culmen (0.64): es una componente de **sexo**. La longitud del culmen, que en la versión principal definía PC2 y separaba a *Adelie* de *Chinstrap*, pasa a PC3 (0.85), que explica el 9.8% de la varianza y queda afuera con 2 componentes. Coincide con lo esperado.

# %%
# La forma del marcador indica el sexo
plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(x=pca_componentes[:, 0], y=pca_componentes[:, 1], hue=y,
                style=datos["Sexo"], palette=PALETA_ESPECIES, s=35,
                alpha=0.8)
plt.title("PCA: proyección sobre las dos primeras componentes")
plt.xlabel(f"PC1 ({var_exp[0]:.1%} de la varianza)")
plt.ylabel(f"PC2 ({var_exp[1]:.1%} de la varianza)")
plt.show()

# %% [markdown]
# Aparecen cuatro grupos: PC1 separa a *Gentoo* y PC2 separa a los machos (arriba) de las hembras (abajo). La separación por especie empeora respecto de la versión principal: *Adelie* y *Chinstrap* del mismo sexo quedan juntas, con *Chinstrap* apenas a la derecha y bastante superposición, mientras que los machos y las hembras de una misma especie quedan lejos. La componente que mejor separa a estas dos especies (PC3) quedó afuera. Coincide con lo esperado.

# %% [markdown]
# ## 3. Isomap
#
# Esperamos que con pocos vecinos los grupos queden más separados y que con muchos el resultado se parezca al de PCA. Como *Gentoo* está muy alejada del resto, con pocos vecinos el grafo podría quedar dividido, algo que las diapositivas de la U2 señalan como un problema. Como ahora cada especie se divide además por sexo, esperamos que el grafo se parta en más partes que en la versión principal. Se prueban 5, 30, 60 y 100 vecinos:

# %%
VECINOS = [5, 30, 60, 100]

# Diapositivas U2 ("ISOMAP - Consideraciones prácticas"): con k chico
# el grafo de vecinos puede quedar desconectado. kneighbors_graph
# (scikit-learn) arma el grafo que usa Isomap y connected_components
# (scipy) cuenta en cuántas partes separadas queda.
for k in VECINOS:
    grafo = kneighbors_graph(X_esc, n_neighbors=k)
    n_partes, _ = connected_components(grafo)
    print(f"{k} vecinos: el grafo tiene {n_partes} parte(s)")

# Qué pingüinos quedan en cada parte con 5 vecinos
grafo = kneighbors_graph(X_esc, n_neighbors=5)
n_partes, partes = connected_components(grafo)
pd.crosstab([y, datos["Sexo"]], partes, rownames=["Especie", "Sexo"],
            colnames=["Parte del grafo (5 vecinos)"])

# %% [markdown]
# Con 5 vecinos el grafo queda partido en 4 partes, que son exactamente los machos y las hembras de *Gentoo* y los machos y las hembras de *Adelie* + *Chinstrap*. Con 30 queda en 2 y con 60 y 100, conectado. Coincide con lo esperado: en la versión principal, con 5 vecinos el grafo quedaba en 2 partes. Cuando el grafo está partido, scikit-learn une las partes por los puntos más cercanos, lo que puede deformar el resultado.
#
# Error de reconstrucción según el número de vecinos y de componentes:

# %%
# reconstruction_error() de scikit-learn compara las distancias entre
# puntos del espacio original con las del espacio reducido: cuanto
# menor es, mejor se conserva la estructura de los datos.
# Diapositivas U2 ("ISOMAP - Consideraciones prácticas").
# eigen_solver="dense" hace el cálculo exacto: por defecto Isomap usa
# un método que parte de un vector aleatorio y el resultado cambia
# mínimamente entre ejecuciones.
errores = []
for k in VECINOS:
    fila = []
    for c in componentes:
        isomap = Isomap(n_neighbors=k, n_components=c,
                        eigen_solver="dense").fit(X_esc)
        fila.append(isomap.reconstruction_error())
    errores.append(fila)

errores = pd.DataFrame(errores, index=VECINOS, columns=componentes)
errores.rename_axis(index="Vecinos", columns="Componentes").round(3)

# %% [markdown]
# En todos los casos el error baja al agregar componentes, sobre todo al pasar de 1 a 2; con 5 y 30 vecinos también baja bastante de 2 a 3. Las filas no se comparan entre sí, porque cada número de vecinos arma un grafo distinto.

# %%
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, k in zip(axes, VECINOS):
    isomap_2d = Isomap(n_neighbors=k, n_components=2,
                       eigen_solver="dense").fit_transform(X_esc)
    sns.scatterplot(x=isomap_2d[:, 0], y=isomap_2d[:, 1], hue=y,
                    palette=PALETA_ESPECIES, s=18, alpha=0.8,
                    legend=(k == 5), ax=ax)
    ax.set(title=f"n_neighbors = {k}", xlabel="Componente 1",
           ylabel="Componente 2")
fig.suptitle("Isomap con 2 componentes variando el número de vecinos")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con 5 vecinos cada especie se fragmenta y parte de *Chinstrap* queda estirada sobre una línea. Con 30 el resultado se deforma en líneas, y una de ellas son solo *Gentoo*, efecto del grafo partido. Con 60 aparecen cuatro grupos: dos de *Gentoo* y dos de *Adelie* + *Chinstrap*. Con 100 el resultado es parecido al de PCA, como se esperaba.

# %%
# La forma del marcador indica el sexo
isomap_2d = Isomap(n_neighbors=60, n_components=2,
                   eigen_solver="dense").fit_transform(X_esc)

plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(x=isomap_2d[:, 0], y=isomap_2d[:, 1], hue=y,
                style=datos["Sexo"], palette=PALETA_ESPECIES, s=35,
                alpha=0.8)
plt.title("Isomap 2D (n_neighbors = 60)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()

# %% [markdown]
# Elegimos **60 vecinos**, el menor valor probado con el grafo conectado. Los cuatro grupos son los machos y las hembras de *Gentoo* y los machos y las hembras de *Adelie* + *Chinstrap*. Las especies siguen siendo distinguibles (dentro de cada sexo, *Adelie* y *Chinstrap* quedan una al lado de la otra, más diferenciadas que en PCA), pero cada una queda partida en dos, y *Adelie* y *Chinstrap* del mismo sexo quedan más cerca entre sí que los machos y las hembras de una misma especie.

# %% [markdown]
# ## 4. t-SNE
#
# Esperamos que t-SNE muestre seis grupos, uno por especie y sexo, que con perplejidad muy baja aparezcan grupos fragmentados y que con perplejidad muy alta los grupos se acerquen. Se varía un parámetro por vez (base: 2 componentes, perplejidad 30 y 1000 iteraciones); la divergencia KL final figura en el título de cada gráfico.

# %%
# En scikit-learn actual el parámetro n_iter (usado en Unidad2.py) se
# llama max_iter. kl_divergence_ es el valor final de la divergencia
# KL, el mismo que informa verbose=1 en el ejemplo de clase.
PERPLEJIDADES = [5, 30, 50, 100]
ITERACIONES = [300, 500, 1000, 3000]

fig, axes = plt.subplots(2, 4, figsize=(19, 9))

for ax, p in zip(axes[0], PERPLEJIDADES):
    tsne = TSNE(n_components=2, perplexity=p, max_iter=1000,
                random_state=SEMILLA)
    tsne_2d = tsne.fit_transform(X_esc)
    sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y,
                    palette=PALETA_ESPECIES, s=14, alpha=0.8,
                    legend=(p == 5), ax=ax)
    ax.set(title=f"Perplejidad = {p}, KL = {tsne.kl_divergence_:.2f}",
           xlabel="t-SNE 1", ylabel="t-SNE 2")

for ax, it in zip(axes[1], ITERACIONES):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=it,
                random_state=SEMILLA)
    tsne_2d = tsne.fit_transform(X_esc)
    sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y,
                    palette=PALETA_ESPECIES, s=14, alpha=0.8,
                    legend=False, ax=ax)
    ax.set(title=f"Iteraciones = {it}, KL = {tsne.kl_divergence_:.2f}",
           xlabel="t-SNE 1", ylabel="t-SNE 2")

fig.suptitle("t-SNE con 2 componentes. Arriba: variación de la "
             "perplejidad (1000 iteraciones). Abajo: variación de las "
             "iteraciones (perplejidad 30)")
plt.tight_layout()
plt.show()

# Variación del número de componentes (perplejidad 30, 1000 iter.)
for c in [2, 3]:
    tsne = TSNE(n_components=c, perplexity=30, max_iter=1000,
                random_state=SEMILLA).fit(X_esc)
    print(f"{c} componentes: KL = {tsne.kl_divergence_:.2f}")

# %% [markdown]
# **Perplejidad:** con 5 cada especie se fragmenta en grupos pequeños; con 30 y 50 aparecen seis grupos claros, dos por especie; con 100, *Adelie* y *Chinstrap* se acercan. Coincide con lo esperado. La KL baja al aumentar la perplejidad (de 0.51 a 0.07), pero no sirve para comparar perplejidades distintas, así que la comparación es visual.
#
# **Iteraciones:** con 300 los grupos todavía están más dispersos; desde 500 la configuración se estabiliza y la KL casi no cambia (0.31, 0.23, 0.22 y 0.21).
#
# **Componentes:** con 3 componentes la KL es menor (0.13 contra 0.22), pero 2 alcanzan para visualizar la separación.

# %%
# La forma del marcador indica el sexo
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
            random_state=SEMILLA)
tsne_2d = tsne.fit_transform(X_esc)

plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y,
                style=datos["Sexo"], palette=PALETA_ESPECIES, s=35,
                alpha=0.8)
plt.title("t-SNE 2D (perplejidad = 30, 1000 iteraciones)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# %% [markdown]
# Con perplejidad 30 y 1000 iteraciones, t-SNE sigue separando las especies, pero no en tres grupos como en la versión principal sino en seis, uno por especie y sexo. Los machos *Adelie* y *Chinstrap* quedan bien separados; las hembras de esas dos especies forman grupos contiguos, con unas pocas *Chinstrap* pegadas al grupo de *Adelie*. Coincide con lo esperado.

# %% [markdown]
# ## 5. K-means
#
# Buscamos k = 3, uno por especie. Por la hipótesis de 1.8, esperamos que con k = 3 *Adelie* y *Chinstrap* se repartan por sexo en lugar de por especie, y que las métricas prefieran k = 6 (especie y sexo). Se prueba k entre 1 y 10 sobre los datos estandarizados.

# %%
# Inercia (curva del codo) y Silhouette para cada k, como en la U3
inercias = []
siluetas_kmeans = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    kmeans.fit(X_esc)
    inercias.append(kmeans.inertia_)
    if k > 1:  # Silhouette necesita al menos 2 clusters
        siluetas_kmeans.append(silhouette_score(X_esc, kmeans.labels_))


# GAP con OptimalK de la librería gap_statistic (Unidad3.py). Se le
# pasa el mismo KMeans de scikit-learn para que las tres métricas
# evalúen el mismo modelo.
def clusterer_kmeans(x, k):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(x)
    return modelo.cluster_centers_, modelo.labels_


gap_kmeans = OptimalK(n_jobs=1, clusterer=clusterer_kmeans,
                      random_state=SEMILLA)
k_gap_kmeans = int(gap_kmeans(X_esc, n_refs=50,
                              cluster_array=np.arange(1, 11)))
# La lista de Silhouette empieza en k = 2
k_silhouette_kmeans = int(np.argmax(siluetas_kmeans)) + 2

print("k óptimo según Silhouette:", k_silhouette_kmeans)
print("k óptimo según GAP:", k_gap_kmeans)

# gap_df es la tabla que arma OptimalK: gap_value es el valor de GAP
# para cada k y sk su desvío (las barras de error del gráfico)
pd.DataFrame({"k": range(1, 11),
              "Inercia": inercias,
              "Silhouette": [np.nan] + siluetas_kmeans,
              "GAP": gap_kmeans.gap_df["gap_value"],
              "Desvío GAP": gap_kmeans.gap_df["sk"]}).round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

axes[0].plot(range(1, 11), inercias, marker="o")
axes[0].set(title="Curva del codo",
            ylabel="Inercia (RSS dentro de los grupos)")

axes[1].plot(range(2, 11), siluetas_kmeans, marker="o")
axes[1].axvline(k_silhouette_kmeans, color="gray", linestyle="--",
                label=f"Máximo: k = {k_silhouette_kmeans}")
axes[1].set(title="Score de Silhouette", ylabel="Silhouette")
axes[1].legend()

axes[2].errorbar(range(1, 11), gap_kmeans.gap_df["gap_value"],
                 yerr=gap_kmeans.gap_df["sk"], marker="o", capsize=3)
axes[2].axvline(k_gap_kmeans, color="gray", linestyle="--",
                label=f"Óptimo GAP: k = {k_gap_kmeans}")
axes[2].set(title="Estadístico GAP", ylabel="GAP")
axes[2].legend()

for ax in axes:
    ax.set(xlabel="Número de clusters (k)", xticks=range(1, 11))
fig.suptitle("K-means: selección del número de clusters")
plt.tight_layout()
plt.show()

# %% [markdown]
# - **Codo:** la inercia cae fuerte hasta k = 2 y sigue bajando hasta k = 6; después queda casi plana.
# - **Silhouette:** máximo en **k = 6** (0.535); k = 5 queda segundo (0.527).
# - **GAP:** crece hasta k = 6 y después forma una meseta; `OptimalK` da **k = 6**.
#
# Las tres métricas coinciden en k = 6, como esperábamos: con `Sexo`, los grupos más compactos son las combinaciones de especie y sexo, no las especies. Se comparan los clusters de k = 2, 3 y 6 con las especies:

# %%
for k in [2, 3, 6]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = kmeans.fit_predict(X_esc)
    print(f"k = {k}")
    display(pd.crosstab(y, etiquetas, rownames=["Especie"],
                        colnames=["Cluster"]))

# %% [markdown]
# - **k = 2:** separa perfectamente a *Gentoo* del resto.
# - **k = 3:** *Gentoo* queda en un cluster, pero *Adelie* y *Chinstrap* se reparten por mitades entre los otros dos (71 y 72 *Adelie*, 33 y 33 *Chinstrap*): ningún cluster corresponde a *Chinstrap*.
# - **k = 6:** cada especie se reparte en dos clusters.
#
# Se verifica si esos repartos siguen al sexo:

# %%
for k in [3, 6]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = kmeans.fit_predict(X_esc)
    print(f"k = {k} vs. especie y sexo")
    display(pd.crosstab([y, datos["Sexo"]], etiquetas,
                        rownames=["Especie", "Sexo"], colnames=["Cluster"]))

# %% [markdown]
# Se confirma en los dos casos:
#
# - **k = 3:** los clusters son *Gentoo*, **todos los machos** de *Adelie* y *Chinstrap*, y **todas las hembras** de esas dos especies. Con `Sexo`, K-means separa por sexo en lugar de por especie: en la versión principal, con k = 3, la mayoría de los *Adelie* y *Chinstrap* quedaban en clusters distintos (la sección 7 lo compara).
# - **k = 6:** cada cluster es una combinación de especie y sexo, con solo 6 pingüinos fuera de su grupo: 5 hembras *Chinstrap* agrupadas con las hembras *Adelie* y 1 macho *Adelie* con los machos *Chinstrap*. Los clusters son casi puros en especie, pero cada especie queda partida en dos: para llegar a tres grupos habría que saber qué clusters unir, y eso requiere conocer la especie.

# %%
# Gráfico 3D con tres atributos, coloreado por cluster (como en
# Unidad3.py). Con Macho como eje, los pingüinos quedan en dos planos
# (0 = hembra, 1 = macho). Las estrellas son los centroides, llevados a
# las unidades originales con inverse_transform.
col_x = "Longitud Culmen (mm)"
col_y = "Masa corporal (g)"
col_z = "Macho"

fig = plt.figure(figsize=(20, 6.5))
for i, k in enumerate([2, 3, 6], start=1):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    kmeans.fit(X_esc)
    centroides = pd.DataFrame(
        escalador.inverse_transform(kmeans.cluster_centers_),
        columns=caracteristicas)
    colores = sns.color_palette("husl", k)

    ax = fig.add_subplot(1, 3, i, projection="3d")
    for c in range(k):
        puntos = X[kmeans.labels_ == c]
        ax.scatter(puntos[col_x], puntos[col_y], puntos[col_z], s=15,
                   alpha=0.6, color=colores[c], label=f"Cluster {c}")
        ax.scatter(centroides.loc[c, col_x], centroides.loc[c, col_y],
                   centroides.loc[c, col_z], marker="*", s=350,
                   color=colores[c], edgecolor="black")
    ax.set(xlabel=col_x, ylabel=col_y, zlabel=col_z, zticks=[0, 1],
           title=f"K-means con k = {k}")
    ax.view_init(elev=20, azim=-60)  # ángulo en el que mejor se ven
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# `Macho` divide los puntos en dos planos: arriba los machos y abajo las hembras. Con k = 2 los dos clusters ocupan ambos planos: *Gentoo* (mayor masa) y el resto. Con k = 3, los clusters de *Adelie* + *Chinstrap* quedan cada uno en un plano, mientras que el de *Gentoo* ocupa los dos (su centroide queda a mitad de altura): lo que separa a esos dos clusters es el sexo, no la longitud del culmen, que es lo que distingue a *Adelie* de *Chinstrap*. Con k = 6 hay tres grupos en cada plano: *Adelie* (culmen corto), *Chinstrap* (culmen largo y poca masa) y *Gentoo* (mayor masa).

# %% [markdown]
# ## 6. Clustering jerárquico
#
# Se usa el enlace de Ward, como en clase. Por lo visto con K-means, esperamos que el dendrograma separe primero a *Gentoo* y después divida cada grupo por sexo antes que por especie.

# %%
enlace = sch.linkage(X_esc, method="ward")

# Dendrograma truncado a las últimas 20 uniones, como en la U3.
# La línea marca un corte que deja 6 clusters.
plt.figure(figsize=(12, 4.5))
sch.dendrogram(enlace, truncate_mode="lastp", p=20, show_leaf_counts=True,
               show_contracted=True, color_threshold=9)
plt.axhline(y=9, color="k", linestyle="--", label="Corte en 6 clusters")
plt.title("Dendrograma (enlace de Ward)")
plt.xlabel("Número de pingüinos en el nodo")
plt.ylabel("Distancia")
plt.legend()
plt.show()

# %% [markdown]
# Coincide con lo esperado. La primera división (≈ 39) separa 118 pingüinos (*Gentoo*) del resto; la segunda (≈ 24) divide a *Adelie* + *Chinstrap* en 105 y 104 (hembras y machos, ver tablas más abajo), y la tercera (≈ 19) divide a *Gentoo* en 58 y 60 (hembras y machos). Recién después se separan las especies dentro de cada sexo (≈ 14 en los machos y ≈ 12 en las hembras). Después del salto de 1 a 2 clusters, el rango de alturas más amplio es el de 6 clusters (entre ≈ 5 y ≈ 12), así que el dendrograma sugiere **2 o 6 clusters**.

# %%
# Silhouette para cada k, como calculate_silhouette de la U3
siluetas_jerarquico = []
for k in range(2, 11):
    jerarquico = AgglomerativeClustering(n_clusters=k, linkage="ward")
    etiquetas = jerarquico.fit_predict(X_esc)
    siluetas_jerarquico.append(silhouette_score(X_esc, etiquetas))


# GAP con OptimalK usando el clustering jerárquico. Los centroides son
# la media de cada cluster, igual que en la función
# calculate_intra_cluster_dispersion de U3_Clustering.ipynb
def clusterer_jerarquico(x, k):
    jerarquico = AgglomerativeClustering(n_clusters=k, linkage="ward")
    etiquetas = jerarquico.fit_predict(x)
    centroides = np.array([x[etiquetas == c].mean(axis=0)
                           for c in range(k)])
    return centroides, etiquetas


gap_jerarquico = OptimalK(n_jobs=1, clusterer=clusterer_jerarquico,
                          random_state=SEMILLA)
k_gap_jerarquico = int(gap_jerarquico(X_esc, n_refs=50,
                                      cluster_array=np.arange(1, 11)))
# La lista de Silhouette empieza en k = 2
k_silhouette_jerarquico = int(np.argmax(siluetas_jerarquico)) + 2

print("k óptimo según Silhouette:", k_silhouette_jerarquico)
print("k óptimo según GAP:", k_gap_jerarquico)
pd.DataFrame({"k": range(1, 11),
              "Silhouette": [np.nan] + siluetas_jerarquico,
              "GAP": gap_jerarquico.gap_df["gap_value"],
              "Desvío GAP": gap_jerarquico.gap_df["sk"]}).round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(range(2, 11), siluetas_jerarquico, marker="o")
axes[0].axvline(k_silhouette_jerarquico, color="gray", linestyle="--",
                label=f"Máximo: k = {k_silhouette_jerarquico}")
axes[0].set(title="Score de Silhouette", ylabel="Silhouette")
axes[0].legend()

axes[1].errorbar(range(1, 11), gap_jerarquico.gap_df["gap_value"],
                 yerr=gap_jerarquico.gap_df["sk"], marker="o", capsize=3)
axes[1].axvline(k_gap_jerarquico, color="gray", linestyle="--",
                label=f"Óptimo GAP: k = {k_gap_jerarquico}")
axes[1].set(title="Estadístico GAP", ylabel="GAP")
axes[1].legend()

for ax in axes:
    ax.set(xlabel="Número de clusters (k)", xticks=range(1, 11))
fig.suptitle("Clustering jerárquico: selección del número de clusters")
plt.tight_layout()
plt.show()

# %% [markdown]
# - **Silhouette:** máximo en **k = 6** (0.532); k = 5 queda segundo (0.525).
# - **GAP:** crece hasta k = 6 y después forma una meseta; `OptimalK` da **k = 6**.

# %%
jerarquico = AgglomerativeClustering(n_clusters=2, linkage="ward")
etiquetas = jerarquico.fit_predict(X_esc)
print("k = 2")
display(pd.crosstab(y, etiquetas, rownames=["Especie"],
                    colnames=["Cluster"]))

for k in [3, 6]:
    jerarquico = AgglomerativeClustering(n_clusters=k, linkage="ward")
    etiquetas = jerarquico.fit_predict(X_esc)
    print(f"k = {k} vs. especie y sexo")
    display(pd.crosstab([y, datos["Sexo"]], etiquetas,
                        rownames=["Especie", "Sexo"], colnames=["Cluster"]))

# %% [markdown]
# - **k = 2:** separa a *Gentoo* del resto.
# - **k = 3:** igual que K-means: *Gentoo*, los machos de *Adelie* y *Chinstrap*, y las hembras de esas dos especies.
# - **k = 6:** cada cluster es una combinación de especie y sexo, salvo 7 hembras *Chinstrap* agrupadas con las hembras *Adelie*.
#
# **Número que mejor representa los datos:** Silhouette, GAP y el dendrograma coinciden en **k = 6**, un cluster por cada combinación de especie y sexo. Pero para el objetivo, separar las especies, ningún k da tres grupos que sean las especies: con k = 3 los clusters de *Adelie* y *Chinstrap* siguen al sexo, y con k = 6 cada especie queda partida en dos.

# %% [markdown]
# ## 7. Efecto de `Sexo` en la separación por especie
#
# Para medir el efecto, se repite el clustering sobre los mismos 327 pingüinos sin la columna `Macho` (como en la versión principal) y se cuenta, en cada cluster, cuántos pingüinos no son de la especie mayoritaria del cluster. Esperamos más errores con `Sexo` cuando k = 3.

# %%
# Mismos 327 pingüinos, sin la columna Macho
X_esc_sin_sexo = StandardScaler().fit_transform(datos[columnas_numericas])


def errores_especie(etiquetas):
    # Pingüinos que no son de la especie mayoritaria de su cluster
    tabla = pd.crosstab(etiquetas, y)
    return int((tabla.sum(axis=1) - tabla.max(axis=1)).sum())


comparacion = {}
for k in [3, 6]:
    modelos = {"K-means": KMeans(n_clusters=k, n_init=10,
                                 random_state=SEMILLA),
               "Jerárquico": AgglomerativeClustering(n_clusters=k,
                                                     linkage="ward")}
    for nombre, modelo in modelos.items():
        comparacion[f"{nombre}, k = {k}"] = {
            "Sin Sexo": errores_especie(modelo.fit_predict(X_esc_sin_sexo)),
            "Con Sexo": errores_especie(modelo.fit_predict(X_esc))}

pd.DataFrame(comparacion).T

# %% [markdown]
# Se confirma. Con k = 3, que es el número de especies, incluir `Sexo` empeora mucho la separación: los errores pasan de 24 a 66 en K-means y de 10 a 66 en el jerárquico. Los 66 son todos los *Chinstrap*: ningún cluster queda con mayoría de *Chinstrap*. Sin `Sexo` los resultados son casi los de la versión principal (26 y 10), que además incluye a los 9 pingüinos sin sexo.
#
# Con k = 6 los errores son parecidos con y sin `Sexo` (6 y 6 en K-means, 10 y 7 en el jerárquico), porque en los dos casos cada especie queda repartida en dos clusters. `Sexo` no borra la información de especie, pero con k = 3 se impone sobre ella.

# %% [markdown]
# ## 8. Conclusiones
#
# 1. **Datos:** se trabajó con 327 pingüinos, luego de eliminar 3 duplicados, 2 registros vacíos, 6 atípicos y los 9 pingüinos sin sexo. `Sexo` se codificó como 0/1 (`Macho`) y se estandarizó junto con las cuatro medidas.
# 2. **Estructura:** *Gentoo* es claramente distinta (aleta más larga, mayor masa y culmen menos profundo), mientras que *Adelie* y *Chinstrap* solo se distinguen por la longitud del culmen. Dentro de cada especie los machos son más grandes que las hembras, y entre *Adelie* y *Chinstrap* esa diferencia es mayor o comparable a la de especie en profundidad del culmen, aleta y masa.
# 3. **Reducción de la dimensionalidad:** incluir `Sexo` empeora la separación entre *Adelie* y *Chinstrap*. En PCA, `Macho` ocupa la segunda componente y desplaza a la tercera la longitud del culmen, que es la que separa a estas dos especies; con 2 componentes se superponen más que en la versión principal. En Isomap (60 vecinos) y t-SNE (perplejidad 30) las especies siguen siendo distinguibles, pero cada una aparece partida en dos grupos por sexo.
# 4. **Clustering:** Silhouette y GAP eligen k = 6 en los dos métodos, porque con `Sexo` los grupos más compactos son las combinaciones de especie y sexo. Con k = 3, los dos métodos agrupan a *Adelie* y *Chinstrap* por sexo y ningún cluster queda con mayoría de *Chinstrap*.
# 5. **Efecto de `Sexo` en el objetivo:** sobre los mismos 327 pingüinos y con k = 3, incluir `Sexo` hace que los errores de especie pasen de 24 a 66 en K-means y de 10 a 66 en el jerárquico. Se confirma la hipótesis y la decisión de la versión principal: para separar especies conviene excluir `Sexo`, porque agrega una división (machos y hembras) que compite con la de especie y, entre *Adelie* y *Chinstrap*, es más marcada que ella.
