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
# # Trabajo Práctico N° 1 — Minería de Datos
#
# | | |
# |---|---|
# | **Año** | 2026 |
# | **Materia** | Minería de Datos — Tecnicatura Universitaria en Inteligencia Artificial |
# | **Integrantes** | Josías Calabozo · Ismael Darruiz · Sebastian Di Carlo |
#
# **Objetivo:** integrar los contenidos de las unidades 2 (reducción de la dimensionalidad) y 3 (aprendizaje no supervisado) sobre el conjunto de datos de pingüinos del archipiélago Palmer (`penguins_size.csv`). La variable **Especie** es nuestro objetivo.
#
# **Contenido**
# 1. Análisis exploratorio y preparación de los datos
# 2. PCA
# 3. Isomap
# 4. t-SNE
# 5. K-means
# 6. Clustering jerárquico
# 7. Conclusiones

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
from IPython.display import display

# Aviso interno de scipy que aparece cuando Isomap completa un grafo de
# vecinos desconectado (ese caso se analiza en la sección 3)
warnings.filterwarnings("ignore", message="Changing the sparsity structure")


# Los avisos se muestran sin la ruta del archivo de la librería que los
# genera
def formato_aviso(mensaje, categoria, *args):
    return f"{categoria.__name__}: {mensaje}\n"


warnings.formatwarning = formato_aviso

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
# *Gentoo* se diferencia de las otras dos especies en la aleta (entre 21 y 27 mm más larga), la masa (entre 1.3 y 1.4 kg más) y la profundidad del culmen (más de 3 mm menos). *Adelie* y *Chinstrap* son casi iguales en profundidad del culmen y masa, difieren poco en la aleta (unos 6 mm, menos de un desvío estándar) y se distinguen sobre todo por la longitud del culmen, unos 10 mm mayor en *Chinstrap*.

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
# El pairplot confirma lo que anticipaba la tabla. *Gentoo* aparece separada, o apenas en contacto con el resto, en todos los gráficos con profundidad del culmen, aleta o masa. La primera columna (longitud del culmen contra cada una de las otras variables) muestra tres nubes separadas; en los demás gráficos *Adelie* y *Chinstrap* se superponen. Algunas distribuciones tienen dos picos (por ejemplo, la masa de *Gentoo*), lo que sugiere subgrupos dentro de cada especie; en 1.8 se verifica si se deben al sexo.

# %% [markdown]
# ### 1.6 Valores atípicos (outliers)
#
# Como las especies tienen promedios muy distintos, un valor normal para una especie puede ser extremo para otra. Por eso se buscan atípicos en todo el conjunto y también dentro de cada especie.

# %%
# Función de U3_Clustering.ipynb (sección "Remoción de Outliers")
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
# No hay atípicos en el conjunto completo, pero sí algunos dentro de cada especie (por ejemplo, un *Gentoo* con culmen de 59.6 mm o un *Chinstrap* de 2700 g). Son medidas posibles, no errores de carga, pero extremas para su especie. Se eliminan porque son pocos (6 de 342, menos del 2%) y porque PCA y K-means, que trabajan con varianzas y medias, son sensibles a los atípicos.
#
# Para encontrarlos se usó la especie, un dato que en un problema no supervisado real no estaría disponible. Por eso, al comparar los clusters con las especies, los resultados pueden ser algo más favorables que sin esta limpieza.

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
# La correlación entre aleta y masa es, en efecto, alta (0.87), y ambas se relacionan con la longitud del culmen (0.65 y 0.59). La profundidad del culmen es negativa con el resto por efecto de *Gentoo* (aleta larga, mayor masa y culmen poco profundo); dentro de cada especie, en cambio, la relación es positiva (en el pairplot, la nube de cada especie sube en los gráficos con profundidad del culmen). Esta redundancia sugiere que PCA podrá resumir los datos en pocas componentes.

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
# El reparto es casi exactamente 50/50. Entre los pingüinos de sexo conocido hay 163 hembras y 164 machos, y en cada especie la diferencia entre ambos sexos es de 0 a 2 individuos (entre 49% y 51% de cada sexo). Es decir, saber el sexo de un pingüino no dice nada sobre su especie.
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
# Los machos son más grandes, como suponíamos: en las tres especies superan a las hembras en las cuatro medidas. La diferencia más marcada es la masa (entre 360 y 800 g); en *Gentoo* los picos de machos y hembras están separados por unos 800 g y se superponen poco, y esa es la causa de los dos picos del pairplot.
#
# La tabla muestra algo importante para el resto del trabajo. Entre *Adelie* y *Chinstrap*, la diferencia por sexo es **mayor que la diferencia por especie** en la profundidad del culmen (alrededor de 1.5 mm contra 0.1 mm) y en la masa (360 a 670 g contra 34 g), y comparable en la aleta (4 a 8 mm contra 6 mm). Solo la longitud del culmen separa más a las especies (10 mm) que a los sexos (3 a 4.5 mm). Por lo tanto, al agrupar por distancias es posible que *Adelie* y *Chinstrap* se dividan por tamaño (sexo) antes que por especie; esto se revisa en las secciones 5 y 6.
#
# #### ¿Por qué no se usa `Sexo` como característica?
#
# `Sexo` se **elimina de las características**, por tres motivos:
# 1. No aporta información sobre la especie (reparto cercano a 50/50 en las tres).
# 2. Para usarla habría que codificarla como 0/1. Al estandarizar, esa columna pesaría lo mismo que cada medida y, a diferencia de ellas, dividiría los datos en dos grupos perfectamente separados, así que los métodos tenderían a agrupar por sexo en lugar de por especie.
# 3. Obligaría a imputar o eliminar los 9 pingüinos sin sexo.
#
# Sí se conserva en `datos` para **interpretar** los resultados: como el sexo modifica las medidas, sirve para verificar si los subgrupos que encuentren los métodos se deben a él.
#
# #### Faltantes de `Sexo`
#
# No se imputan. Con un reparto 50/50, la moda (global o por especie, como en la U2) se decide por uno o dos individuos, y la categoría "Desconocido" crea un valor que no existe en el fenómeno estudiado. Como `Sexo` no entra en ningún método, sus 9 faltantes no afectan a PCA, Isomap, t-SNE ni al clustering: esos pingüinos se usan en todos los análisis y solo quedan fuera de las tablas que cruzan clusters con sexo.

# %% [markdown]
# ### 1.9 Estandarización
#
# `Especie` se guarda en `y` y solo se usa para colorear los gráficos y comparar resultados. Las cuatro medidas se estandarizan.

# %%
X = datos[columnas_numericas]
y = datos["Especie"]

escalador = StandardScaler()
X_esc = escalador.fit_transform(X)

# %% [markdown]
# ## 2. PCA
#
# Por las correlaciones altas, esperamos que 2 componentes expliquen la mayor parte de la varianza: una de tamaño (donde se separa *Gentoo*) y otra del culmen (donde se separan *Adelie* y *Chinstrap*).

# %%
pca = PCA()
pca_componentes = pca.fit_transform(X_esc)
var_exp = pca.explained_variance_ratio_
var_cum = np.cumsum(var_exp)

pd.DataFrame({"Autovalor": pca.explained_variance_,
              "Varianza explicada (%)": 100 * var_exp,
              "Varianza acumulada (%)": 100 * var_cum},
             index=["PC1", "PC2", "PC3", "PC4"]).round(2)

# %% [markdown]
# Se aplican los tres criterios de la U2: varianza acumulada (umbral del 80%), Kaiser y codo.

# %%
componentes = [1, 2, 3, 4]
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
      "componente")

# %% [markdown]
# Como esperábamos, varianza acumulada y codo indican **2 componentes** (88.1% de la varianza): en el gráfico del codo, los autovalores caen fuerte de PC1 a PC2 y después casi se aplanan. Kaiser indica 1, porque solo PC1 tiene autovalor mayor a 1. Elegimos el criterio de **varianza acumulada**, que coincide con el codo y permite graficar en 2D. Kaiser descartaría PC2, que aporta casi el 19% de la varianza y, como se ve más abajo, es la que separa a *Adelie* de *Chinstrap*.

# %%
# Cargas de las dos primeras componentes
pd.DataFrame(pca.components_[:2].T, index=columnas_numericas,
             columns=["PC1", "PC2"]).round(2)

# %% [markdown]
# Coincide en parte con lo esperado. PC1 combina aleta, masa y longitud del culmen, con signo opuesto a la profundidad del culmen: sus valores altos corresponden a aleta larga, mayor masa y culmen poco profundo, el perfil de *Gentoo*. No es un eje de tamaño corporal: dentro de cada especie la profundidad del culmen crece junto con las otras medidas (1.7), así que un eje de tamaño tendría todas las cargas del mismo signo. PC1 es el eje que **separa a *Gentoo* del resto**. PC2 depende casi solo de las dos medidas del culmen: representa el **tamaño del pico**.

# %%
plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(x=pca_componentes[:, 0], y=pca_componentes[:, 1], hue=y,
                palette=PALETA_ESPECIES, s=35, alpha=0.8)
plt.title("PCA: proyección sobre las dos primeras componentes")
plt.xlabel(f"PC1 ({var_exp[0]:.1%} de la varianza)")
plt.ylabel(f"PC2 ({var_exp[1]:.1%} de la varianza)")
plt.show()

# %% [markdown]
# *Gentoo* queda separada sobre PC1, y *Adelie* y *Chinstrap* se separan sobre PC2, aunque con una zona de superposición. Coincide con lo visto en el pairplot.

# %% [markdown]
# ## 3. Isomap
#
# Esperamos que con pocos vecinos los grupos queden más separados y que con muchos el resultado se parezca al de PCA. Como *Gentoo* está muy alejada del resto, con pocos vecinos el grafo podría quedar dividido, algo que las diapositivas de la U2 señalan como un problema. Primero se prueban 5, 30, 60 y 100 vecinos con 2 componentes, y después una tercera componente con el número de vecinos elegido.

# %%
VECINOS = [5, 30, 60, 100]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, k in zip(axes, VECINOS):
    isomap_2d = Isomap(n_neighbors=k, n_components=2).fit_transform(X_esc)
    sns.scatterplot(x=isomap_2d[:, 0], y=isomap_2d[:, 1], hue=y,
                    palette=PALETA_ESPECIES, s=18, alpha=0.8,
                    legend=(k == 5), ax=ax)
    ax.set(title=f"n_neighbors = {k}", xlabel="Componente 1",
           ylabel="Componente 2")
fig.suptitle("Isomap con 2 componentes variando el número de vecinos")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con 5 y 30 vecinos el grafo de vecinos queda dividido en dos partes, como anticipábamos (Isomap lo avisa debajo del código), y *Gentoo*, que está muy alejada del resto, queda aplastada sobre una línea: es la deformación típica de un grafo desconectado. Con 60 vecinos el aviso desaparece, *Gentoo* queda menos aplastada y las tres especies quedan diferenciadas. Con 100 el resultado es casi igual al de PCA, también según lo previsto.
#
# Elegimos **60 vecinos**: es el menor valor probado con el grafo conectado, y con más vecinos (100) el resultado se parece al de PCA y no aporta nada que PCA no muestre. *Gentoo* queda bien separada, y *Adelie* y *Chinstrap* quedan una al lado de la otra, con algo de mezcla en la zona de contacto.
#
# Con 60 vecinos se prueba una tercera componente. Esperamos que aporte poco, como en PCA, donde PC3 explica el 9.2% de la varianza. Se grafica cada par de componentes; las dos primeras son las mismas que las del Isomap con 2 componentes, así que el primer gráfico es el Isomap 2D elegido.

# %%
isomap_3d = Isomap(n_neighbors=60, n_components=3).fit_transform(X_esc)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    sns.scatterplot(x=isomap_3d[:, i], y=isomap_3d[:, j], hue=y,
                    palette=PALETA_ESPECIES, s=25, alpha=0.8,
                    legend=(j == 1), ax=ax)
    ax.set(title=f"Componente {i + 1} vs. componente {j + 1}",
           xlabel=f"Componente {i + 1}", ylabel=f"Componente {j + 1}")
fig.suptitle("Isomap con 3 componentes (n_neighbors = 60)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Como en PCA, la tercera componente aporta poco: no mejora la separación. *Gentoo* se separa por la componente 1, y *Adelie* y *Chinstrap* por la componente 2; en el gráfico de las componentes 1 y 3 estas dos especies se superponen más que en el de 1 y 2. Con 2 componentes alcanza.

# %% [markdown]
# ## 4. t-SNE
#
# Esperamos que t-SNE separe las especies mejor que PCA e Isomap, que con perplejidad muy baja aparezcan grupos fragmentados y que con perplejidad muy alta los grupos empiecen a fusionarse. Se varía un parámetro por vez (base: 2 componentes, perplejidad 30 y 1000 iteraciones); la divergencia KL final figura en el título de cada gráfico.

# %%
# max_iter es el nombre actual de n_iter (Unidad2.py)
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

# %% [markdown]
# **Perplejidad:** con 5 cada especie se fragmenta en grupos pequeños; con 30 y 50 aparecen tres grupos claros; con 100, *Adelie* y *Chinstrap* empiezan a fusionarse. Son los tres comportamientos previstos. La KL baja al aumentar la perplejidad (de 0.60 a 0.10), pero no sirve para comparar perplejidades distintas, así que la comparación es visual.
#
# **Iteraciones:** con 300 la configuración todavía no se estabilizó (los grupos tienen otra forma y la KL es mayor); desde 500 se estabiliza y la KL casi no cambia (0.45, 0.38, 0.38 y 0.37).
#
# Se elige perplejidad 30 y 1000 iteraciones, y con esos valores se comparan 2 y 3 componentes.

# %%
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
            random_state=SEMILLA)
tsne_2d = tsne.fit_transform(X_esc)
modelo_3d = TSNE(n_components=3, perplexity=30, max_iter=1000,
                 random_state=SEMILLA)
tsne_3d = modelo_3d.fit_transform(X_esc)

fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 2, 1)
sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y,
                palette=PALETA_ESPECIES, s=35, alpha=0.8, ax=ax)
ax.set(title=f"2 componentes, KL = {tsne.kl_divergence_:.2f}",
       xlabel="t-SNE 1", ylabel="t-SNE 2")

ax = fig.add_subplot(1, 2, 2, projection="3d")
for especie, color in PALETA_ESPECIES.items():
    puntos = tsne_3d[y == especie]
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], s=15, alpha=0.7,
               color=color, label=especie)
ax.set(title=f"3 componentes, KL = {modelo_3d.kl_divergence_:.2f}",
       xlabel="t-SNE 1", ylabel="t-SNE 2", zlabel="t-SNE 3")
ax.legend()
fig.suptitle("t-SNE con perplejidad 30 y 1000 iteraciones")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con perplejidad 30 y 1000 iteraciones, t-SNE confirma la hipótesis y logra la separación más clara de los tres métodos: las tres especies forman grupos separados, *Adelie* y *Chinstrap* son contiguas y unos pocos *Chinstrap* quedan pegados al grupo de *Adelie*. Las distancias entre los grupos no se interpretan, porque t-SNE no las conserva.
#
# **Componentes:** con 3 componentes la KL es menor (0.27 contra 0.38), pero los grupos son los mismos que en 2D. Con 2 componentes alcanza.

# %% [markdown]
# ## 5. K-means
#
# Si los grupos coinciden con las especies, el óptimo debería ser k = 3. Pero como *Adelie* y *Chinstrap* se superponen en tres variables, las métricas podrían preferir k = 2 (*Gentoo* contra el resto). Se prueba k entre 1 y 10 sobre los datos estandarizados.

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
# - **Codo:** la inercia cae fuerte hasta k = 2 y menos hasta k = 3; después baja de forma gradual.
# - **Silhouette:** máximo en **k = 2** (0.537); k = 3 queda segundo (0.453).
# - **GAP:** crece hasta k = 5 y se estabiliza; `OptimalK` da **k = 6**, aunque entre 5 y 10 los valores están dentro de las barras de error.
#
# Ninguna métrica indica k = 3. Se comparan los clusters de k = 2, 3 y 6 con las especies:

# %%
for k in [2, 3, 6]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = kmeans.fit_predict(X_esc)
    print(f"k = {k}")
    display(pd.crosstab(y, etiquetas, rownames=["Especie"],
                        colnames=["Cluster"]))

# %% [markdown]
# - **k = 2:** separa perfectamente a *Gentoo* del resto.
# - **k = 3:** *Chinstrap* no queda absorbida, pero se mezcla con *Adelie*: un cluster tiene 127 *Adelie* y 5 *Chinstrap*, y otro 61 *Chinstrap* y 21 *Adelie*.
# - **k = 6:** cada especie se reparte principalmente en dos clusters.
#
# Por lo visto en 1.8, sospechamos que tanto los errores de k = 3 como los subgrupos de k = 6 se deben al sexo, que no se usó para agrupar. Se verifica:

# %%
for k in [3, 6]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = kmeans.fit_predict(X_esc)
    print(f"k = {k} vs. especie y sexo "
          "(se omiten los de sexo desconocido)")
    display(pd.crosstab([y, datos["Sexo"]], etiquetas,
                        rownames=["Especie", "Sexo"], colnames=["Cluster"]))

# %% [markdown]
# Se confirma en los dos casos:
#
# - **k = 3:** los errores siguen al tamaño. De los 21 *Adelie* agrupados con los *Chinstrap*, 20 son machos (el otro no tiene sexo registrado), y los 5 *Chinstrap* agrupados con los *Adelie* son hembras. Es lo anticipado en 1.8: como en profundidad del culmen, aleta y masa la diferencia por sexo es mayor o comparable a la diferencia entre estas dos especies, los *Adelie* grandes quedan cerca de los *Chinstrap* y las *Chinstrap* chicas cerca de los *Adelie*.
# - **k = 6:** los subgrupos coinciden casi exactamente con el sexo (por ejemplo, 69 de las 72 hembras *Adelie* están en un cluster y 56 de los 71 machos en otro). Como en las tres especies los machos superan a las hembras en las cuatro medidas (entre 360 y 800 g más de masa), GAP encuentra más clusters que especies.

# %%
# Clusters sobre el t-SNE elegido en la sección 4 (tsne_2d, perplejidad
# 30 y 1000 iteraciones): el color es la especie y la forma, el cluster
MARCADORES = ["o", "X", "^", "s", "P", "D"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for ax, k in zip(axes, [2, 3, 6]):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    clusters = pd.Series(kmeans.fit_predict(X_esc), name="Cluster")
    sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y,
                    style=clusters, markers=MARCADORES[:k],
                    palette=PALETA_ESPECIES, s=45, alpha=0.8, ax=ax)
    ax.set(title=f"K-means con k = {k}", xlabel="t-SNE 1",
           ylabel="t-SNE 2")
    ax.legend(loc="upper center", fontsize=8)  # zona vacía del gráfico
fig.suptitle("Clusters de K-means sobre el t-SNE (color: especie, "
             "forma: cluster)")
plt.tight_layout()
plt.show()

# %% [markdown]
# En el t-SNE las especies forman tres grupos, así que se puede ver qué separa cada k:
#
# - **k = 2:** *Gentoo* tiene una forma, y *Adelie* y *Chinstrap* comparten la otra.
# - **k = 3:** casi todos los *Chinstrap* comparten forma, pero esa forma también aparece en la parte de arriba del grupo de *Adelie*, la más cercana a *Chinstrap*: son los 21 *Adelie* agrupados con los *Chinstrap*, casi todos machos. Los 5 *Chinstrap* agrupados con los *Adelie* son justamente los que t-SNE ubica pegados al grupo de *Adelie*.
# - **k = 6:** cada especie aparece con dos formas, que corresponden principalmente a hembras y machos.
#
# **Número de clusters:** según las métricas, el óptimo es k = 2 (Silhouette) o k = 6 (GAP), y ninguna indica k = 3. Cada uno refleja un nivel de la estructura de los datos: k = 2 separa a *Gentoo* del resto, y k = 6 divide además cada especie por sexo. Como el objetivo del trabajo es la especie, para el gráfico 3D usamos **k = 3**, aunque las métricas no lo elijan: separa a *Gentoo* sin errores, pero mezcla 26 pingüinos de *Adelie* y *Chinstrap*.
#
# Se grafican la longitud y la profundidad del culmen y la longitud de la aleta: la longitud del culmen es la que separa a *Adelie* de *Chinstrap*, la profundidad y la aleta separan a *Gentoo*, y la masa casi repite la información de la aleta (correlación 0.87). Cada cluster lleva el color de la especie que tiene mayoría en él, con la misma paleta que el resto del trabajo.

# %%
# Gráfico 3D con tres atributos originales, coloreado por cluster
# (como en Unidad3.py), para k = 3. Las estrellas son los centroides,
# llevados a las unidades originales con inverse_transform.
col_x = "Longitud Culmen (mm)"
col_y = "Profundidad Culmen (mm)"
col_z = "Longitud Aleta (mm)"

kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X_esc)
centroides = pd.DataFrame(
    escalador.inverse_transform(kmeans.cluster_centers_),
    columns=columnas_numericas)
# Especie mayoritaria de cada cluster, que define su color
especie = pd.crosstab(kmeans.labels_, y).idxmax(axis=1)

fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(projection="3d")
for c in range(3):
    puntos = X[kmeans.labels_ == c]
    color = PALETA_ESPECIES[especie[c]]
    ax.scatter(puntos[col_x], puntos[col_y], puntos[col_z], s=15,
               alpha=0.6, color=color,
               label=f"Cluster {c}: mayoría {especie[c]}")
    ax.scatter(centroides.loc[c, col_x], centroides.loc[c, col_y],
               centroides.loc[c, col_z], marker="*", s=350,
               color=color, edgecolor="black")
ax.set(xlabel=col_x, ylabel=col_y, zlabel=col_z, title="K-means con k = 3")
ax.view_init(elev=25, azim=120)  # ángulo en el que mejor se ven
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# Se separan los pingüinos de aleta larga y culmen poco profundo (*Gentoo*), pero la frontera entre los dos grupos de culmen profundo no coincide con la división entre *Adelie* y *Chinstrap*: el cluster de culmen largo incluye también a los *Adelie* más grandes.

# %% [markdown]
# ## 6. Clustering jerárquico
#
# Se usa el enlace de Ward, que, como K-means, busca grupos con poca suma de cuadrados interna; así los dos métodos se pueden comparar. Por lo visto con K-means, esperamos que el dendrograma separe primero a *Gentoo* y después a *Adelie* de *Chinstrap*.

# %%
enlace = sch.linkage(X_esc, method="ward")

# Dendrograma truncado a las últimas 20 uniones, como en la U3.
# La línea marca un corte que deja 3 clusters, cada uno con el color de
# su especie mayoritaria (en el orden en que aparecen: Gentoo,
# Chinstrap y Adelie).
sch.set_link_color_palette([PALETA_ESPECIES["Gentoo"],
                            PALETA_ESPECIES["Chinstrap"],
                            PALETA_ESPECIES["Adelie"]])
plt.figure(figsize=(12, 4.5))
sch.dendrogram(enlace, truncate_mode="lastp", p=20, show_leaf_counts=True,
               show_contracted=True, color_threshold=15,
               above_threshold_color="gray")
sch.set_link_color_palette(None)  # vuelve a los colores por defecto
plt.axhline(y=15, color="k", linestyle="--", label="Corte en 3 clusters")
# Líneas vacías: solo agregan cada color de especie a la leyenda
for especie, color in PALETA_ESPECIES.items():
    plt.plot([], [], color=color, label=f"Mayoría {especie}")
plt.title("Dendrograma (enlace de Ward)")
plt.xlabel("Número de pingüinos en el nodo")
plt.ylabel("Distancia")
plt.legend()
plt.show()

# %% [markdown]
# El orden de las divisiones es el que anticipaba K-means: la primera (≈ 40) separa 122 pingüinos (*Gentoo*) y la segunda (≈ 19) divide el resto en 56 (*Chinstrap*) y 158 (*Adelie*, ver tablas más abajo). El tramo vertical más largo sin uniones está entre ≈ 19 y ≈ 40, así que el dendrograma sugiere sobre todo **2 clusters**. El tramo que deja 3 clusters (entre ≈ 12 y ≈ 19) es mucho más corto y apenas más largo que el que deja 5 (entre ≈ 6 y ≈ 12).

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
# - **Silhouette:** máximo en **k = 2** (0.537); k = 3 queda segundo (0.465, algo mejor que en K-means).
# - **GAP:** crece hasta k = 5 y después forma una meseta; `OptimalK` da **k = 5**.

# %%
jerarquico = AgglomerativeClustering(n_clusters=2, linkage="ward")
etiquetas = jerarquico.fit_predict(X_esc)
print("k = 2")
display(pd.crosstab(y, etiquetas, rownames=["Especie"],
                    colnames=["Cluster"]))

for k in [3, 5]:
    jerarquico = AgglomerativeClustering(n_clusters=k, linkage="ward")
    etiquetas = jerarquico.fit_predict(X_esc)
    print(f"k = {k} vs. especie y sexo "
          "(se omiten los de sexo desconocido)")
    display(pd.crosstab([y, datos["Sexo"]], etiquetas,
                        rownames=["Especie", "Sexo"], colnames=["Cluster"]))

# %% [markdown]
# - **k = 2:** separa a *Gentoo* del resto.
# - **k = 3:** reproduce las especies mucho mejor que K-means: todos los *Adelie* y *Gentoo* quedan en su cluster, y solo 10 *Chinstrap* quedan agrupados con los *Adelie*. Los 10 son hembras, las *Chinstrap* más chicas, igual que en K-means.
# - **k = 5:** *Adelie* y *Gentoo* se dividen principalmente por sexo (por ejemplo, los 60 machos *Gentoo* en un cluster y 49 de las 58 hembras en otro); *Chinstrap* queda mayormente en un solo cluster (56 de 66).

# %%
# Gráfico 3D con los mismos atributos, ángulo y colores que en K-means,
# para k = 3. Las estrellas son la media de cada cluster (el jerárquico
# no calcula centroides).
jerarquico = AgglomerativeClustering(n_clusters=3, linkage="ward")
etiquetas = jerarquico.fit_predict(X_esc)
especie = pd.crosstab(etiquetas, y).idxmax(axis=1)

fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(projection="3d")
for c in range(3):
    puntos = X[etiquetas == c]
    media = puntos.mean()
    color = PALETA_ESPECIES[especie[c]]
    ax.scatter(puntos[col_x], puntos[col_y], puntos[col_z], s=15,
               alpha=0.6, color=color,
               label=f"Cluster {c}: mayoría {especie[c]}")
    ax.scatter(media[col_x], media[col_y], media[col_z], marker="*",
               s=350, color=color, edgecolor="black")
ax.set(xlabel=col_x, ylabel=col_y, zlabel=col_z,
       title="Jerárquico con k = 3")
ax.view_init(elev=25, azim=120)
ax.legend(loc="upper left", fontsize=8)
plt.show()

# %% [markdown]
# Frente al 3D de K-means, la diferencia está en la frontera entre los dos grupos de culmen profundo. En el jerárquico, el cluster de culmen largo (mayoría *Chinstrap*) tiene solo *Chinstrap* (56 pingüinos); en K-means tenía 82, porque incluía a los *Adelie* más grandes. Por eso el jerárquico comete menos errores.
#
# **Número que mejor representa los datos:** según las métricas, k = 2 (Silhouette y el tramo más largo del dendrograma) o k = 5 (GAP); ninguna indica k = 3. Como en K-means, cada número refleja un nivel de la estructura: k = 2 separa a *Gentoo* del resto, y k = 5 divide además *Adelie* y *Gentoo* por sexo. Por el objetivo del trabajo elegimos **k = 3**, que reproduce las especies con solo 10 pingüinos mal agrupados. Esa elección no surge de las métricas sino de lo que se busca encontrar.

# %% [markdown]
# ## 7. Conclusiones
#
# 1. **Datos:** se trabajó con 336 pingüinos, luego de eliminar 3 duplicados, 2 registros vacíos y 6 atípicos dentro de su especie (para encontrarlos se usó la especie, que en un problema no supervisado real no estaría disponible). Las cuatro medidas se estandarizaron. `Sexo` se excluyó de las características: no aporta información sobre la especie (reparto cercano a 50/50 en las tres) y, al incluirla, los métodos tenderían a agrupar por sexo en lugar de por especie. Sus 9 faltantes no se imputaron, porque con un reparto 50/50 la moda es arbitraria. Como la variable no entra en ningún método, se dejaron como faltantes sin perder esos pingüinos.
# 2. **Estructura:** *Gentoo* es claramente distinta (aleta más larga, mayor masa y culmen menos profundo), mientras que *Adelie* y *Chinstrap* se distinguen sobre todo por la longitud del culmen. Dentro de cada especie los machos son más grandes que las hembras en las cuatro medidas, y en profundidad del culmen, aleta y masa esa diferencia es mayor o comparable a la que hay entre *Adelie* y *Chinstrap*. Por eso los errores de clustering entre estas dos especies son *Adelie* machos y *Chinstrap* hembras.
# 3. **Reducción de la dimensionalidad:** PCA con 2 componentes conserva el 88.1% de la varianza y se interpreta fácilmente, pero no separa del todo a *Adelie* de *Chinstrap*. Isomap depende mucho del número de vecinos (con pocos deforma a *Gentoo* y con muchos se parece a PCA); con 60 separa las tres especies, y una tercera componente no mejora la separación. t-SNE (perplejidad 30) logra la separación más clara; con 3 componentes la KL baja, pero los grupos no cambian.
# 4. **Clustering:** en los dos métodos, Silhouette elige k = 2 (*Gentoo* contra el resto) y GAP 5 o 6 (subgrupos por sexo); ninguna métrica elige k = 3. Usamos k = 3 por el objetivo del trabajo, y con ese valor el clustering jerárquico reproduce las especies mucho mejor que K-means (10 pingüinos mal agrupados contra 26).
# 5. **Hipótesis inicial:** los tres grupos esperados se confirman solo en parte. *Gentoo* se separa con cualquier método, pero la separación entre *Adelie* y *Chinstrap* depende del método y de sus parámetros. El número de clusters "óptimo" depende del criterio usado y del nivel de detalle que se busque.
