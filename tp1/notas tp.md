No se vio en clase: conviene sacarlo
Sección Qué hace el TP En el material de la cátedra Recomendación
1.8 Prueba chi-cuadrado entre sexo y especie No aparece Sacar
1.8 Gráficos de violín por especie y sexo No aparece Sacar
1.8 Tabla de diferencias macho − hembra contra Chinstrap − Adelie No aparece Sacar
1.8 K-means de prueba con Sexo codificado No aparece Sacar
1.8 Gráfico de los 9 pingüinos sin sexo y discusión de 4 formas de imputar En parte: la U2 ve imputar por moda o por grupo y descarta "Desconocido" Dejarlo en 2 o 3 líneas
3 kneighbors*graph + connected_components + tabla de las partes del grafo Las diapositivas solo advierten que con un k chico el grafo queda desconectado Sacar y describir lo que se ve en la grilla de gráficos
5 y 6 Tablas de clusters contra especie y sexo No aparece Dejar solo la comparación con especie
5 y 6 GAP con barras de error (errorbar, sk) En la U3 solo se grafica la curva Graficar solo gap_value con plot
5 Tres gráficos 3D (k = 2, 3 y 6) con inverse_transform El 3D con los centroides como estrellas sí está (Unidad3.py); los tres paneles y inverse_transform no Un solo gráfico sobre los datos estandarizados, como en el notebook U3
No aparece tal cual, pero conviene mantenerlo
Tablas de clusters contra especie (pd.crosstab): el código no está en el material. Pero es la forma más directa de resolver la tarea que deja la U3: "verificar en qué puntos no coinciden los clusters con las etiquetas verdaderas".
clusterer= en OptimalK (L615, L772): en clase OptimalK se usa con su K-means por defecto, incluso en el ejemplo de clustering jerárquico de las diapositivas. Recomiendo mantenerlo, porque sin eso el GAP "del jerárquico" en realidad evalúa K-means.
Tampoco conviene usar el GAP escrito a mano del notebook U3. En la versión para K-means, la función calculate_intra_cluster_dispersion siempre ajusta X_std e ignora el X que recibe. Así nunca usa los datos aleatorios de referencia y el resultado no significa nada.
reconstruction_error() de Isomap (L472): la idea está en la diapositiva "ISOMAP - Consideraciones prácticas", que propone evaluar comparando distancias. Cubre el pedido de "variar componentes" sin sumar gráficos, así que la mantendría.
Si la sacan, reemplácenla por un Isomap 3D como el del notebook U2 y quiten también eigen_solver="dense".
Pesos de las variables en las componentes de PCA (components*) (L424): el código no se vio. Las diapositivas sí dicen que las componentes se interpretan por lo que aporta cada variable. Es opcional.
groupby().agg() (L142) y drop*duplicates(): son pandas básico. Eliminar duplicados hace falta para limpiar los datos.
Sí está en clase
Análisis exploratorio: describe, isna().sum(), value_counts, el pairplot con hue, kde y corner (U2 y U3), la matriz de correlación (U1 a U3) y los boxplots por clase (U3).
Outliers: la función remove_outliers_iqr y la limpieza por especie son de la U3. El conteo sobre todo el conjunto es de la U2.
Dejar Sexo fuera de las características: la cátedra aplica PCA, Isomap, t-SNE y clustering solo sobre variables numéricas.
PCA: los tres criterios, el gráfico de barras con la varianza acumulada y sus etiquetas (U2), explained_variance* como autovalores (Unidad2.py) y el gráfico 2D.
Isomap: variar n*neighbors y graficar cada caso es justamente la consigna del notebook U2.
t-SNE: la perplejidad y las iteraciones están en las diapositivas de la U2. La divergencia KL es la que imprime el ejemplo de clase con verbose=1, y max_iter es el n_iter de clase con su nombre nuevo.
K-means: el codo con la inercia, Silhouette y OptimalK.
Clustering jerárquico: el método de Ward y el dendrograma truncado con línea de corte son iguales al notebook U3. También está Silhouette con AgglomerativeClustering y los centroides calculados como la media de cada cluster.
Otras cosas para tener en cuenta
Qué se pierde al sacar Sexo: es lo que explica por qué se mezclan Adelie y Chinstrap` (los mal agrupados son machos Adelie y hembras Chinstrap). Si quieren conservar esa conclusión con lo mínimo, dejen una sola tabla (K-means con k = 3 contra especie y sexo) y una frase. Además hay que ajustar los textos de 1.5, 5 y 6, las conclusiones 1, 2 y 4, y los imports.
Criterio de Kaiser: el notebook U2 lo define de dos formas distintas.
En la lista de criterios dice "autovalores > 1". Es la que usa el TP y da 1 componente.
Al aplicarlo usa "≥ 10% de la varianza explicada". Con estos datos daría 2 componentes, porque PC2 explica cerca del 19%.
La del TP es la estándar, pero conviene saberlo por si el docente espera la otra.
El enunciado pide no explicar los parámetros: varios comentarios del código lo hacen (reconstruction_error, eigen_solver, kl_divergence*, gap_df). Conviene acortarlos.
El enunciado pide pocos gráficos: el Isomap final (60 vecinos) y el t-SNE final (perplejidad 30) repiten un panel de la grilla anterior con los mismos parámetros.
Integrantes: el enunciado pide grupos de dos y la cabecera tiene tres nombres, por si no lo acordaron con la cátedra.
