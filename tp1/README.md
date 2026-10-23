# TP1 — Minería de Datos (2026)

Trabajo Práctico N° 1 de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo

Análisis del dataset de pingüinos de Palmer (`penguins_size.csv`): análisis exploratorio, reducción de dimensionalidad (PCA, Isomap, t-SNE) y clustering (K-means y jerárquico), usando la especie como variable objetivo.

El entorno y la forma de trabajar con los notebooks (jupytext) son comunes a todos los TP: están en el [README del repo](../README.md).

## Archivos

```
tp1/
├── data/
│   └── penguins_size.csv          # dataset provisto por la cátedra
├── docs/
│   └── TP1_enunciado.pdf          # consigna del trabajo
├── notebooks/
│   ├── TP1_Mineria_Datos.py       # informe versionado en git (formato jupytext)
│   ├── TP1_Mineria_Datos_con_sexo.py  # versión alternativa que usa Sexo como característica
│   └── *.ipynb                    # copias locales generadas con jupytext (no se suben)
├── notas tp.md                    # qué partes del TP se vieron en clase y cuáles conviene sacar
└── README.md
```

## Ejecución

Con el entorno activado y desde la raíz del repo, generar el notebook a partir del `.py`:

```bash
jupytext --sync tp1/notebooks/TP1_Mineria_Datos.py
```

Para la versión con `Sexo`, lo mismo con `tp1/notebooks/TP1_Mineria_Datos_con_sexo.py`.

Después, abrir `tp1/notebooks/TP1_Mineria_Datos.ipynb` con el kernel del `.venv` y ejecutar todas las celdas. El notebook lee los datos desde `../data/`, así que se debe abrir desde la carpeta `tp1/notebooks/`. La corrida completa tarda un par de minutos.

## Entrega

La consigna acepta el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, abrir `tp1/notebooks/TP1_Mineria_Datos.ipynb`, ejecutar **Restart & Run All** y subir ese archivo (con todas las salidas visibles) al campus.

Si se prefiere una versión para leer en el navegador:

```bash
jupyter nbconvert --to html tp1/notebooks/TP1_Mineria_Datos.ipynb
```
