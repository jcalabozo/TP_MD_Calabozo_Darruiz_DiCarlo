# TP3 — Minería de Datos (2026)

Trabajo Práctico N° 3 de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo

Clasificación multiclase con los modelos de las unidades 5 y 6: recomendar un cultivo (`clase`, 10 clases) a partir de 16 parámetros de suelo y clima (`crop_recommendation_geo.csv`, 50004 muestras), con SVM de kernel lineal, SVM de kernel gaussiano (RBF) y Random Forest, y comparar los tres. El notebook detecta y elimina una fila con valores faltantes, 198 filas duplicadas y dos filas con valores imposibles.

Todo se evalúa con validación cruzada estratificada de 5 folds sobre el conjunto limpio, con `random_state = 13`.

El entorno y la forma de trabajar con los notebooks (jupytext) son comunes a todos los TP: están en el [README del repo](../README.md).

## Archivos

```
tp3/
├── data/
│   └── crop_recommendation_geo.csv          dataset provisto por la cátedra
├── docs/
│   └── TP3_enunciado.pdf                    consigna de la cátedra
├── notebooks/
│   ├── TP3_calabozo_darruiz_dicarlo.py      notebook del TP, versionado en git (formato jupytext)
│   └── TP3_calabozo_darruiz_dicarlo.ipynb   notebook ejecutado, con sus salidas (también se sube)
└── README.md
```

## Ejecución

Con el entorno activado y desde la raíz del repo, generar el notebook a partir del `.py`:

```bash
jupytext --sync tp3/notebooks/TP3_calabozo_darruiz_dicarlo.py
```

Después, abrir `tp3/notebooks/TP3_calabozo_darruiz_dicarlo.ipynb` con el kernel **Python 3.12 (TP MD)** (se registra en la instalación, ver el [README del repo](../README.md)) y ejecutar todas las celdas. El notebook lee `../data/` con una ruta relativa a su carpeta: se ejecuta con `tp3/notebooks/` como carpeta de trabajo, que es lo que hacen Jupyter y VS Code por defecto.

Para ejecutarlo y guardar las salidas sin abrirlo:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 tp3/notebooks/TP3_calabozo_darruiz_dicarlo.ipynb
```

`--ExecutePreprocessor.timeout=-1` quita el límite de tiempo por celda, porque las búsquedas de hiperparámetros tardan más que el límite por defecto.

**Tiempo de ejecución.** Una corrida completa tarda alrededor de una hora en una PC con 12 núcleos (con menos núcleos, más), casi todo en las búsquedas de los SVM; solo el SVM lineal con C = 100 tarda unos 15 minutos por fold. Los resultados de los modelos son siempre los mismos (todo usa `random_state = 13`); solo cambian los tiempos de entrenamiento, que dependen de la PC. Si se corrige solo texto, no hace falta volver a ejecutar: `jupytext --sync` pasa los cambios al `.ipynb` y conserva las salidas. Si cambia el código, hay que ejecutar todo de nuevo y revisar que los números de los textos coincidan con las salidas.

## Entrega

La consigna pide el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, ejecutar todo si cambió el código (**Restart & Run All** o el `nbconvert` de arriba) y subir `tp3/notebooks/TP3_calabozo_darruiz_dicarlo.ipynb`, con todas las salidas visibles, al campus.
