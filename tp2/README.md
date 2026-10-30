# TP2 — Minería de Datos (2026)

Trabajo Práctico N° 2 de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo

Dos problemas con los modelos de las unidades 4 y 5:

- **Parte A, regresión:** estimar la ganancia (`Profit`) de 1000 empresas a partir de sus gastos con árboles de decisión (`DecisionTreeRegressor`).
- **Parte B, clasificación:** predecir la droga que se le receta a un paciente (`Droga`) con un árbol de decisión con post-poda, Bayes ingenuo (`CategoricalNB`) y k-NN, y comparar los tres.

Todo se evalúa con dos particiones, 80/20 y 70/30, con `random_state = 63`.

El entorno y la forma de trabajar con los notebooks (jupytext) son comunes a todos los TP: están en el [README del repo](../README.md).

## Archivos

```
tp2/
├── data/
│   ├── 1000_Companies.csv                   dataset de la Parte A
│   └── drugType.csv                         dataset de la Parte B
├── docs/
│   └── TP2_enunciado.pdf                    consigna de la cátedra
├── notebooks/
│   ├── TP2_calabozo_darruiz_dicarlo.py      notebook del TP, versionado en git (formato jupytext)
│   └── TP2_calabozo_darruiz_dicarlo.ipynb   copia local generada con jupytext (no se sube)
└── README.md
```

## Ejecución

Con el entorno activado y desde la raíz del repo, generar el notebook a partir del `.py`:

```bash
jupytext --sync tp2/notebooks/TP2_calabozo_darruiz_dicarlo.py
```

Después, abrir `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb` con el kernel **Python 3.12 (TP MD)** (se registra en la instalación, ver el [README del repo](../README.md)) y ejecutar todas las celdas. El notebook lee `../data/` con una ruta relativa a su carpeta: se ejecuta con `tp2/notebooks/` como carpeta de trabajo, que es lo que hacen Jupyter y VS Code por defecto.

Para ejecutarlo y guardar las salidas sin abrirlo:

```bash
jupyter nbconvert --to notebook --execute --inplace tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb
```

La corrida completa tarda alrededor de dos minutos, casi todo en las búsquedas de hiperparámetros de los árboles. Si cambia el código, hay que volver a ejecutar todo y revisar que los números de los textos coincidan con las salidas.

## Entrega

La consigna pide el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, ejecutar todo (**Restart & Run All** o el `nbconvert` de arriba) y subir `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb`, con todas las salidas visibles, al campus.
