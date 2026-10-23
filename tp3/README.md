# TP3 — Minería de Datos (2026)

Trabajo Práctico N° 3 de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo

Clasificación multiclase con los modelos de las unidades 5 y 6: recomendar un cultivo (`Cultivo`, 12 clases) a partir de parámetros de suelo y clima (`dxCropRecommendation.csv`), con SVM de kernel lineal, SVM de kernel gaussiano (RBF) y Random Forest, y comparar los tres. El dataset fue modificado a propósito por la cátedra: hay que detectar y tratar los valores faltantes y los duplicados.

Todo se evalúa con validación cruzada estratificada de 5 folds, con `random_state = 13`.

**Estado: sin resolver.** El notebook tiene solo la cabecera, la semilla, la ruta de los datos y los títulos de las actividades del enunciado.

El entorno y la forma de trabajar con los notebooks (jupytext) son comunes a todos los TP: están en el [README del repo](../README.md).

## Archivos

```
tp3/
├── data/
│   └── dxCropRecommendation.csv             dataset provisto por la cátedra
├── docs/
│   └── TP3_enunciado.pdf                    consigna de la cátedra
├── notebooks/
│   ├── TP3_calabozo_darruiz_dicarlo.py      notebook del TP, versionado en git (formato jupytext)
│   └── TP3_calabozo_darruiz_dicarlo.ipynb   copia local generada con jupytext (no se sube)
└── README.md
```

## Ejecución

Con el entorno activado y desde la raíz del repo, generar el notebook a partir del `.py`:

```bash
jupytext --sync tp3/notebooks/TP3_calabozo_darruiz_dicarlo.py
```

Después, abrir `tp3/notebooks/TP3_calabozo_darruiz_dicarlo.ipynb` con el kernel del `.venv`. El notebook lee los datos desde `../data/`, con rutas relativas a su carpeta: se ejecuta con `tp3/notebooks/` como carpeta de trabajo, que es lo que hacen Jupyter y VS Code por defecto.

## Entrega

La consigna pide el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, ejecutar **Restart & Run All** y subir `tp3/notebooks/TP3_calabozo_darruiz_dicarlo.ipynb`, con todas las salidas visibles, al campus.
