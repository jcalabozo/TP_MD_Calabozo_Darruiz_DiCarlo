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
│   ├── TP2_enunciado.pdf                    consigna de la cátedra
│   ├── informe.md                           informe en Markdown, del que sale el PDF
│   ├── informe.pdf                          informe con carátula
│   ├── caratula.toml                        datos de la carátula del PDF
│   └── figuras/                             figuras del informe (las guarda el notebook)
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

Después, abrir `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb` con el kernel del `.venv` y ejecutar todas las celdas. El notebook lee `../data/` y guarda las figuras del informe en `../docs/figuras/`, con rutas relativas a su carpeta: se ejecuta con `tp2/notebooks/` como carpeta de trabajo, que es lo que hacen Jupyter y VS Code por defecto.

Para ejecutarlo y guardar las salidas sin abrirlo:

```bash
jupyter nbconvert --to notebook --execute --inplace tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb
```

La corrida completa tarda menos de un minuto.

## Informe en PDF

`docs/informe.pdf` sale de `docs/informe.md` con la plantilla común de informes (`tuia/_plantillas/informe/`, que no está en este repo). Desde la carpeta `tuia`, con el entorno de las plantillas activado:

```bash
python _plantillas/informe/generar_informe.py tuia-mineria-datos/TP_MD_Calabozo_Darruiz_DiCarlo/tp2/docs/informe.md tuia-mineria-datos/TP_MD_Calabozo_Darruiz_DiCarlo/tp2/docs/caratula.toml
```

Si cambian los resultados, primero hay que volver a ejecutar el notebook (que actualiza `docs/figuras/`) y corregir los números del informe.

## Entrega

La consigna pide el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, ejecutar todo (**Restart & Run All** o el `nbconvert` de arriba) y subir `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb`, con todas las salidas visibles, al campus.
