# TP1 — Minería de Datos (2026)

Trabajo Práctico N° 1 de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastian Di Carlo

Análisis del dataset de pingüinos de Palmer (`penguins_size.csv`): análisis exploratorio, reducción de dimensionalidad (PCA, Isomap, t-SNE) y clustering (K-means y jerárquico), usando la especie como variable objetivo.

## Estructura del proyecto

```
tp1-mineria-datos/
├── data/
│   └── penguins_size.csv          # dataset provisto por la cátedra
├── docs/
│   └── TP1_enunciado.pdf          # consigna del trabajo
├── notebooks/
│   ├── TP1_Mineria_Datos.py       # informe versionado en git (formato jupytext)
│   ├── TP1_Mineria_Datos_con_sexo.py  # versión alternativa que usa Sexo como característica
│   └── *.ipynb                    # copias locales generadas con jupytext (no se suben)
├── jupytext.toml                  # empareja cada .ipynb con su .py
├── requirements.txt               # dependencias con versiones fijas
├── .gitattributes
├── .gitignore
└── README.md
```

## ¿Por qué hay un `.py` y un `.ipynb`?

Los `.ipynb` son archivos JSON que guardan también las salidas (tablas, imágenes). En git eso genera diffs ilegibles y conflictos cada vez que alguien ejecuta el notebook. Por eso usamos [jupytext](https://jupytext.readthedocs.io/):

- **`TP1_Mineria_Datos.py`** es el archivo que se sube a git. Tiene el mismo contenido que el notebook (código y textos), sin las salidas, y se lee como código normal.
- **`TP1_Mineria_Datos.ipynb`** es la copia local de cada uno, donde se trabaja y se ven los gráficos. Está en `.gitignore`.
- `jupytext --sync` mantiene los dos archivos iguales.

## Instalación (una sola vez)

Requisito: **Python 3.12** (es la versión con la que se probó).

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/jcalabozo/tp1-mineria-datos.git
   cd tp1-mineria-datos
   ```

2. Crear y activar el entorno virtual:

   **Windows (PowerShell)**
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux / macOS**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar las dependencias **en este orden**:

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install --no-build-isolation gap-stat==2.0.3
   pip install -r requirements.txt
   ```

   > `gap-stat` (la librería que se usa en clase para el estadístico GAP) no se puede compilar con la configuración por defecto de las versiones actuales de pip. Por eso se instala antes y con `--no-build-isolation`. Si se ejecuta directamente `pip install -r requirements.txt`, falla.

   > Si PowerShell no permite activar el entorno, ejecutar una vez:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

4. Generar el notebook a partir del `.py`:

   ```bash
   jupytext --sync notebooks/TP1_Mineria_Datos.py
   ```

   Para la versión con `Sexo`, lo mismo con `notebooks/TP1_Mineria_Datos_con_sexo.py`.

## Cómo trabajar (cada vez)

Con el entorno activado:

1. **Antes de empezar**, traer los cambios de los demás y actualizar el notebook:

   ```bash
   git pull
   jupytext --sync notebooks/TP1_Mineria_Datos.py
   ```

2. **Trabajar** en `notebooks/TP1_Mineria_Datos.ipynb` con VS Code (kernel del `.venv`) o con `jupyter notebook`, como siempre.

3. **Al terminar**, pasar los cambios al `.py` y subirlos:

   ```bash
   jupytext --sync notebooks/TP1_Mineria_Datos.ipynb
   git add notebooks/TP1_Mineria_Datos.py
   git commit -m "Descripción breve del cambio"
   git push
   ```

> ⚠️ **Siempre hacer el paso 3 antes de un `git pull`.** Si se edita el `.ipynb`, no se sincroniza y se hace pull, el siguiente `jupytext --sync` puede pisar esos cambios con la versión del `.py` que vino del repositorio.

Recomendaciones:
- Antes de subir cambios, reiniciar el kernel y ejecutar todo (**Restart & Run All**) para verificar que el notebook corre de principio a fin (tarda un par de minutos).
- Si dos personas modifican la misma parte, git marca el conflicto en el `.py`, que se resuelve como cualquier archivo de código.
- El notebook lee los datos desde `../data/`, así que se debe abrir desde la carpeta `notebooks/`.

## Entrega

La consigna acepta el informe en formato `.ipynb`. Para entregar: sincronizar con el último `.py` del repositorio, abrir `notebooks/TP1_Mineria_Datos.ipynb`, ejecutar **Restart & Run All** y subir ese archivo (con todas las salidas visibles) al campus.

Si se prefiere una versión para leer en el navegador:

```bash
jupyter nbconvert --to html notebooks/TP1_Mineria_Datos.ipynb
```
