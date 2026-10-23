# TPs — Minería de Datos (2026)

Trabajos Prácticos de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial).

**Integrantes:** Josías Calabozo · Ismael Darruiz · Sebastián Di Carlo

| Carpeta | Trabajo | Temas |
|---|---|---|
| [`tp1/`](tp1/) | Trabajo Práctico N° 1 | Pingüinos de Palmer: análisis exploratorio, reducción de dimensionalidad (PCA, Isomap, t-SNE) y clustering (K-means y jerárquico) |
| [`tp2/`](tp2/) | Trabajo Práctico N° 2 | Árboles de decisión para regresión y clasificación, Bayes ingenuo y k-NN |
| [`tp3/`](tp3/) | Trabajo Práctico N° 3 | Recomendación de cultivos: SVM (kernel lineal y RBF) y Random Forest. **Sin resolver:** solo tiene la estructura |

Cada TP tiene su README con lo que resuelve, sus archivos y cómo se entrega. Lo de esta página, el entorno y la forma de trabajar con los notebooks, es común a todos.

## Estructura del proyecto

```
TP_MD_Calabozo_Darruiz_DiCarlo/
├── tp1/
│   ├── data/                      # dataset provisto por la cátedra
│   ├── docs/                      # consigna del trabajo
│   ├── notebooks/                 # notebooks del TP (.py en git, .ipynb locales)
│   └── README.md
├── tp2/
│   ├── data/                      # datasets provistos por la cátedra
│   ├── docs/                      # consigna, informe (Markdown y PDF) y sus figuras
│   ├── notebooks/                 # notebook del TP (.py en git, .ipynb local)
│   └── README.md
├── tp3/
│   ├── data/                      # dataset provisto por la cátedra
│   ├── docs/                      # consigna del trabajo
│   ├── notebooks/                 # notebook del TP (.py en git, .ipynb local)
│   └── README.md
├── jupytext.toml                  # empareja cada .ipynb con su .py
├── requirements.txt               # dependencias de los TP, con versiones fijas
├── .gitattributes
├── .gitignore
└── README.md
```

## ¿Por qué hay un `.py` y un `.ipynb`?

Los `.ipynb` son archivos JSON que guardan también las salidas (tablas, imágenes). En git eso genera diffs ilegibles y conflictos cada vez que alguien ejecuta el notebook. Por eso usamos [jupytext](https://jupytext.readthedocs.io/):

- **El `.py`** (por ejemplo, `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.py`) es el archivo que se sube a git. Tiene el mismo contenido que el notebook (código y textos), sin las salidas, y se lee como código normal.
- **El `.ipynb`** es la copia local de cada uno, donde se trabaja y se ven los gráficos. Está en `.gitignore`.
- `jupytext --sync` mantiene los dos archivos iguales. `jupytext.toml`, en la raíz, vale para los notebooks de todos los TP.

## Instalación (una sola vez)

Requisito: **Python 3.12** (es la versión con la que se probó). Un solo entorno, en la raíz del repo, sirve para todos los TP.

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/jcalabozo/TP_MD_Calabozo_Darruiz_DiCarlo.git
   cd TP_MD_Calabozo_Darruiz_DiCarlo
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

   > `gap-stat` (la librería que se usa en clase para el estadístico GAP, en el TP1) no se puede compilar con la configuración por defecto de las versiones actuales de pip. Por eso se instala antes y con `--no-build-isolation`. Si se ejecuta directamente `pip install -r requirements.txt`, falla.

   > Si PowerShell no permite activar el entorno, ejecutar una vez:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

4. Generar los notebooks a partir de los `.py`:

   ```bash
   jupytext --sync tp1/notebooks/TP1_Mineria_Datos.py tp1/notebooks/TP1_Mineria_Datos_con_sexo.py tp2/notebooks/TP2_calabozo_darruiz_dicarlo.py tp3/notebooks/TP3_calabozo_darruiz_dicarlo.py
   ```

## Cómo trabajar (cada vez)

Con el entorno activado y desde la raíz del repo. Los ejemplos son del TP2; para el TP1 es igual, con sus notebooks.

1. **Antes de empezar**, traer los cambios de los demás y actualizar el notebook:

   ```bash
   git pull
   jupytext --sync tp2/notebooks/TP2_calabozo_darruiz_dicarlo.py
   ```

2. **Trabajar** en `tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb` con VS Code (kernel del `.venv`) o con `jupyter notebook`, como siempre.

3. **Al terminar**, pasar los cambios al `.py` y subirlos:

   ```bash
   jupytext --sync tp2/notebooks/TP2_calabozo_darruiz_dicarlo.ipynb
   git add tp2/notebooks/TP2_calabozo_darruiz_dicarlo.py
   git commit -m "Descripción breve del cambio"
   git push
   ```

> ⚠️ **Siempre hacer el paso 3 antes de un `git pull`.** Si se edita el `.ipynb`, no se sincroniza y se hace pull, el siguiente `jupytext --sync` puede pisar esos cambios con la versión del `.py` que vino del repositorio.

Recomendaciones:
- Antes de subir cambios, reiniciar el kernel y ejecutar todo (**Restart & Run All**) para verificar que el notebook corre de principio a fin.
- Si dos personas modifican la misma parte, git marca el conflicto en el `.py`, que se resuelve como cualquier archivo de código.
- Cada notebook lee sus datos desde `../data/`, con rutas relativas a su carpeta `notebooks/`: se debe abrir desde ahí, que es lo que hacen Jupyter y VS Code por defecto.

## Entrega

Las dos consignas aceptan el informe en formato `.ipynb`: se sube el notebook ejecutado, con todas las salidas visibles. Qué archivo se entrega en cada TP está en su README.
