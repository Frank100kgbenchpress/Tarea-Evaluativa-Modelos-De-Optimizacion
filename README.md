# Entorno y uso del notebook para la Tarea de Optimización

Este README contiene pasos concretos (para bash y conda) para crear un entorno Python aislado, instalar las dependencias necesarias y registrar un kernel IPython que puedas seleccionar en Jupyter o VS Code.

Requisitos mínimos: Python 3.8+ (recomendado 3.10+).

Archivos importantes:

- `requirements.txt` — lista de dependencias (numpy, matplotlib, ipykernel, jupyterlab, scipy).
- `codigo.ipynb` — tu notebook con el código de optimización (ya presente en el workspace).

## Opción A — Entorno virtual (venv) y pip (Linux / bash)

1. Crear y activar el entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Actualizar pip e instalar dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Registrar un kernel para Jupyter/VS Code (opcional pero recomendado):

```bash
python -m ipykernel install --user --name=tarea-opt --display-name "Python (tarea-opt)"
```

4. Iniciar Jupyter (opcional):

```bash
jupyter lab
# o
jupyter notebook
```

Luego abre `codigo.ipynb` y en la interfaz (Jupyter o VS Code) selecciona el kernel "Python (tarea-opt)".

## Opción B — Usando conda

```bash
conda create -n tarea-opt python=3.10 -y
conda activate tarea-opt
pip install -r requirements.txt
# o instalar por conda (opcional):
conda install numpy matplotlib scipy -y
python -m ipykernel install --user --name=tarea-opt --display-name "Python (tarea-opt)"
```

## Cómo ejecutar tu código en un notebook

1. Abre `codigo.ipynb` en Jupyter (o en VS Code como Notebook interactivo).
2. Asegúrate de elegir el kernel `Python (tarea-opt)`.
3. Ejecuta las celdas de arriba hacia abajo. Si prefieres, copia y pega el script Python que tienes en una sola celda de código y ejecútala.

## Recomendaciones y problemas numéricos

- La función usa `np.tan(1.5 * np.sin(x+y))` — ten cuidado: `tan` puede explotar si su argumento está cerca de pi/2 + k*pi. Al evaluar en regiones grandes (ej. [-100, 100]) puede producir overflow o valores NaN.
- Para evitar overflow en la visualización, usamos una malla más pequeña (p. ej. `[-2,2]`).
- Si el optimizador produce NaNs, prueba:
  - reducir el tamaño del paso `alpha` en descenso de gradiente; o
  - añadir comprobaciones para detectar `np.isnan`/`np.isinf` y cortar el paso; o
  - limitar la búsqueda a una región razonable.
- Para BFGS, si la condición `y^T s` es muy pequeña, evita actualizar B (ya lo haces en el código con un umbral).

## Comandos rápidos de verificación

Activar venv y ejecutar el notebook (resumen):

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=tarea-opt --display-name "Python (tarea-opt)"
jupyter lab
```

## Notas finales

Si usas VS Code: instala la extensión oficial de Jupyter y la de Python. Abre `codigo.ipynb`, elige el kernel `Python (tarea-opt)` desde la parte superior derecha y ejecuta las celdas.

Si quieres, puedo:

- crear un kernel automáticamente desde aquí (no puedo ejecutar comandos en tu máquina sin permiso), o
- añadir una celda inicial en `codigo.ipynb` que muestre la versión de paquetes y confirme el kernel.

Fin.
