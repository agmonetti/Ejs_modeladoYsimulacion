# Ejercicios de Modelado y Simulación

Este repositorio contiene implementaciones en Python de métodos numéricos clásicos

## Métodos Implementados

### 1. Método de Bisección (`biseccion.py`)
Implementación del método de bisección (o partición de intervalos) el cual se basa en el Teorema de Bolzano.
- **Características**:
  - Búsqueda automática de intervalos (barrido escalar para detectar cambios de signo).
  - Tabla de iteraciones detallada usando la librería `tabulate`.
  - Visualización gráfica de la función y la raíz encontrada mediante `matplotlib`.
- **Uso**: El usuario debe proveer el intervalo inicial `[a, b]` y la función a evaluar `f(x)`.

### 2. Método de Iteración de Punto Fijo (`puntoFijo.py`)
Implementación de la iteración de punto fijo para encontrar las raíces de una ecuación transformándola a la forma $x = g(x)$.
- **Características**:
  - Imprime el valor de $x$ en cada iteración.
  - Verifica dinámicamente la convergencia analizando la tolerancia requerida comparada con el error absoluto de iteraciones sucesivas.
- **Uso**: El usuario debe proveer el punto inicial $x_0$ y la función $g(x)$.

## Dependencias

```bash
pip install numpy matplotlib tabulate
```
