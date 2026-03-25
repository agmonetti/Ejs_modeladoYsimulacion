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

### 3. Método de Newton-Raphson (`nRaphson.py`)
Implementación del método de Newton-Raphson para encontrar las raíces de una función no lineal aproximándose mediante su derivada.
- **Características**:
  - Cálculo de la derivada numérica utilizando el método de diferencias centrales (no requiere librería extra para derivar).
  - Tabla de iteraciones interactiva generada con `tabulate`.
  - Visualización gráfica de la función y la posición exacta de la raíz usando `matplotlib`.
- **Uso**: El usuario debe proveer la función principal $f(x)$ y una aproximación o valor inicial $x_0$.

### 4. Método de Aceleración de Aitken (`aitken.py`)
Implementación de la mejora $\Delta^2$ de Aitken aplicada sobre el método secuencial de Punto Fijo.
- **Características**:
  - Mejora y acelera drásticamente la tasa de convergencia en relación al Punto Fijo estándar.
  - Presentación mediante consola de una tabla con los cálculos de las iteraciones puente unificadas ($x_1$, $x_2$) y el valor acelerado precalculado.
- **Uso**: El usuario debe proveer el punto inicial $x_0$ y la función iterante $g(x)$.

## Dependencias

```bash
pip install numpy matplotlib tabulate sympy
```
