# Ejercicios de Modelado y Simulación

Este repositorio contiene implementaciones en Python de métodos numéricos clásicos, organizados modularmente por unidad temática.

## Estructura

### 1. Búsqueda de Raíces (Ecuaciones No Lineales)
Ubicación: `/Busqueda_Raices/`

Algoritmos para encontrar en qué punto una función cruza el eje X ($f(x) = 0$).
* **Bisección (`biseccion.py`)**: Método cerrado basado en el Teorema de Bolzano. Incluye barrido escalar automático para detectar cambios de signo.
* **Punto Fijo (`puntoFijo.py`)**: Método abierto que transforma la ecuación a la forma $x = g(x)$. Verifica dinámicamente la convergencia analizando el error absoluto.
* **Newton-Raphson (`nRaphson.py`)**: Método abierto apoyado en la derivada numérica (diferencias centrales).
* **Aceleración de Aitken (`aitken.py`)**: Mejora $\Delta^2$ aplicada sobre el método de Punto Fijo para acelerar drásticamente la tasa de convergencia.
* **Orquestador General (`bnap.py`)**: Consolidación de los cuatro métodos en un único archivo para su ejecución simultánea. Incluye análisis comparativo de convergencia y gráficos unificados mediante `matplotlib`.

### 2. Derivación Numérica
Ubicación: `/Derivacion_Numerica/`

* **Diferencias Finitas (`diferencias_finitas.py`)**: Implementación del cálculo numérico de la primera y segunda derivada mediante aproximaciones por diferencias finitas centradas. Compara automáticamente con valores exactos usando `sympy` y calcula el error absoluto.

### 3. Integración Numérica (Fórmulas de Newton-Cotes)
Ubicación: `/Integrales_Numericas/`

Familia de algoritmos para la aproximación del área bajo la curva. 
**Características destacadas del módulo:**
* **Formato de Pizarra:** Las salidas por consola generan tablas de iteración y desarrollos algebraicos idénticos a las resoluciones exigidas en clase (UADE).
* **Precisión Dinámica:** Permiten inyectar la cantidad exacta de decimales requeridos para el reporte.
* **Error de Truncamiento Autónomo:** Calculan automáticamente su propia cota de error máximo ($E_T$) utilizando derivadas numéricas de orden superior (diferencias centrales), sin requerir derivación manual.

**Métodos Implementados:**
* **Rectángulo Medio (`rectanguloCompuesto.py`)**: Regla compuesta evaluando el punto medio de cada subintervalo.
* **Trapecios (`trapecioSimple.py`, `trapecioCompuesto.py`)**: Aproximación lineal (Newton-Cotes de orden 1).
* **Simpson 1/3 (`simpson1_3Simple.py`, `simpson1_3Compuesto.py`)**: Aproximación cuadrática. El método compuesto incluye validación estricta de intervalos pares ($n$ par).
* **Simpson 3/8 (`simpson3_8Simple.py`, `simpson3_8Compuesto.py`)**: Aproximación cúbica. El método compuesto incluye validación estricta de intervalos múltiplos de 3.
* **Resumen Comparativo (`unificacionMetodosCompuestos.py`)**: Script maestro que ejecuta las formas compuestas del Rectángulo, Trapecio y Simpson 1/3 sobre una misma función, emitiendo una tabla comparativa de resultados finales y cotas de error.

### 4. Interpolación Numérica
Ubicación: `/Interpolacion_Numerica/`

* **Interpolación de Lagrange (`lagrange.py`)**: Reconstrucción de polinomios a partir de puntos conocidos. Soporta tanto funciones explícitas como conjuntos de datos discretos. Construye paso a paso el polinomio, calcula errores locales y cotas teóricas globales, y grafica la comparativa.

---

## Dependencias

```bash
pip install numpy matplotlib tabulate sympy