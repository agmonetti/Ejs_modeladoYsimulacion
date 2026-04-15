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

### 5. Simulación de Monte Carlo
Ubicación: `/Simulacion_Montecarlo/`

Aplicación con Interfaz Gráfica (GUI) para la resolución estocástica de integrales definidas, análisis de convergencia y estimación de intervalos de confianza estadísticos.
**Características destacadas del módulo:**
* **Motor Matemático Blindado:** Compilación de strings a funciones vectorizadas de NumPy vía SymPy para ejecución ultrarrápida.
* **Análisis Estadístico Predictivo:** Cálculo dinámico de Z-Scores (usando SciPy), validación contra un Error Máximo permitido y proyección de $N$ para factores de reducción $j$ exigidos en parciales.
* **Output de Pizarra:** Generación del desarrollo analítico de la sumatoria paso a paso.

**Archivos Principales:**
* **`motor_montecarlo.py`**: Core lógico. Implementa Acierto y Fallo (Hit-or-Miss), Método del Valor Promedio (1D, 2D y 3D) y la Cuadratura de Gauss como pivote de error exacto.
* **`interfaz.py`**: Frontend construido con Tkinter. Contiene gráficos interactivos con Matplotlib, tablas de recolección de muestras, teclado matemático avanzado flotante y selectores de confianza.
* **`main.py`**: Archivo orquestador de entrada a la aplicación.

### 6. Ecuaciones Diferenciales Ordinarias (EDO)
Ubicación: `/Ecuaciones_Diferenciales_Ordinarias/`

Implementaciones numéricas para resolver problemas de valor inicial (Cauchy) de la forma $y' = f(x, y)$, con comparación contra solución analítica usando `sympy` y visualización con `matplotlib`.

**Métodos Implementados:**
* **Euler (`euler.py`)**: Método explícito de primer orden. Muestra tabla iterativa completa ($x_n$, $y_n$, $y_{n+1}$), error absoluto respecto de la solución exacta y gráfico comparativo.
* **Euler Mejorado / Heun (`heun.py`)**: Método predictor-corrector de segundo orden. Incluye columnas de predicción y corrección por paso, junto con análisis de error y curva comparada.
* **Runge-Kutta de 4to Orden (`rungeKutta4.py`)**: Método de cuarto orden con cálculo de $k_1$, $k_2$, $k_3$, $k_4$ por iteración, tabla detallada y validación frente a la solución exacta.

* **Comparador General (`comparador.py`)**: Script unificado que ejecuta los tres métodos sobre una misma EDO, imprime tabla comparativa de aproximaciones y errores, y grafica tanto soluciones como evolución del error en escala logarítmica.

---

## Dependencias

```bash
pip install numpy matplotlib tabulate sympy scipy