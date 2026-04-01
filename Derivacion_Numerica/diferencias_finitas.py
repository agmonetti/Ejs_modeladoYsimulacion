import sympy as sp
import numpy as np

def calcular_diferencias_finitas_centrales(func_str, x_val, h_val):
    """
    Calcula la primera y segunda derivada usando diferencias finitas centradas
    y las compara con el valor analítico exacto.
    """
    print("="*60)
    print(f" DIFERENCIAS FINITAS CENTRADAS ")
    print(f" f(x) = {func_str} | Evaluando en x = {x_val} | h = {h_val}")
    print("="*60, "\n")
    
    # 1. Configuración Simbólica (SymPy) para valores exactos
    x = sp.Symbol('x')
    f_sym = sp.sympify(func_str)
    
    # Derivadas analíticas
    df1_sym = sp.diff(f_sym, x)
    df2_sym = sp.diff(df1_sym, x)
    
    # Valores exactos evaluados
    exacto_1ra = float(df1_sym.subs(x, x_val).evalf())
    exacto_2da = float(df2_sym.subs(x, x_val).evalf())
    
    # 2. Cálculo Numérico (El algoritmo de la materia)
    # Función lambda rápida para evaluar f(x)
    f = sp.lambdify(x, f_sym, 'numpy')
    
    # Fórmulas numéricas centradas
    aprox_1ra = (f(x_val + h_val) - f(x_val - h_val)) / (2 * h_val)
    aprox_2da = (f(x_val + h_val) - 2 * f(x_val) + f(x_val - h_val)) / (h_val**2)
    
    # Errores absolutos
    error_1ra = abs(exacto_1ra - aprox_1ra)
    error_2da = abs(exacto_2da - aprox_2da)
    
    # 3. Salida de Resultados
    print("--- PRIMERA DERIVADA f'(x) ---")
    print(f"Aproximación Numérica : {aprox_1ra:.6f}")
    print(f"Valor Exacto (Analítico): {exacto_1ra:.6f}")
    print()
    print(f"Error Absoluto: {error_1ra:.6f} ----- (que es igual a {error_1ra:.6e})")
    
    print()
    print("--------------------------------------------")
    print("--- SEGUNDA DERIVADA f''(x) ---")
    print(f"Aproximación Numérica : {aprox_2da:.6f}")
    print(f"Valor Exacto (Analítico): {exacto_2da:.6f}")
    print()
    print(f"Error Absoluto: {error_2da:.6f} ----- (que es igual a {error_2da:.6e})\n")


# ejemplo f(x) = sen(x), en x=pi/4, con h=0.1
calcular_diferencias_finitas_centrales('ln(x+1)', np.pi / 4, 0.1)