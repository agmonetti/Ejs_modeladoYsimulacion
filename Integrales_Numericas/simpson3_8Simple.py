import numpy as np

# 1. Definimos la función a integrar
def funcion(x):
    return np.sin(x)

# 2. Helper para el error (Cuarta derivada)
def cuarta_derivada_numerica(f, x, dx=1e-3):
    return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)

def simpson_38_simple(f, a, b, precision=6):
    print("\n" + "="*50)
    print(" MÉTODO: SIMPSON 3/8 SIMPLE ")
    print("="*50)
    
    # En la regla simple, n siempre es 3
    n = 3
    h = (b - a) / n
    print(f"Parámetros: n = {n}  ->  h = {h}")
    
    # --- 1. DEFINICIÓN DE PUNTOS ---
    x1 = a + h
    x2 = a + 2*h
    
    # Evaluamos la función
    fa = f(a)
    fx1 = f(x1)
    fx2 = f(x2)
    fb = f(b)
    
    # --- 2. DESARROLLO ESCRITO DE LA FÓRMULA ---
    fraccion_h = f"3*{h}/8" if isinstance(h, float) else f"3({b}-{a})/24"
    desarrollo = f"I ~= {fraccion_h} [ {fa:.{precision}f} + 3({fx1:.{precision}f}) + 3({fx2:.{precision}f}) + {fb:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- 3. CÁLCULO NUMÉRICO DE LA INTEGRAL ---
    integral = (3 * h / 8) * (fa + 3 * fx1 + 3 * fx2 + fb)
    
    # --- 4. CÁLCULO DEL ERROR DE TRUNCAMIENTO ---
    x_fino = np.linspace(a, b, 1000) 
    max_cuarta = np.max(np.abs(cuarta_derivada_numerica(f, x_fino)))
    
    # Fórmula del error para Simpson 3/8 simple: (3/80) * h^5 * max|f''''(x)|
    cota_error = (3 / 80) * (h**5) * max_cuarta
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.{precision}f}")
    print(f"E_t <= ±{cota_error:.8e}")
    print("="*50 + "\n")
    
    return integral, cota_error

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    a = 0
    b = np.pi
    decimales = 6
    
    resultado, error = simpson_38_simple(funcion, a, b, precision=decimales)