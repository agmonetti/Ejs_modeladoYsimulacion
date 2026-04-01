import numpy as np
import warnings
from tabulate import tabulate

# -----------------------------------
#  1. LA FUNCIÓN PURA
# -----------------------------------
def funcion(x):
    return np.sin(x) / x


# -----------------------------------
#  2. EL ESCUDO NUMÉRICO CON ALERTAS
# -----------------------------------
def evaluar_seguro(f, x, silencioso=False):
    """
    Evalúa cualquier función f en el vector x. 
    Si encuentra un NaN o Infinito, lo repara y avisa por consola.
    """
    x_arr = np.atleast_1d(x).astype(float)
    
    # Suprimimos temporalmente las advertencias rojas de NumPy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = f(x_arr)
        
    # Buscamos dónde explotó la función
    malos_indices = np.where(np.isnan(y) | np.isinf(y))[0]
    
    # Si encontró problemas, los repara y avisa
    if len(malos_indices) > 0:
        if not silencioso:
            print("Se detectó una indeterminación (NaN/Infinito) al evaluar la función.")
            
        epsilon = 1e-9 # Un paso infinitamente pequeño
        for i in malos_indices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y[i] = f(x_arr[i] + epsilon)
            
            if not silencioso:
                print(f" -> El punto crítico x = {x_arr[i]} fue aproximado por límite a f(x) ~= {y[i]:.6f}")
                
    return y[0] if len(y) == 1 else y


# -----------------------------------
#  3. DERIVADA PROTEGIDA (MODO SIGILOSO)
# -----------------------------------
def cuarta_derivada_numerica(f, x, dx=1e-3):
    """Aproxima la cuarta derivada pasando por el escudo en modo silencioso."""
    f_x2dx = evaluar_seguro(f, x + 2*dx, silencioso=True)
    f_xdx  = evaluar_seguro(f, x + dx, silencioso=True)
    f_x    = evaluar_seguro(f, x, silencioso=True)
    f_mdx  = evaluar_seguro(f, x - dx, silencioso=True)
    f_m2dx = evaluar_seguro(f, x - 2*dx, silencioso=True)
    
    return (f_x2dx - 4*f_xdx + 6*f_x - 4*f_mdx + f_m2dx) / (dx**4)


# -----------------------------------
#  4. MÉTODO: SIMPSON 3/8 COMPUESTO
# -----------------------------------
def simpson_38_compuesto(f, a, b, n, precision=6):
    print("\n" + "="*60)
    print(" MÉTODO: SIMPSON 3/8 COMPUESTO ")
    print("="*60)
    
    if n % 3 != 0:
        raise ValueError("¡Error! La regla de Simpson 3/8 exige que el número de subintervalos (n) sea MÚLTIPLO DE 3.")

    h = (b - a) / n
    fraccion_display = f"{int(b-a)}/{n}" if (b-a).is_integer() else f"{h:.{precision}f}"
    print(f"Parámetros: n = {n}  ->  h = {fraccion_display}")
    
    # --- EVALUACIÓN CON ALERTA ---
    x = np.linspace(a, b, n + 1)
    # Al no pasarle 'silencioso=True', si hay un NaN aquí, lo imprimirá
    y = evaluar_seguro(f, x) 
    
    # --- CONSTRUCCIÓN DE LA TABLA ---
    tabla_pizarra = []
    for i in range(n + 1):
        x_str = f"{i}/{n}" if a == 0 and b == 1 else f"{round(x[i], precision)}"
        tabla_pizarra.append([i, x_str, round(y[i], precision)])
        
    print("\nTABLA DE VALORES:")
    print(tabulate(tabla_pizarra, headers=["n", "x_n", "f(x_n)"], tablefmt="grid", disable_numparse=True))
    
    # --- AGRUPACIÓN Y DESARROLLO DE LA FÓRMULA ---
    grupo_1 = y[1:n:3]
    grupo_2 = y[2:n:3]
    grupo_3 = y[3:n-1:3]
    
    str_1 = " + ".join([f"{val:.{precision}f}" for val in grupo_1])
    str_2 = " + ".join([f"{val:.{precision}f}" for val in grupo_2])
    str_3 = " + ".join([f"{val:.{precision}f}" for val in grupo_3])
    
    fraccion_h = f"3({fraccion_display})/8"
    termino_mult3 = f" + 2({str_3})" if str_3 else ""
    desarrollo = f"I ~= {fraccion_h} [ {y[0]:.{precision}f} + 3({str_1}) + 3({str_2}){termino_mult3} + {y[-1]:.{precision}f} ]"
    
    print("\nDESARROLLO DE LA FÓRMULA:")
    print(desarrollo)
    
    # --- CÁLCULO NUMÉRICO DE LA INTEGRAL ---
    S = y[0] + y[-1] + 3 * np.sum(grupo_1) + 3 * np.sum(grupo_2) + 2 * np.sum(grupo_3)
    integral = (3 * h / 8) * S
    
    # --- CÁLCULO DEL ERROR DE TRUNCAMIENTO ---
    dx_seguridad = 1e-3
    x_fino = np.linspace(a + 2*dx_seguridad, b - 2*dx_seguridad, 1000) 
    
    arreglo_cuartas_derivadas = cuarta_derivada_numerica(f, x_fino)
    max_cuarta = np.max(np.abs(arreglo_cuartas_derivadas))
    
    cota_error = ((b - a)**5 / (80 * n**4)) * max_cuarta
    
    print(f"\nRESULTADOS FINALES:")
    print(f"I ~= {integral:.8f}")
    print(f"E_t <= ±{cota_error:.8e}")
    print("="*60 + "\n")
    
    return integral, cota_error


# -----------------------------------
#  BLOQUE PRINCIPAL
# -----------------------------------
if __name__ == "__main__":
    a = 0
    b = 1
    n = 6 
    decimales = 6
    
    resultado, error = simpson_38_compuesto(funcion, a, b, n, precision=decimales)