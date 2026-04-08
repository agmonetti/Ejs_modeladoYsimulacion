import numpy as np
import sympy as sp
import warnings
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

def calcular_z_score(nivel_confianza=0.95):
    """
    Calcula el z-score para un nivel de confianza dado.
    
    Args:
        nivel_confianza: Valor entre 0 y 1 (ej: 0.95 para 95%)
    
    Returns:
        z_score: Valor Z correspondiente
    """
    alfa = 1.0 - nivel_confianza
    z_score = norm.ppf(1.0 - (alfa / 2.0))
    return z_score

def compilar_funcion(texto_funcion, variables='x'):
    """
    Convierte un string matemático en una función NumPy ultrarrápida.
    Soporta múltiples variables (ej: variables='x y' o 'x y z').
    Reemplaza 'e^' con 'exp(' para compatibilidad con notación matemática.
    """
    try:
        # Reemplazar notación e^ con exp(
        texto_funcion = texto_funcion.replace('e^', 'exp(')
        # Contar paréntesis faltantes y cerrarlos
        abiertos = texto_funcion.count('(')
        cerrados = texto_funcion.count(')')
        if abiertos > cerrados:
            texto_funcion = texto_funcion + ')' * (abiertos - cerrados)
        
        vars_sympy = sp.symbols(variables)
        expr = sp.sympify(texto_funcion)
        # lambdify requiere una tupla de variables si son múltiples
        if isinstance(vars_sympy, sp.Symbol):
            f_compilada = sp.lambdify(vars_sympy, expr, modules=['numpy'])
        else:
            f_compilada = sp.lambdify(vars_sympy, expr, modules=['numpy'])
        return f_compilada, None
    except Exception as e:
        return None, f"Error de sintaxis: {str(e)}"

# ==========================================
#  MÉTODOS 1D
# ==========================================
def simular_hit_or_miss_1d(f, a, b, N):
    x_test = np.linspace(a, b, 1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_eval = np.nan_to_num(f(x_test), nan=0.0)
        
    y_min, y_max = np.min(y_eval), np.max(y_eval)
    y_base = min(0, y_min) * 1.1 if min(0, y_min) < 0 else 0
    y_techo = max(0, y_max) * 1.1
    
    x_rand = np.random.uniform(a, b, N)
    y_rand = np.random.uniform(y_base, y_techo, N)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_eval = f(x_rand)
    
    exitos_pos = (y_rand > 0) & (y_rand <= f_eval)
    exitos_neg = (y_rand < 0) & (y_rand >= f_eval)
    exitos_totales = exitos_pos | exitos_neg
    n_exitos = np.sum(exitos_totales)
    
    area_caja = (b - a) * (y_techo - y_base)
    integral_aprox = area_caja * (n_exitos / N) 
    
    if np.sum(exitos_neg) > np.sum(exitos_pos):
        integral_aprox = -integral_aprox

    return integral_aprox, x_rand, y_rand, exitos_totales, f_eval

def simular_valor_medio_1d(f, a, b, N):
    x_rand = np.random.uniform(a, b, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_eval = f(x_rand)
        f_eval_limpio = f_eval[np.isfinite(f_eval)]
    
    if len(f_eval_limpio) == 0: return 0, x_rand, np.zeros_like(x_rand)
    promedio = np.mean(f_eval_limpio)
    integral = (b - a) * promedio
    return integral, x_rand, f_eval

def simular_convergencia_1d(f, a, b, N):
    x_rand = np.random.uniform(a, b, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_eval = np.nan_to_num(f(x_rand), nan=0.0)
        
    promedios_acumulados = np.cumsum(f_eval) / np.arange(1, N + 1)
    return (b - a) * promedios_acumulados

def integral_gauss(f, a, b, n_puntos=5):
    x, w = leggauss(n_puntos)
    x_trans = 0.5 * (x + 1) * (b - a) + a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_eval = f(x_trans)
    return 0.5 * (b - a) * np.sum(w * f_eval)

# ==========================================
#  ANÁLISIS ESTADÍSTICO (CON I.C. DINÁMICO)
# ==========================================
def analisis_estadistico_1d(f, a, b, N, M, nivel_confianza=0.95):
    """
    Calcula la estadística de M simulaciones con un nivel de confianza configurable.
    """
    x_rand = np.random.uniform(a, b, (M, N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_eval = np.nan_to_num(f(x_rand), nan=0.0)
        
    integrales = (b - a) * np.mean(f_eval, axis=1)
    
    media = np.mean(integrales)
    varianza = np.var(integrales, ddof=1)
    desviacion = np.std(integrales, ddof=1)
    
    # --- CÁLCULO DINÁMICO DEL Z-SCORE ---
    # Si confianza es 0.95, alfa = 0.05. Buscamos el valor Z para 0.975 (dos colas)
    alfa = 1.0 - nivel_confianza
    z_score = norm.ppf(1.0 - (alfa / 2.0))
    
    error_estandar = desviacion / np.sqrt(M)
    ic_inf = media - z_score * error_estandar
    ic_sup = media + z_score * error_estandar
    
    return {
        "muestras": M, "media": media, "varianza": varianza,
        "desviacion": desviacion, "error_estandar": error_estandar,
        "confianza_pct": nivel_confianza * 100,
        "z_score": z_score,
        "ic": (ic_inf, ic_sup), 
        "ancho_ic": ic_sup - ic_inf, 
        "distribucion": integrales
    }

# ==========================================
#  MÉTODOS MULTIVARIABLES (2D y 3D)
# ==========================================
def simular_integral_multiple(f, limites, N, dim=2):
    """
    limites: [(ax, bx), (ay, by)] para 2D, o [(ax, bx), (ay, by), (az, bz)] para 3D.
    Utiliza el método del Valor Promedio.
    Retorna: (integral_aprox, x, y, [z], f_eval, estadistica_dict)
    """
    if dim == 2:
        ax, bx = limites[0]
        ay, by = limites[1]
        x_rand = np.random.uniform(ax, bx, N)
        y_rand = np.random.uniform(ay, by, N)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand, y_rand)
            # Reemplazar NaN e Inf por 0 (no filtrar, mantener N muestras)
            f_eval = np.nan_to_num(f_eval, nan=0.0, posinf=0.0, neginf=0.0)
            
        escala = (bx - ax) * (by - ay)
        promedio = np.mean(f_eval)
        integral = escala * promedio
        
        # Estadística - usar todas las N muestras
        desv_std = np.std(f_eval, ddof=1) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        
        stats = {
            "escala": escala,
            "promedio": promedio,
            "desv_std": desv_std,
            "error_est": error_est,
            "N": N
        }
        
        return integral, x_rand, y_rand, f_eval, stats

    elif dim == 3:
        ax, bx = limites[0]
        ay, by = limites[1]
        az, bz = limites[2]
        x_rand = np.random.uniform(ax, bx, N)
        y_rand = np.random.uniform(ay, by, N)
        z_rand = np.random.uniform(az, bz, N)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand, y_rand, z_rand)
            # Reemplazar NaN e Inf por 0 (no filtrar, mantener N muestras)
            f_eval = np.nan_to_num(f_eval, nan=0.0, posinf=0.0, neginf=0.0)
            
        escala = (bx - ax) * (by - ay) * (bz - az)
        promedio = np.mean(f_eval)
        integral = escala * promedio
        
        # Estadística - usar todas las N muestras
        desv_std = np.std(f_eval, ddof=1) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        
        stats = {
            "escala": escala,
            "promedio": promedio,
            "desv_std": desv_std,
            "error_est": error_est,
            "N": N
        }
        
        return integral, x_rand, y_rand, z_rand, f_eval, stats