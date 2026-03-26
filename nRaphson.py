import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def derivative(f, x, dx=1e-6):
    """Aproximación de la derivada usando diferencias centrales"""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)

def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-6, precision=10):
    x = valor_inicial
    results = []   
    
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(derivative(f, x,), precision)
        
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
            
        x_new = round(x - fx / dfx, precision)
        error_abs = abs(x_new - x)
        
        results.append([i, x, fx, dfx, x_new, round(error_abs, precision)])
        
        if error_abs < tolerancia:
            # Se invoca la impresión una sola vez al confirmar la convergencia
            print(tabulate(results, headers=["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error"], tablefmt="grid"))
            return x_new
            
        x = x_new
        
    # Se invoca la impresión una sola vez en caso de divergencia o falta de iteraciones
    print(tabulate(results, headers=["Iteración", "x", "f(x)", "f'(x)", "Resultado", "Error"], tablefmt="grid"))
    raise ValueError("El método no convergió o faltan iteraciones.")

def graficar(f, raiz):
    # Graficar la función
    x = np.linspace(0, 3, 400)
    y = f(x)
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Marcar la raíz encontrada
    plt.plot(raiz, f(raiz), 'ro', label=f'Raíz: x = {raiz:.5f}')
    
    # Añadir leyenda y mostrar la gráfica
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gráfica de la función y su raíz')
    plt.show()



# Definir la función para la cual quieres encontrar la raíz
def f(x):
    return x**3 -x-4

# Valor inicial
valor_inicial = 1

# Encontrar la raíz utilizando el método de Newton-Raphson
raiz = newton_raphson(f, valor_inicial)

# Imprimir la raíz encontrada
print(f"La raíz encontrada es: {raiz}")

# Graficar la función y la raíz
graficar(f, raiz)

