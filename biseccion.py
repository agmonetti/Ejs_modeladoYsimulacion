import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def biseccion(f, a, b, iteraciones=100, tolerancia=1e-3, precision=5):
    # Verificación inicial
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo.")

    results = []

    for i in range(iteraciones):
        c = (a + b) / 2.0
        fc = f(c)

        results.append([i+1, round(a, precision), round(b, precision), round(c, precision), round(fc, precision)])

        # Condición de parada
        if abs(fc) < tolerancia or (b - a) / 2.0 < tolerancia:
            # Impresión de la matriz de resultados una única vez al alcanzar la convergencia
            print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
            return c

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    # Impresión en caso de no convergencia para análisis de divergencia
    print(tabulate(results, headers=["i", "a", "b", "c", "f(c)"]))
    raise ValueError("El método no convergió en el número máximo de iteraciones.")


## no funciona desde mi visual en arch - desde zsh, grafico visual perfectamente
def graficar_biseccion(f, a, b, raiz):
    # Graficar la función
    x = np.linspace(a - 1, b + 1, 400)
    y = f(x)
    plt.plot(x, y, label='$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Marcador analítico de la raíz
    if raiz is not None:
        plt.plot(raiz, f(raiz), 'ro', label=f'Raíz $\\approx$ {raiz:.5f}')
        
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Instrucción de renderizado y proyección de la ventana gráfica
    plt.show() 

#funcion para encontrar los intervalos dado una funcion x
def buscar_intervalos(f, inicio, fin, paso=0.5):
    """
    Realiza un barrido escalar para encontrar intervalos [a, b] 
    donde la función cambia de signo (f(a) * f(b) < 0).
    """
    intervalos = []
    
    # Generamos el vector de puntos a evaluar
    puntos_x = np.arange(inicio, fin + paso, paso)
    
    for i in range(len(puntos_x) - 1):
        a = puntos_x[i]
        b = puntos_x[i+1]
        
        # Aplicación del Teorema de Bolzano
        if f(a) * f(b) < 0:
            # Guardamos el intervalo redondeado para mayor limpieza visual
            intervalos.append((round(a, 4), round(b, 4)))
            
    return intervalos


def f(x):
    return(x + 2) * (x + 1) * (x - 1)**3 * (x - 2)

# Intervalo inicial
a =-1.5
b =1.75

# Encontrar y mostrar la raíz
raiz = biseccion(f, a, b)
print(f"La raíz encontrada es: {raiz}")
print("Graficando...")
graficar_biseccion(f, a, b, raiz)

#Ejercicios busqueda de raices ppt:
# x**2 -4 -> raiz: 2.000000238418579
# x**3 -x -2 -> raiz: 1.5213804244995117
#x**2 -3 -> raiz: 1.732050895690918
#np.exp(x) - 2 - x -> raiz: 1.146193504333496
# np.cos(x) +1 -x -> raiz: 1.2834291458129883
# np.log(x) +x -5 -> raiz: 3.693441390991211
# x - np.cos(x) -> raiz: 0.0.7390851974487305


#Guia - punto 1:
# item a) np.exp(x) - 2 - x -> hallar intervalo
# Buscamos raíces entre -5 y 5, avanzando de a 0.5
"""
intervalos_encontrados = buscar_intervalos(f, -5, 5, 0.5)
print("Intervalos encontrados para f(x) = np.exp(x) - 2 - x:")
for a, b in intervalos_encontrados:
    print(f"Raíz detectada en el intervalo: [{a}, {b}]")
"""

# punto 3: tolerancia 10⁻3
# a) np.sqrt(x) - np.cos(x), intervalo: [0,1]
# b) x - 2**(-x), intervalo: [0,1]
# c) np.exp(x) - x**2 + 3*x - 2, intervalo: [0,1]
# d) 2*x * np.cos(x) - (x + 1)**2, intervalo: [-3,-2] y [-1,0]
# e) x * np.cos(x) - 2*x**2 + 3*x - 1, intervalo: [0.2,.3] y [1.2,1.3]