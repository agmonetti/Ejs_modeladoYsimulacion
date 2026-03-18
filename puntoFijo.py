import numpy as np

def fixed_point_iteration(x0, tol=1e-6, max_iter=100):
    x = x0
    iter_values = [x0]

    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        print("Iteracion " + str(i) + ": " + str(x_new))

        if abs(x_new - x) < tol:
            print("Tolerance exceeded..." )
            print(f"Converged to {x_new} after {i} iterations.")
            print(f"final error: {abs(x_new - x)}")
            break

        x = x_new

    return x_new, iter_values


def g(x):
    return  x - x*np.exp(-x)

x0 = 0.5
root, iter_values = fixed_point_iteration(x0)

#print(np.pi)

#Listas de g(x) usados:
#0.4 * np.exp(x**2)
#np.exp(-x)
#(1/3) * (2 * np.sqrt(3) + x)
#np.sqrt(np.exp(x) / 3)
#np.pi + 0.5 * np.sin(x / 2)
