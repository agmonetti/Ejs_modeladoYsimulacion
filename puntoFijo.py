import numpy as np

def g(x):
    return np.exp(-x)

def fixed_point_iteration(x0, tol=1e-5, max_iter=100):
    x = x0
    iter_values = [x0]

    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)

        if abs(x_new - x) < tol:
            print("Tolerance reached")
            break

        x = x_new

    return x_new, iter_values

x0 = 0.5
root, iter_values = fixed_point_iteration(x0)

print("raiz:", root)