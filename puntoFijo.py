import numpy as np

def g(x):
    return np.pi + 0.5 * np.sin(x / 2)

def fixed_point_iteration(x0, tol=1e-5, max_iter=100):
    x = x0
    iter_values = [x0]

    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        print(x_new)

        if abs(x_new - x) < tol:
            print("Tolerance exceeded...")
            break

        x = x_new

    return x_new, iter_values

x0 = 1.5
root, iter_values = fixed_point_iteration(x0)

print("raiz:", root)
print("Iteraciones:", len(iter_values)-1)
