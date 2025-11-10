# =============================================================
# TAREA DE OPTIMIZACIÓN - IMPLEMENTACIONES
# Función: f(x, y) = 0.1 * (exp(x^2 + y^2) - tan(1.5 * sin(x + y)))
# Métodos: Descenso del Gradiente y Quasi-Newton (BFGS)
# =============================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Definición de la función objetivo y su gradiente
# -------------------------------------------------------------
def f(xy):
    """Función objetivo f(x, y)."""
    x, y = xy
    return 0.1 * (np.exp(x**2 + y**2) - np.tan(1.5 * np.sin(x + y)))

def grad_f(xy):
    """Gradiente de f(x, y) = [df/dx, df/dy]."""
    x, y = xy
    exp_term = np.exp(x**2 + y**2)
    trig_term = np.cos(x + y) * (1 / np.cos(1.5 * np.sin(x + y)))**2
    dfdx = 0.1 * (2 * x * exp_term - 1.5 * trig_term)
    dfdy = 0.1 * (2 * y * exp_term - 1.5 * trig_term)
    return np.array([dfdx, dfdy])

# -------------------------------------------------------------
# 2. Descenso del Gradiente
# -------------------------------------------------------------
def gradient_descent(f, grad_f, x0, alpha=1e-3, tol=1e-6, max_iter=10000):
    """Implementación del descenso del gradiente con paso fijo."""
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x -= alpha * g
        path.append(x.copy())
    return x, f(x), i+1, np.array(path)

# -------------------------------------------------------------
# 3. Método Quasi-Newton (BFGS)
# -------------------------------------------------------------
def quasi_newton_bfgs(f, grad_f, x0, tol=1e-6, max_iter=1000):
    """Implementación básica del método Quasi-Newton BFGS."""
    x = np.array(x0, dtype=float)
    n = len(x)
    B = np.eye(n)
    path = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        # Dirección de búsqueda
        p = -np.linalg.solve(B, g)
        # Búsqueda de línea simple (paso adaptativo)
        alpha = 1.0
        while f(x + alpha*p) > f(x) + 1e-4 * alpha * np.dot(g, p):
            alpha *= 0.5
        
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - g
        
        if np.dot(y, s) > 1e-10:  # Evita divisiones por cero
            Bs = B @ s
            B += np.outer(y, y) / np.dot(y, s) - np.outer(Bs, Bs) / np.dot(s, Bs)
        
        x = x_new
        path.append(x.copy())
    
    return x, f(x), k+1, np.array(path)

# -------------------------------------------------------------
# 4. Pruebas y Comparación en la región [-100, 100]^2
# -------------------------------------------------------------
x0 = np.array([1.0, 1.0])   # punto inicial dentro del dominio

# Descenso del gradiente
x_gd, f_gd, it_gd, path_gd = gradient_descent(f, grad_f, x0, alpha=1e-3)
# Quasi-Newton (BFGS)
x_bfgs, f_bfgs, it_bfgs, path_bfgs = quasi_newton_bfgs(f, grad_f, x0)

print("=== Resultados ===")
print(f"Descenso del Gradiente -> x* = {x_gd}, f(x*) = {f_gd:.6f}, iteraciones = {it_gd}")
print(f"Quasi-Newton (BFGS)    -> x* = {x_bfgs}, f(x*) = {f_bfgs:.6f}, iteraciones = {it_bfgs}")

# -------------------------------------------------------------
# 5. Visualización del recorrido de los algoritmos
# -------------------------------------------------------------
# Crear un mapa de calor de la función en un rango pequeño (para evitar overflow)
x_vals = np.linspace(-2, 2, 200)
y_vals = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 0.1 * (np.exp(X**2 + Y**2) - np.tan(1.5 * np.sin(X + Y)))

plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x,y)')

plt.plot(path_gd[:,0], path_gd[:,1], 'r.-', label='Gradiente')
plt.plot(path_bfgs[:,0], path_bfgs[:,1], 'w.-', label='BFGS')
plt.scatter(0,0, c='cyan', marker='*', s=150, label='Origen (referencia)')
plt.title("Trayectorias de optimización de f(x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
