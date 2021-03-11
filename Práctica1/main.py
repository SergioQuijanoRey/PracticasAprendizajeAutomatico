"""
Practica 1
Sergio Quijano Rey
sergioquijano@correo.ugr.es
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Buenas tardes")


# Funciones Auxiliares / Comunes
#===============================================================================
def wait_for_user_input():
    input("Pulse ENTER para continuar...")

def gradient_descent(starting_point, loss_function, gradient, learning_rate: float = 0.001, max_iterations: int = 100_000, target_error: float = 1e-10):
    current_solution = starting_point
    current_iteration = 0

    error_at_iteration = []

    while current_iteration < max_iterations and loss_function(current_solution[0], current_solution[1]) > target_error:
        current_solution = current_solution - learning_rate * gradient(current_solution[0], current_solution[1])
        current_iteration = current_iteration + 1
        error_at_iteration.append(loss_function(current_solution[0], current_solution[1]))

    return current_solution, current_iteration, error_at_iteration


# Ejercicio 1
#===============================================================================
def ejercicio1():
    print("Ejecutando ejercicio 1")

    # Definimos la funcion de coste
    # Me quedo con el cuerpo del cuadrado porque lo usaremos en las derivadas parciales
    inside = lambda u, v: np.power(u, 3) * np.exp(v - 2) - 2.0 * np.power(v, 2) * np.exp(-u)
    E = lambda u, v: np.power(inside(u, v), 2)

    # Derivadas parciales, cuya expresion hemos calculado a mano y cuyo procedimiento
    # esta reflejado en el documento
    dEu = lambda u, v: 2 * inside(u, v) * (3 * np.power(u, 2) * np.exp(v - 2) + 2 * np.power(v, 2) * np.exp(-u))
    dEv = lambda u, v: 2 * inside(u, v) * (np.power(u, 3) * np.exp(v - 2) - 4 * v * np.exp(-u))

    # Gradiente de la funcion de coste
    gradient = lambda u, v: np.array([dEu(u, v), dEv(u, v)])

    # Mostramos la expresion del gradiente
    # TODO -- escribir la funcion del gradiente

    # Parametros para el gradiente descendente
    learning_rate = 0.1
    max_iterations = 10000000000
    target_error = 1e-14
    starting_point = np.array([1.0,1.0])

    # Lanzamos el descenso y mostramos los resultados
    weights, iterations, error_at_iteration = gradient_descent(starting_point, E, gradient, learning_rate, max_iterations, target_error)

    Y = error_at_iteration
    X = np.arange(0, len(Y))

    plt.title("Evolucion del error")
    plt.xlabel("Iteracion")
    plt.ylabel("Error")
    plt.plot(X, Y)
    plt.show()
    wait_for_user_input()

    print(f"Numero de iteraciones: {iterations}")
    print(f"Pesos encontrados: {weights}")
    wait_for_user_input()


# Corremos todos los ejercicios
#===============================================================================
if __name__ == "__main__":
    ejercicio1()
