"""
Practica 1
Sergio Quijano Rey
sergioquijano@correo.ugr.es
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Funciones Auxiliares / Comunes
#===============================================================================
def wait_for_user_input():
    input("Pulse ENTER para continuar...")

# Algoritmos
#===============================================================================
def gradient_descent(starting_point, loss_function, gradient, learning_rate: float = 0.001, max_iterations: int = 100_000, target_error: float = 1e-10, verbose: bool = False):
    """
    Implementa el algoritmo de batch gradient descent

    Si verbose == true, entonces guardamos los errores y soluciones obtenidas en cada iteracion
    """
    current_solution = starting_point
    current_iteration = 0

    # Guardamos errores y soluciones de las iteraciones cuando verbose == True
    error_at_iteration = None
    solution_at_iteration = None
    if verbose == True:
        error_at_iteration = [loss_function(current_solution[0], current_solution[1])]
        solution_at_iteration = [starting_point]

    while current_iteration < max_iterations and loss_function(current_solution[0], current_solution[1]) > target_error:
        current_solution = current_solution - learning_rate * gradient(current_solution[0], current_solution[1])
        current_iteration = current_iteration + 1

        if verbose == True:
            error_at_iteration.append(loss_function(current_solution[0], current_solution[1]))
            solution_at_iteration.append(current_solution)

    return current_solution, current_iteration, np.array(error_at_iteration), np.array(solution_at_iteration)

def gradient_descent_and_plot_error(starting_point, loss_function, gradient, learning_rate: float = 0.001, max_iterations: int = 100_000, target_error: float = 1e-10):
    """
    Ejecutamos el gradiente descendente y mostramos la grafica de la evolucion del error
    Para no repetir muchas veces el mismo codigo en el que lanzamos el algoritmo y
    mostramos la grafica de la evolucion del error
    """

    print(f"Corriendo el algoritmo de descenso del gradiente para eta = {learning_rate}")
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent(starting_point, loss_function, gradient, learning_rate, max_iterations, target_error, verbose = True)

    # Mostramos la grafica de descenso del error
    Y = error_at_iteration
    X = np.arange(0, len(Y))

    plt.title(f"Evolucion del error para eta = {learning_rate}")
    plt.xlabel("Iteracion")
    plt.ylabel("Error")
    plt.plot(X, Y)
    plt.show()
    wait_for_user_input()

    # Por si necesitamos realizar otras operaciones con los resultados
    return weights, iterations, error_at_iteration, solution_at_iteration


# Ejercicio 1
#===============================================================================
def ejercicio1_apartado2():
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

    # Mostramos la expresion de la funcion de coste y la expresion del gradiente
    print("La funcion de coste es:")
    print("\t(u³ * exp(v−2) −2v² exp(−u))²")
    print("La derivada parcial respecto de u es:")
    print("\t2 * (u³ * exp(v−2) −2v² exp(−u)) * (3u² * exp(v-2) + 2v²exp(-u)")
    print("La derivada parcial respecto v es:")
    print("\t2* (u³ * exp(v−2) −2v² exp(−u)) * (u³ * exp(v-2) -4vexp(-u))")
    print("")
    wait_for_user_input()

    # Mostramos la grafica de la funcion de error
    print("Mostrando grafica de la funcion de error")
    birds_eye_loss_plot(E, -5, 5, -5, 5, 1000)
    print("")

    # Parametros para el gradiente descendente
    learning_rate = 0.1
    max_iterations = 10000000000
    target_error = 1e-14
    starting_point = np.array([1.0, 1.0])

    # Lanzamos el descenso y mostramos la grafica de evolucion del error
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, E, gradient, learning_rate, max_iterations, target_error)

    # Mostramos algunos datos numericos del resultado
    print("Resultados:")
    print(f"\tNumero de iteraciones: {iterations}")
    print(f"\tPesos encontrados: {weights}")
    wait_for_user_input()

    # Cuantas iteraciones tarda el algoritmo en obtener un valor de E inferior a 10e-14
    # Hacemos indixes[0] para quedarnos con la lista de indices que devuelve, e
    # indixes[0][0] para quedarnos con el primer elemento de dicha lista
    indixes = np.where(error_at_iteration < 10e-14)
    first_index_under_error = indixes[0][0]
    print(f"La primera iteracion en la que el error esta por debajo de 10e-14 es: {first_index_under_error}")
    print(f"Las primeras coordenadas que estan por debajo de ese error: {solution_at_iteration[first_index_under_error]}")
    print("")
    wait_for_user_input()

def ejercicio1_apartado3():
    # Funcion de perdida que nos dan y calculo sus derivadas parciales
    # No muestro por pantalla las expresiones porque el enunciado no lo pide
    f = lambda x, y: np.power(x + 2.0, 2.0) + 2.0 * np.power(y - 2.0, 2.0) + 2 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    dfx = lambda x, y: 2.0 * (x - 2.0) + 4.0 * np.pi * np.sin(2.0 * np.pi * y) * np.cos(2.0 * np.pi * x)
    dfy = lambda x, y: 4.0 * (y - 2.0) + 4.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    gradient = lambda x, y: np.array([dfx(x, y), dfy(x, y)])

    # Muestro la funcion de error porque tuve algunos problemas con las graficas
    # de errores (no bajaba el error de forma consistente) y necesitaba visualizar
    # la forma de la funcion
    print("Mostrando la grafica de la funcion de error")
    birds_eye_loss_plot(f, -5, 5, -5, 5, 1_000)
    print("")

    # Parametros para el gradiente descendiente
    # No se especifica el error que hay que alcanzar asi que lo pongo a cero
    # para que se consuman las iteraciones
    learning_rate = 0.01
    max_iterations = 50
    target_error = 0
    starting_point = np.array([-1.0,1.0])

    # Lanzamos el descenso y mostramos la grafica con los resultados
    gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)

    # Realizamos el mismo proceso pero modificando el valor del learning rate
    learning_rate = 0.1
    gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)

    # Ahora buscamos los errores minimos y valores de la solucion cuando partimos
    # desde distintos valores de partida

    # Como no se especifica nada en el enunciado del ejercicio, establezco el
    # learning rate, el numero maximo de iteraciones y el target_error
    learning_rate = 0.01
    max_iterations = 10_000
    target_error = 1e-15

    # Muestro explicitamente que estos valores los he fijado yo al no tener indicaciones
    print("Fijo los siguientes parametros para el gradiente descendente:")
    print(f"\tlearning_rate: {learning_rate}")
    print(f"\tmax_iterations: {max_iterations}")
    print(f"\ttarget_error: {target_error}")
    print("")
    wait_for_user_input()

    # Vector con los puntos de partida que se nos piden
    # Para recorrerlo comodamente en un bucle y evitar repitir codigo
    starting_points = [
        [-0.5, -0.5],
        [1, 1],
        [2, 1],
        [-2, 1],
        [-3, 3],
        [-2, 2]
    ]

    for starting_point in starting_points:
        starting_point = np.array(starting_point)
        weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)
        print(f"Resultados para starting_point: {starting_point}")
        print(f"\t(x, y): {weights}")
        print(f"\tError final: {error_at_iteration[-1]}")
        wait_for_user_input()

def birds_eye_loss_plot(loss_function, lower_x: float = -1, upper_x: float = 1, lower_y: float = -1, upper_y: float = 1, points_pers_axis: int = 1000):
    """
    Muestro la grafica del error con un codigo de colores en 2D
    Para poder estudiar lo que pasa cuando corremos ciertos algoritmos de descenso del gradiente de forma intuitiva

    Consulto la funcion de la grafica de la documentacion oficial de matplotlib:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
    y de aqui resuelvo mi duda sobre como saber que magnitud representa cada color:
        https://matplotlib.org/3.3.4/gallery/images_contours_and_fields/contourf_demo.html
    """

    # Valores de las variables independientes
    X_values = np.linspace(lower_x, upper_x, points_pers_axis)
    Y_values = np.linspace(lower_y, upper_y, points_pers_axis)

    # Matriz con los valores de los errores segun las dos variables independientes
    # Hago un bucle for sobre la matriz inicializada a ceros porque no encuentro
    # otra forma de hacer el mapeo que busco
    loss_values_matrix = np.zeros(shape = (points_pers_axis, points_pers_axis))
    for x_index in range(0, len(X_values)):
        for y_index in range(0, len(Y_values)):
            # Hago los indices invertidos porque el primer indice mueve la fila
            # (mueve la direccion vertical) y el segundo indice mueve la columna
            # (mueve la direccion horizontal)
            loss_values_matrix[y_index][x_index] = loss_function(X_values[x_index], Y_values[y_index])

    # Mostramos la grafica de los puntos
    plt.title("Funcion de error")
    plt.contourf(X_values, Y_values, loss_values_matrix)

    # Para poder ver las magnitudes que representa cada color en la grafica
    plt.colorbar()

    plt.show()
    wait_for_user_input()

def ejercicio1():
    print("Ejecutando ejercicio 1")

    print("Apartado 2)")
    print("=" * 80)
    ejercicio1_apartado2()

    print("Apartado 3)")
    print("=" * 80)
    ejercicio1_apartado3()

# Corremos todos los ejercicios
#===============================================================================
if __name__ == "__main__":
    ejercicio1()
