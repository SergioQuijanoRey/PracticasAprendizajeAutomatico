import math
import numpy as np
import matplotlib.pyplot as plt

def run():
    """Corre el codigo necesario para resolver el ejercicio 3"""

    # Parametros para el ejercicio
    lower = 0
    # TODO -- cambiar por np.pi
    upper = 4 * math.pi
    number_of_points = 100

    print(
        f"Separando el intervalo [{lower}, {upper}] en {number_of_points} puntos equidistantes")
    values = np.linspace(lower, upper, number_of_points)

    print(f"Mapeando los valores a las tres funciones dadas")
    sin_values, cos_values, complex_function_values = map_values_to_functions(
        values)

    print(f"Los valores son are: {values}")
    print(f"Valores en el seno: {sin_values}\n")
    print(f"Valores en el coseno: {cos_values}\n")
    print(f"Valores en tanh(sin + cos): {complex_function_values}\n")

    print("Mostrando la grafica de los valores")
    plot_three_functions(values, sin_values, cos_values,
                         complex_function_values)

def map_values_to_functions(values):
    """
    Evalua las tres funciones dadas en los valores pasados como parametro
    Devuelve las tres listas sin_values, cos_values, tanh(sin + cos)
    """

    sin_values = [np.sin(x) for x in values]
    cos_values = [np.cos(x) for x in values]
    complex_function_values = [np.tanh(np.sin(x) + np.cos(x)) for x in values]

    return sin_values, cos_values, complex_function_values


def plot_three_functions(values, sin_values, cos_values, complex_function_values):
    """
    Grafica las tres funciones como se especifica en el ejercicio

    Para cambiar los colores y el estilo de linea he leido el siguiente enlace
    de la documentacion oficial de matplotlib:
        https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    """

    # Cambio la escala del eje x a una trigonometrica
    # En el docstring de la funcion indico de donde copio esta funcion
    set_x_axis_scale_to_pi()

    # En verde, con lineas discontinuas
    plt.plot(values, sin_values, "--g")

    # En negro, con lineas discontinuas
    plt.plot(values, cos_values, "--k")

    # En rojo, con lineas discontinuas
    plt.plot(values, complex_function_values, "--r")
    plt.show()

def set_x_axis_scale_to_pi():
    """
    Cambio el eje x a uno basado en fracciones de PI, mejor para graficar
    funciones trigonometricas

    El codigo lo copio de:
        https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
    """

    # Para formatear el eje X con multiplos de PI
    def format_func(value, tick_number):
        # Calcula el numero de multiplos de PI / 2
        N = int(np.round(2 * value / np.pi))

        # Formatea acorde a este multiplo
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    # Toma el objeto ax para hacer manipulaciones complejas del plot
    _, ax = plt.subplots()

    # Coloca los multiplos descritos
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
