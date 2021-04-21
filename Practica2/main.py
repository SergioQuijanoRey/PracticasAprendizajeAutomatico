"""
Practica 2: Aprendizaje Automatico
Sergio Quijano Rey, sergioquijano@correo.ugr.es
"""
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Para hacer graficas en 3D
from matplotlib import cm               # Para cambiar el color del grafico 3D

# Funciones auxiliares
# ===================================================================================================


def wait_for_user_input():
    """Esperamos a que el usuario pulse una tecla para continuar con la ejecucion"""
    input("Pulse ENTER para continuar...")

# Funciones dadas por los profesores
# ===================================================================================================


def simula_unif(N, dim, rango):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    media = 0
    out = np.zeros((N, dim), np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1)  # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b


def readData(file_x, file_y, digits, labels):
    """
    Funcion dada por los profesores para leer un fichero de datos
    Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores
    """
    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
    for i in range(0, datay.size):
        if datay[i] == digits[0] or datay[i] == digits[1]:
            if datay[i] == digits[0]:
                y.append(labels[0])
            else:
                y.append(labels[1])
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y

# Graficos
# ===================================================================================================

# Funciones para mostrar graficos
# ===================================================================================================


def scatter_plot(x_values, y_values, title="Scatter Plot Simple", x_label="Eje X", y_label="Eje Y"):
    """
    Grafico simple tipo scatter plot, sin colorear segun clases porque no tenemos clases
    De nuevo, no tenemos clases que separar asi que pintamos todos los puntos del mismo color

    Parameters:
    ===========
    x_values: valores de la coordenada x de los datos
    y_values: valores de la coordenada y de los datos
    title: titulo del grafico
    x_label: etiqueta para el eje x
    y_label: etiqueta para el eje y
    """

    # Tomo estos elementos para hacer graficas elaboradas
    fig, ax = plt.subplots()

    # Muestro el scatter plot de los datos
    # Añado alpha por si los datos se acumulan unos sobre otros, para que esto
    # sea facilmente visible
    ax.scatter(x_values, y_values, c="grey", alpha=0.6)

    # Titulo para la grafica
    plt.title(title)

    # Añado las leyendas en los ejes
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Muestro el grafico
    plt.show()
    wait_for_user_input()

# Ejercicio 1
# ===================================================================================================


def ejercicio1():
    """Codigo que lanza todos los apartados del primer ejercicio"""
    print("Lanzando ejercicio 1")
    print("=" * 80)

    ejercicio1_apartado1()


def ejercicio1_apartado1():
    """Codigo que lanza la tarea del primer apartado del primer ejercicio"""
    # Parametros de la tarea pedida
    number_of_points = 50   # Numero de datos
    dimensions = 2          # Dimensiones de cada dato
    lower = -50             # Extremo inferior del intervalo en cada coordenada
    upper = 50              # Extremo superior del intervalo en cada coordenada
    lower_sigma = 2         # Extremo inferior para el valor de la desviacion
    upper_sigma = 7         # Extremo superior para el valor de la desviacion

    # Generamos los dos conjuntos de datos
    uniform_dataset = simula_unif(
        number_of_points,
        dimensions,
        rango=[lower, upper]
    )
    gauss_dataset = simula_gaus(
        number_of_points,
        dimensions,
        sigma=[lower_sigma, upper_sigma]
    )

    # Mostramos los dos datasets obtenidos
    print("Dataset generado con una distribucion uniforme")
    scatter_plot(uniform_dataset[:, 0], uniform_dataset[:, 1],
                 f"Scatter Plot de la distribucion uniforme de {number_of_points} puntos en el rango [{lower}, {upper}]")

    print("Dataset generado con una distribucion gaussiana")
    scatter_plot(gauss_dataset[:, 0], gauss_dataset[:, 1],
                 f"Scatter Plot de la distribucion gaussiana de {number_of_points} puntos con sigma en [{lower_sigma, upper_sigma}]")


# Funcion principal
# ===================================================================================================
if __name__ == "__main__":
    # Fijamos la semilla para no depender tanto de la aleatoriedad y conseguir resultados
    # reproducibles
    # TODO -- descomentar esto para fijar la semilla aleatoria
    # np.random.seed(123456789)

    # Lanzamos el primer ejercicio
    ejercicio1()
