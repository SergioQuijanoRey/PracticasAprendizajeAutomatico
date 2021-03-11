from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as pltcols
import numpy as np

# Ejercicio 1
#===============================================================================

def run_ejercicio_1():
    """Ejecuta las funciones necesarias para resolver el primer ejercicio"""

    print("Leyendo datos de la base de datos iris desde scikit-learn")
    data, classes, feature_names = read_iris_data()

    print("Mostrando la grafica de los datos")
    plot_iris_dataset(data, classes, feature_names)


def read_iris_data():
    """
    Toma los datos de la base de datos iris desde scikit learn
    Nos quedamos con las caracteristicas primera y tercera y con las clases
    Para ello he consultado el codigo de la documentacion de scikit learn:
        https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    """

    # Tomamos los datos del dataset
    # Esta es la parte en la que copio codigo de la fuente mencionada
    iris_dataset = datasets.load_iris()

    # Separamos caracteristicas de las clases
    data = iris_dataset.data
    classes = iris_dataset.target
    feature_names = iris_dataset.feature_names

    # Nos quedamos solo con la primera y tercera caracteristica que corresponden
    # a los indices 0 y 2
    data = [data[indx][0:3:2] for indx in range(len(data))]

    return data, classes, feature_names


def plot_iris_dataset(data, classes, feature_names):
    """Hacemos un scatter plot de los datos junto a las clases en las que estan divididos"""

    # Separamos los valores de x e y
    data = np.array(data)
    x_values = data[:, 0]
    y_values = data[:, 1]

    # Tomamos la figura y ejes por separado en vez de hacer manipulaciones directas
    # para poder poner leyendas y otras operaciones complejas
    _, ax = plt.subplots()

    # Tomo los titulos de las caracteristicas y los asigno al grafico
    # Tomo la idea de: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
    x_legend = feature_names[0]
    y_legend = feature_names[1]
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    # Coloreamos las distintas clases segun los colores que se nos ha especificado
    # La parte de hacer ListedColormap la saco de: https://stackoverflow.com/questions/12487060/
    colormap = ['orange', 'black', 'green']
    scatter = ax.scatter(x_values, y_values, c=classes,
                         cmap=pltcols.ListedColormap(colormap))

    # Asignamos las etiquetas a los colores
    # Consultado de la documentacion oficial de matplotlib
    # En concreto: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html#automated-legend-creation
    legend = ax.legend(*scatter.legend_elements(), title="Clases", loc = "upper left")
    ax.add_artist(legend)

    # Ponemos un titulo a la grafica
    plt.title("Grafica de caracteristicas y sus clases")

    # Mostramos el grafico
    plt.show()

# Ejercicio 2
#===============================================================================
def run_ejercicio_2():
    """Corre las operaciones necesarias para resolver el problema del ejercicio 2"""

    print("Ejecutando ejercicio 2")

    # Genero un data_set en formato array de pares para hacer una rapida comprobacion
    # de que la funcion de separacion hace lo que tiene que hacer
    size = 10

    # (size, 2) para crear size entradas en el array de pares aleatorios
    data_set = np.random.uniform(-10, 10, (size, 2))
    data_set = np.array(data_set)

    # Separo los datos con la funcion programada
    training, test = split_data_set_matrix(data_set)

    # Muestro el resultado de separar los datos
    print("FORMA MATRICIAL")
    print("=" * 80)
    print("")

    print(f"El training queda: {training}")
    print("")
    print(f"El test queda: {test}")
    print("")


    # Ahora lo pruebo pero con los datos por separado
    # Para ello me quedo con todos los indices de fila y fijo la columna 0 o 1
    print("FORMA X, Y")
    print("=" * 80)
    print("")

    X = data_set[:,0]
    Y = data_set[:,1]
    print(f"Dataset: {data_set}")
    print(f"X: {X}")
    print(f"Y: {Y}")
    print("")

    # Separo los datos con la funcion programada
    X_training, X_test, Y_training, Y_test = split_data_set_splitted(X, Y)

    # Muestro el resultado de separar los datos
    print("Resultados:")
    print(f"X_training: {X_training}")
    print(f"Y_training: {Y_training}")
    print(f"X_test: {X_test}")
    print(f"Y_test: {Y_test}")



def split_data_set_matrix(data_set):
    """
    Dado un data set en formato matricial lo separa en un 75% para training y un 25%
    para test. Para conservar los elementos de cada clase en test y training se
    mezclan aleatoriamente los datos

    Para encontrar esta funcionalidad, he leido esta pagina de la documentacion
    oficial de scikit learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    # Uso la funcion de scikitlearn para separar el data_set
    # Esta funcion por defecto mezcla los datos para asegurar la representacion
    # de los datos en los dos subconjuntos
    training, test = train_test_split(data_set, train_size = 0.75, test_size = 0.25)
    return training, test

def split_data_set_splitted(X, Y):
    """
    Dado un data set en formato dos arrays, uno con X y otro con Y, lo separa en
    un 75% para training y un 25% para test. Para conservar los elementos de cada
    clase en test y training se mezclan aleatoriamente los datos

    Para encontrar esta funcionalidad, he leido esta pagina de la documentacion
    oficial de scikit learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    # Uso la funcion de scikitlearn para separar el data_set
    # Esta funcion por defecto mezcla los datos para asegurar la representacion
    # de los datos en los dos subconjuntos
    X_training, X_test, Y_training, Y_test= train_test_split(X, Y, train_size = 0.75, test_size = 0.25)
    return X_training, X_test, Y_training, Y_test

# Ejercicio 3
#===============================================================================
def run_ejercicio_3():
    """Corre el codigo necesario para resolver el ejercicio 3"""

    # Parametros para el ejercicio
    lower = 0
    upper = 4 * np.pi
    number_of_points = 100

    print(f"Separando el intervalo [{lower}, {upper}] en {number_of_points} puntos equidistantes")
    values = np.linspace(lower, upper, number_of_points)
    print("")

    print(f"Mapeando los valores a las tres funciones dadas")
    sin_values, cos_values, complex_function_values = map_values_to_functions(values)
    print("")

    print(f"Los valores son: {values}\n")
    print(f"Valores en el seno: {sin_values}\n")
    print(f"Valores en el coseno: {cos_values}\n")
    print(f"Valores en tanh(sin + cos): {complex_function_values}\n")
    print("")

    print("Mostrando la grafica de los valores")
    plot_three_functions(values, sin_values, cos_values, complex_function_values)

def map_values_to_functions(values):
    """
    Evalua las tres funciones dadas en los valores pasados como parametro
    Devuelve las tres listas sin_values, cos_values, tanh(sin + cos)
    """

    sin_values = np.sin(values)
    cos_values = np.cos(values)

    # Defino una nueva funcion anonima que uso para mapear values
    complex_function = lambda x: np.tanh(np.sin(x) + np.cos(x))
    complex_function_values = complex_function(values)

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

    # Pongo un titulo al grafico
    plt.title("Gráfica de las tres funciones")

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

    El codigo lo copio completamente, sin apenas cambios, de:
        https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html

    Lo unico que hago sobre la copia es comentar el codigo y definir una funcion
    dentro de esta funcion para que no quede demasiado sucio el resto de mi codigo
    """

    # Para formatear el eje X con multiplos de PI
    # Defino aqui la funcion para que no ensucie el resto del codigo
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

# Funcion principal del fichero
#===============================================================================
if __name__ == "__main__":
    print("Lanzando todos los ejercicios")
    print("")

    print("Ejercicio 1")
    print("=" * 80)
    run_ejercicio_1()
    print("")

    print("Ejercicio 2")
    print("=" * 80)
    run_ejercicio_2()
    print("")

    print("Ejercicio 3")
    print("=" * 80)
    run_ejercicio_3()
