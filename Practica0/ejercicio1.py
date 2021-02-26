"""Module to implement exercise 3 functionality"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcols


def run():
    print("Leyendo datos de la base de datos iris desde scikit-learn")
    data, classes = read_iris_data()
    plot_iris_dataset(data, classes)


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

    # Nos quedamos solo con la primera y tercera caracteristica que corresponden
    # a los indices 0 y 2
    data = [data[indx][0:3:2] for indx in range(len(data))]

    return data, classes


def plot_iris_dataset(data, classes):
    """Hacemos un scatter plot de los datos junto a las clases en las que estan divididos"""

    # Separamos los valores de x e y
    x_values = [data[x][0] for x in range(len(data))]
    y_values = [data[x][1] for x in range(len(data))]

    # Tomamos la figura y ejes por separado en vez de hacer manipulaciones directas
    # para poder poner leyendas y otras operaciones complejas
    _, ax = plt.subplots()

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
