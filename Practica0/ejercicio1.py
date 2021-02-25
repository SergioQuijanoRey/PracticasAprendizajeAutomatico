"""Module to implement exercise 3 functionality"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


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
    iris = datasets.load_iris()

    # Separamos caracteristicas de las clases
    data = iris.data
    classes = iris.target

    # Nos quedamos solo con la primera y tercera caracteristica que corresponden
    # a los indices 0 y 2
    data = [data[indx][0:3:2] for indx in range(len(data))]

    return data, classes


def plot_iris_dataset(data, classes):
    """Hacemos un scatter plot de los datos junto a las clases en las que estan divididos"""

    # Separamos los valores de x e y
    x_values = [data[x][0] for x in range(len(data))]
    y_values = [data[x][1] for x in range(len(data))]

    # c = classes colorea los puntos segun las clases a las que pertenezcan
    plt.scatter(x_values, y_values, c = classes)
