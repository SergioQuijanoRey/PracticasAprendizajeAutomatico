"""
Author:
    - Sergio Quijano Rey
    - sergioquijano@correo.ugr.es
Practica 3 - Funciones comunes a los dos problemas: el de clasificacion y el de regresion
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def wait_for_user_input():
    """Esperamos a que el usuario pulse una tecla para continuar"""
    input("Pulse una tecla para CONTINUAR...")

def print_full(df):
    """
    Muestra todos los datos de un pandas.DataFrame
    Codigo obtenido de
        https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
    """
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def print_bar(car = "=", width = 80):
    """Muestra por pantalla una barra horizontal"""
    print(car * width)

# Estadisticas
#===============================================================================
def calculate_stats(df):
    """
    Calcula un nuevo dataframe con las estadisticas relevantes del dataframe pasado como parametro

    Parameters:
    ===========
    df: dataframe del que queremos calcular algunas estadisticas

    Returns:
    ========
    stats: dataframe con las estadisticas calculadas
    """
    stats = pd.DataFrame()
    stats["type"] = df.dtypes
    stats["mean"] = df.mean()
    stats["median"] = df.median()
    stats["var"] = df.var()
    stats["std"] = df.std()
    stats["min"] = df.min()
    stats["max"] = df.max()
    stats["p25"] = df.quantile(0.25)
    stats["p75"] = df.quantile(0.75)

    # Considero missing value algun valor que se null o que sea NaN (Not a Number)
    stats["missing vals"] = df.isnull().sum() + df.isna().sum()

    return stats

# Graficos
#===============================================================================
def plot_tsne(df, perplexity = 50):
    """
    Realiza una grafica tsne. Potencialmente tardara mucho tiempo en realizar los computos

    Codigo extraido a partir de la documentacion oficial de sklearn:
        https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py

    Parameters:
    ===========
    df: dataframe con los datos
    perplexity: valor del parametro que influye enormemente en la topologia de la grafica obtenida
    """

    # Parametros para el trasnformador
    tsne = manifold.TSNE(n_components=2, init='random', perplexity=perplexity)

    # Guardamos los datos trasnformados
    Y = tsne.fit_transform(df)

    # Hacemos el scatter plot de los datos
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()
    wait_for_user_input()

def plot_boxplot(df, columns, title):
    """
    Hacemos graficos de barras de un dataframe

    Parameters:
    ===========
    df: el dataframe que contiene los datos
    columns: lista con los nombres de las columnas de las que queremos hacer la grafica
    title: titulo de la grafica
    """

    boxplot = df.boxplot(column=columns)
    plt.title(title)
    plt.show()
    wait_for_user_input()
