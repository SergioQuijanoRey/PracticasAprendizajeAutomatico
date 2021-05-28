"""
Author:
    - Sergio Quijano Rey
    - sergioquijano@correo.ugr.es
Practica 3 - Problema de regresion

TODO:
    [ ] Comprobar si los datos estan balanceados
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from core import *

# Parametros globales del programa
#===============================================================================
file_path = "./datos/train.csv"

# Carga y exploracion de los datos
#===============================================================================
def load_data_csv(file_path: str):
    """
    Carga todos los datos de un fichero en formato csv con heading

    Parameters:
    ===========
    file_path: fichero del que se toman los datos, en el formato especificado

    Returns:
    ========
    df: Pandas Dataframe con todos los datos cargados. No se ha separado en train / test

    """
    df = pd.read_csv(file_path)
    return df

def split_data(df, test_percentage):
    """
    Separamos el dataset en un conjunto de entrenamiento y en otro conjunto de test

    Parameters:
    ===========
    df: dataframe en el que tenemos almacenados todos los datos
    test_percentage: tanto por uno que queremos que sea asignado al conjunto de test

    Returns:
    ========
    df_train: dataframe con los datos de entrenamiento
    df_test: dataframe con los datos de test
    """

    # TODO -- el stratify no debe ser None -- le voy a preguntar al profesor
    df_test, df_train = train_test_split(df, test_size = test_percentage, shuffle = True, stratify = None)

    return df_test, df_train

def explore_training_set(df):
    """
    Muestra caracteristicas relevantes del dataset de entrenamiento

    Parameters:
    ===========
    df: dataframe del que queremos realizar la exploracion
        No debe contener datos de test, pues no queremos visualizarlos

    Returns:
    ========
    stats: dataframe con las estadisticas calculadas

    """

    # Extreamos algunas estadisticas de este dataframe
    stats = calculate_stats(df)

    print("Estadisticas del dataset de entrenamiento:")
    print_bar()
    print_full(stats)
    wait_for_user_input()

    print("Grafica de cajas de la variable de salida")
    print_bar()
    plot_boxplot(df, columns=["critical_temp"], title="Grafico de cajas de la variable de salida Tc")

    # TSNE -> No aporta informacion relevante en la exploracion de los datos
    #  print("Mostrando grafica tsne --> Puede consumir mucho tiempo de computo")
    #  df_X, _ = split_train_dataset_into_X_and_Y(df)  # Me quedo solo con los datos X, Y lo ignoro
    #  plot_tsne(df_X, perplexity = 10)
    #  plot_tsne(df_X, perplexity = 50)
    #  plot_tsne(df_X, perplexity = 100)

# Preprocesado de los datos
#===============================================================================
def split_dataset_into_X_and_Y(df):
    """
    Tenemos un dataframe con las variables dependientes y la variable dependiente. Esta funcion los
    separa en un dataframe para cada tipo de variable

    Parameters:
    ===========
    df: dataframe con los datos

    Returns:
    ========
    df_X: dataframe con las variables dependientes
    df_Y: dataframe con las variables a predecir (en este caso, una unica variable)
    """

    return df.loc[:, df.columns != "critical_temp"], df["critical_temp"]

# TODO -- creo que esta mal programado
def merge_dataset_from_X_and_Y(df_X, df_Y):
    """
    Operacion inversa a split_dataset_into_X_and_Y, junta el dataframe de caracteristicas de
    prediccion y el dataframe de variables de salida en uno unico

    Parameters:
    ===========
    df_X: dataframe de caracteristicas de prediccion
    df_Y: dataframe de variables de salida

    Returns:
    ========
    df: dataframe con los dos dataframes correctamente juntados
    """

    # Tomamos los datos de las variables de prediccion
    df = df_X

    # Añadimos los datos de las variables de salida
    try:
        for col in df_Y.columns:
            df[col] = df_Y[col]

    # Tenemos un pd.Series en vez de un pd.DataFrame
    except:
        # TODO -- que hacen los {left, right}_index???
        df = df.merge(df_Y.to_frame(), left_index = True, right_index = True)

    return df

def remove_outliers(df, times_std_dev, output_cols = []):
    """
    Elimina las filas de la matriz representada por df en las que, en alguna columna, el valor de la
    fila esta mas de times_std_dev veces desviado de la media

    Paramters:
    ==========
    df: dataframe sobre el que trabajamos
    times_std_dev: umbral de desviacion respecto a la desviacion estandar
                   Un valor usual es 3.0, porque el 99.74% de los datos de una distribucion normal
                   se encuentran en el intervalo (-3 std + mean, 3 std + mean)
    output_cols: columnas de salida, sobre las que no queremos comprobar los outliers

    Returns:
    ========
    cleaned_df: dataframe al que hemos quitado las filas asociadas a los outliers descritos
    """
    # Quitamos las columnas de salida al dataframe. Se usa para la siguiente linea en la que hacemos
    # la seleccion
    df_not_output = df

    # Filtramos las columnas, columna por columna
    for col in output_cols:
        df_not_output = df_not_output.loc[:, df_not_output.columns != col]

    # Filtramos los outliers, sin tener en cuenta las columnas de variables de salida
    return df[(np.abs(stats.zscore(df_not_output)) < times_std_dev).all(axis=1)]

def normalize_dataset(train_df, test_df):
    """
    Normaliza el dataset, usando solo la informacion de los datos de entrenamiento. A los datos de
    test se les aplica la misma transformacion. Notar que no se esta usando informacion de la
    muestra de test para aplicar la normalizacion. Pero pasamos el conjunto de test para aplicar
    la misma trasnformacion a estos datos

    Parameters:
    ===========
    train_df: dataframe de datos de entrenamiento, de los que se calculan los estadisticos para la
              transformacion
    test_df: dataframe de datos de test. No se toma informacion de esta muestra para calcular la
             trasnformacion

    Returns:
    ========
    normalized_train: dataframe con los datos de entrenamiento normalizado
    normalized_test: dataframe con los datos de test normalizados con la misma transformacion
                     calculada a partir de los datos de entrenamiento
    """
    # Guardamos los nombres de las columna del dataframe, porque la tranformacion va a hacer que
    # perdamos este metadato
    prev_cols = train_df.columns

    scaler = StandardScaler()
    normalized_train = scaler.fit_transform(train_df)
    normalized_test = scaler.transform(test_df)

    # La transformacion devuelve np.arrays, asi que volvemos a dataframes
    normalized_train = pd.DataFrame(normalized_train, columns = prev_cols)
    normalized_test = pd.DataFrame(normalized_test, columns = prev_cols)

    return normalized_train, normalized_test

# Funcion principal
#===============================================================================
if __name__ == "__main__":
    print("==> Carga de los datos")
    df = load_data_csv(file_path)
    print(f"Tamaño de todo el dataset: {len(df)}")
    wait_for_user_input()

    print("==> Separamos training y test")
    # 20% del dataset a test
    df_train, df_test = split_data(df, 0.2)
    print(f"Tamaño del set de entrenamiento: {len(df_train)}")
    print(f"Tamaño del set de test: {len(df_test)}")
    wait_for_user_input()

    # Mostramos las caracteristicas de los datos
    print("==> Exploracion de los datos de entrenamiento")
    explore_training_set(df_train)

    print("==> Procesamiento de los datos")
    print("--> Separamos el dataframe de variables X y el dataframe de variable Y")
    df_train_X, df_train_Y = split_dataset_into_X_and_Y(df_train) # Separamos datos de entrenamiento
    df_test_X, df_train_Y = split_dataset_into_X_and_Y(df_test)   # Separamos datos de test

    # TODO -- cuidado -- puedo estarme cargando datos de forma sesgada
    print("--> Borrando outliers")
    prev_len = len(df_train_X)

    # No borramos los outliers en la variable de salida. Precisamente son los valores que nos
    # interesan en la aplicacion practica
    df_train = remove_outliers(df_train, times_std_dev = 4.0, output_cols = ["critical_temp"])
    print(f"Tamaño tras la limpieza de outliers del train_set: {len(df_train)}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train)}")
    print(f"Porcentaje de filas eliminadas: {float(prev_len - len(df_train)) / float(prev_len) * 100.0}%")
    wait_for_user_input()

    print("--> Normalizando dataset")
    # Notar que la variable de salida temperatura critica no la estamos normalizando
    df_train_X, df_test_X = normalize_dataset(df_train_X, df_test_X)

    print("Mostramos las estadisticas de los datos de entrenamiento normalizados")
    explore_training_set(merge_dataset_from_X_and_Y(df_train_X, df_train_Y)) # Junto los dos dataframes para ser pasados como parametro
    wait_for_user_input()
