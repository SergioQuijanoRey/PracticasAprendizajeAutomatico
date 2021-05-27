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
    print("==========================================")
    print_full(stats)
    wait_for_user_input()

    # TODO -- falta visualizar los datos en una grafica t-sne

# Preprocesado de los datos
#===============================================================================
def split_train_dataset_into_X_and_Y(df):
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

def remove_outliers(df, times_std_dev):
    """
    Elimina las filas de la matriz representada por df en las que, en alguna columna, el valor de la
    fila esta mas de times_std_dev veces desviado de la media

    Paramters:
    ==========
    df: dataframe sobre el que trabajamos
    times_std_dev: umbral de desviacion respecto a la desviacion estandar
                   Un valor usual es 3.0, porque el 99.74% de los datos de una distribucion normal
                   se encuentran en el intervalo (-3 std + mean, 3 std + mean)

    Returns:
    ========
    cleaned_df: dataframe al que hemos quitado las filas asociadas a los outliers descritos

    TODO -- borra demasiados datos de nuestro dataset
    """
    return df[(np.abs(stats.zscore(df)) < times_std_dev).all(axis=1)]

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
    print(f"Prev cols: {prev_cols}")

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
    df_train_X, df_train_Y = split_train_dataset_into_X_and_Y(df_train) # Separamos datos de entrenamiento
    df_test_X, df_train_Y = split_train_dataset_into_X_and_Y(df_test)   # Separamos datos de test

    print("--> Borrando outliers")
    prev_len = len(df_train_X)
    df_train = remove_outliers(df_train_X, times_std_dev = 4.0)
    print(f"Tamaño tras la limpieza de outliers del train_set: {len(df_train)}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train)}")
    wait_for_user_input()

    print("--> Normalizando dataset")
    # Notar que la variable de salida temperatura critica no la estamos normalizando
    df_train_X, df_test_X = normalize_dataset(df_train_X, df_test_X)

    print("Mostramos las estadisticas de los datos de entrenamiento normalizados")
    explore_training_set(df_train_X)
    wait_for_user_input()
