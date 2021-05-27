"""
Author:
    - Sergio Quijano Rey
    - sergioquijano@correo.ugr.es
Practica 3 - Problema de clasificacion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parametros globales del programa
#===============================================================================
data_file = "./datos/Sensorless_drive_diagnosis.txt"

# Cargado de los datos
#===============================================================================
def load_data():
    """Cargamos los datos usando del problema"""
    df = pd.read_csv(data_file)
    return df_train, df_test

# Preprocesado de los datos
#===============================================================================
def process_data(df):
    # Dataframe que vamos a procesar
    df_proc = df

    # Eliminamos las caracteristicas que tengan una variabilidad practicamente nula
    # Estas caracteristicas son aquellas que tengan una desviacion tipica menor que un epsilon
    # prefijado
    epsilon = 0.35
    stats = pd.DataFrame()
    stats["var"] = df_proc.std()
    bad_elements = stats[(stats < epsilon).any(1)]
    print(bad_elements)


# Funcion principal
#===============================================================================
if __name__ == "__main__":
    # Cargamos los datos del problema
    df_train, df_test = load_data()

    # Preprocesamos los datos del problema
    df_train = process_data(df_train)
