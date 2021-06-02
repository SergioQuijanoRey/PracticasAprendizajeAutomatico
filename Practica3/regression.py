"""
Author:
    - Sergio Quijano Rey
    - sergioquijano@correo.ugr.es
Practica 3 - Problema de regresion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# TODO -- borrar -- debemos pasar esto al principio de este archivo
from core import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

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

def explore_training_set(df, show_box_plot = True):
    """
    Muestra caracteristicas relevantes del dataset de entrenamiento

    Parameters:
    ===========
    df: dataframe del que queremos realizar la exploracion
        No debe contener datos de test, pues no queremos visualizarlos
    show_box_plot: indica si queremos mostrar el boxplot de la variable de salida o no

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

    if show_box_plot == True:
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

def append_series_to_dataframe(dataframe, series, column_name):
    """
    Añade un pandas.Series a un pandas.Dataframe

    Se usa como operacion inversa a split_dataset_into_X_and_Y, junta el dataframe de caracteristicas de
    prediccion y el dataframe de variables de salida en uno unico

    Parameters:
    ===========
    dataframe: dataframe con toda la matriz de datos
               Sera el dataframe con las caracteristicas de entrada
    series: pandas.Series con la columna que queremos añadir
            Sera la columna con la caracteristica a predecir
    column_name: el nombre de la columna que queremos añadir

    Returns:
    ========
    df: dataframe con los datos correctamente juntados
    """

    df = dataframe.copy()
    df[column_name] = series.copy()

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

def check_outliers_removal(df_train_original, df_train_cleaned):
    """
    Comprueba cómo estamos eliminando los outliers en los datos de entrada. Eliminando outliers
    en los datos de entrada, podemos estar sesgando la muestra. Por ejemplo, eliminando todas las
    filas que tienen un valor alto de Tc, que en este caso, son de las que mas nos interesan

    Parameters:
    ===========
    df_train_original: dataframe original, sin eliminar outliers
    df_train_cleaned: dataframe al que se han borrado los outliers
    """

    # Calculamos el dataframe de las filas que hemos borrado
    # Uso isin, que devuelve NaN si no se encuentra, por lo que despues aplico dropna
    # El simbolo ~ hace el not, que es lo que quiero calcular (elementos en la tabla original que
    # no se encuentran en la tabla limpiada)
    df_diff = df_train_original[~df_train_original.isin(df_train_cleaned)]
    df_diff = df_diff.dropna()

    # Solo me interesa ver que valores toma la variable de salida
    df_diff = df_diff["critical_temp"]
    df_diff_stats = calculate_stats(df_diff.to_frame())

    print_full(df_diff_stats)
    wait_for_user_input()

def standarize_dataset(train_df, test_df):
    """
    Estandariza el dataset, usando solo la informacion de los datos de entrenamiento. A los datos de
    test se les aplica la misma transformacion. Notar que no se esta usando informacion de la
    muestra de test para aplicar la estandarizacion. Pero pasamos el conjunto de test para aplicar
    la misma trasnformacion a estos datos

    Parameters:
    ===========
    train_df: dataframe de datos de entrenamiento, de los que se calculan los estadisticos para la
              transformacion
    test_df: dataframe de datos de test. No se toma informacion de esta muestra para calcular la
             trasnformacion

    Returns:
    ========
    standarized_train: dataframe con los datos de entrenamiento estandarizados
    standarized_test: dataframe con los datos de test estandarizados con la misma transformacion
                     calculada a partir de los datos de entrenamiento
    """
    # Guardamos los nombres de las columna del dataframe, porque la tranformacion va a hacer que
    # perdamos este metadato
    prev_cols = train_df.columns

    scaler = StandardScaler()
    standarized_train = scaler.fit_transform(train_df)
    standarized_test = scaler.transform(test_df)

    # La transformacion devuelve np.arrays, asi que volvemos a dataframes
    standarized_train = pd.DataFrame(standarized_train, columns = prev_cols)
    standarized_test = pd.DataFrame(standarized_test, columns = prev_cols)

    return standarized_train, standarized_test

def apply_PCA(df_train_X, df_test_X, explained_variation = 0.90, number_components = None):
    """
    Aplica PCA al conjunto de entrada de los datos de entrenamiento

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada de entrenamiento
                Importante: no debe contener la variable de salida
    df_test_X: dataframe con los datos de entrada de test. Solo los queremos para aplicar la misma
               transformacion que a los datos de entrada. No los usamos en el proceso de calcular
               la trasnformacion
    explained_variation: varianza explicada por los datos transformados que queremos alcanzar
                         Se aplica solo cuando number_components == None
    number_components: numero de componentes que queremos obtener, independientemente de la varianza
                       explicada obtenida. Es opcional

    Returns:
    ========
    df_transformed_X: datos de entrenamiento transformados
    df_test_transformed_X: datos de test transformados usando la misma transformacion calculada a
                           partir de los datos de entrenamiento
    """

    # Comprobacion de seguridad
    if type(explained_variation) is not float:
        raise Exception("El porcentaje de variabilidad explicada debe ser un flotante")

    # Si tenemos numero de componentes, no hacemos caso a la varianza explicada
    pca = None
    if number_components is not None:
        pca = PCA(number_components)
    else:
        # Queremos que PCA saque tantas dimensiones como porcentaje de variacion explidada especificado
        pca = PCA(explained_variation)

    # Nos ajustamos a la muestra de datos de entrenamiento
    print("Ajustando los datos de entrenamiento a la transformacion")
    pca.fit(df_train_X)

    # Aplicamos la transformacion al conjunto de entrenamiento y de test
    # Usamos variables para que no se modifiquen los dataframes pasados como parametro
    df_transformed_X = pca.transform(df_train_X)
    df_test_transformed_X = pca.transform(df_test_X)

    # Recuperamos los datos en formato dataframe
    # No podemos ponerle nombres a las columnas porque PCA mezcla las columnas sin tener nosotros
    # control sobre como se hace la transformacion
    df_transformed_X = pd.DataFrame(df_transformed_X)
    df_test_transformed_X = pd.DataFrame(df_test_transformed_X)

    # Mostramos algunos datos por pantalla
    print(f"Ajuste realizado:")
    print(f"\tPorcentaje de la varianza explicado: {pca.explained_variance_ratio_}")
    print(f"\tPorcentaje de la varianza explicado total: {sum(pca.explained_variance_ratio_)}")
    print(f"\tNumero de dimensiones obtenidas: {len(df_transformed_X.columns)}")
    wait_for_user_input()

    return df_transformed_X, df_test_transformed_X

def show_results(coeffs, df_train_X, df_train_Y, df_test_X, df_test_Y):
    """
    Muestra los resultados del entrenamiento

    Parameters:
    ===========
    coeffs: coeficientes que definen el hiperplano
    df_train_X: conjunto de entrenamiento, variables de entrada
    df_train_Y: conjunto de entrenamiento, variable de salida
    df_test_X: conjunto de test, variables de entrada
    df_test_Y: conjunto de test, variable de salida
    """

    # Computamamos las predicciones en la muestra de entrenamiento
    f = lambda x: np.dot(x, coeffs)
    predictions = f(df_train_X.to_numpy())

    # Calculamos el error cuadratico medio en la muestra
    in_sample_error = mean_squared_error(predictions, df_train_Y.to_numpy())

    # Computamos las predicciones en el test y calculamos Etest
    predictions = f(df_test_X.to_numpy())
    test_sample_error = mean_squared_error(predictions, df_test_Y.to_numpy())

    # Mostramos los resultados
    print("Resultados del entrenamiento")
    print_bar()
    print(f"--> Error cuadratico medio en la muestra: {in_sample_error}")
    print(f"--> Error cuadratico medio en el test: {test_sample_error}")

def show_cross_validation_step1(df_train_X, df_train_Y, df_train_X_original):
    """
    Lanza cross validation y muestra los resultados obtenidos
    En esta primera etapa, fijaremos el valor de lambda para regularizacion, y moveremos el conjunto
    de entrenamiento (sin PCA y sus transformaciones, PCA y sus trasnformaciones) y los modelos
    considerados: ajuste lineal sin regularizador, o regularizacion LASSO o Ridge

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada, a los que hemos aplicado PCA
    df_train_Y: dataframe con los datos de salida
    df_train_X_original: dataframe con los datos sin aplicar PCA
    """

    # Valor del parametro de regularizacion. En el siguiente cross validation ya fijamos un buen
    # valor para alpha
    alpha = 0.05

    # Modelos que vamos a validar
    reg = linear_model.LinearRegression()
    ridge = linear_model.Ridge(alpha=alpha)
    lasso = linear_model.Lasso(alpha=alpha)
    models = [reg, ridge, lasso]

    # Transformaciones polinomiales que vamos a realizar de los datos
    # Sin PCA no queremos hacer tantas transformaciones
    pca_transforms = [1, 2, 3, 4]
    non_pca_transforms = [1] # No podemos hacer trasnformaciones superiores, el ordenador no tiene
                             # memoria suficiente

    # Cross validation <- 10 fold, con shuffle de los datos (puede introducir variabilidad en los resultados)
    cv = KFold(n_splits=10, shuffle=True)

    for model in models:
        for order in pca_transforms:
            # Transformamos los datos de entrada con polinomios
            # Notar que por defecto intruduce la columna del bias
            poly = PolynomialFeatures(order)
            df_modified_X = pd.DataFrame(poly.fit_transform(df_train_X))

            scores = cross_val_score(model, df_modified_X.to_numpy(), df_train_Y.to_numpy(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            print(f"PCA -> Model {model}, pol_order: {order}:")
            print(f"\tMedia: {np.mean(scores)}")
            print(f"\tMinimo: {np.min(scores)}")
            print(f"\tMaximo: {np.max(scores)}")

    # Hacemos lo mismo pero para el dataset sin PCA
    for model in models:
        for order in non_pca_transforms:
            # Transformamos los datos de entrada con polinomios
            poly = PolynomialFeatures(order)
            df_modified_X = pd.DataFrame(poly.fit_transform(df_train_X_original))

            scores = cross_val_score(model, df_modified_X.to_numpy(), df_train_Y.to_numpy(), scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            print(f"No PCA -> Model {model}, pol_order: {order}:")
            print(f"\tMedia: {np.mean(scores)}")
            print(f"\tMinimo: {np.min(scores)}")
            print(f"\tMaximo: {np.max(scores)}")

    # Paramos la ejecucion hasta que el usuario pulse una tecla
    print_bar(car = "-")
    wait_for_user_input()

def show_cross_validation_step2(df_train_X, df_train_Y):
    """
    Lanza cross validation y muestra los resultados obtenidos, en la segunda etapa
    En esta etapa, ya hemos elegido que vamos a usar el dataset con PCA y transformacion polinomica
    Tambien hemos elegido el modelo a usar, y el regularizador. Asi que en esta fase de CV lo que
    vamos a elegir es el valor de lambda de penalizacion del regularizador

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada, a los que hemos aplicado PCA y la transformacion
                polinomica que se escoge en el CV step 1
    df_train_Y: dataframe con los datos de salida
    """

    # Hacemos la trasnformacion de los datos que vamos a usar
    pol_order = 3



# Funcion principal
#===============================================================================
if __name__ == "__main__":
    print("==> Estableciendo semilla inicial aleatoria")
    np.random.seed(123456789)

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
    df_test_X, df_test_Y = split_dataset_into_X_and_Y(df_test)   # Separamos datos de test

    print("--> Borrando outliers")
    prev_len = len(df_train_X)

    # No borramos los outliers en la variable de salida. Precisamente son los valores que nos
    # interesan en la aplicacion practica
    df_train_original = df_train # Para mas tarde comparar entre el df limpiado y sin limpiar
    df_train = remove_outliers(df_train, times_std_dev = 4.0, output_cols = ["critical_temp"])
    print(f"Tamaño tras la limpieza de outliers del train_set: {len(df_train)}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train)}")
    print(f"Porcentaje de filas eliminadas: {float(prev_len - len(df_train)) / float(prev_len) * 100.0}%")
    wait_for_user_input()

    # Comprobamos si el borrado de outliers en los datos de entrada afecta a los datos de salida
    # que estamos borrando. No queremos sesgar los datos de salida de los que disponemos como ejemplos
    # Por ejemplo, no podemos borrar todos las filas en las que el valor de salida es alto, porque
    # son los que nos interesa en la práctica
    print("--> Comprobando como ha afectado el borrado de outliers a la variable de salida")
    check_outliers_removal(df_train_original, df_train)

    print("--> Aplicando PCA a los datos")

    # Guardo los datos originales sin aplicar PCA
    df_train_X_original = df_train_X.copy()
    df_test_X_original = df_test_X.copy()

    # Aplicamos PCA
    df_train_X, df_test_X = apply_PCA(df_train_X, df_test_X, number_components = 10)

    print("--> Dataset despues de la transformacion PCA:")
    explore_training_set(df_train_X, show_box_plot = False)

    print("--> Estandarizando dataset PCA")
    # Notar que la variable de salida temperatura critica no la estamos estandarizando
    df_train_X, df_test_X = standarize_dataset(df_train_X, df_test_X)

    # Juntamos de nuevo los dos dataframes en uno solo, para mostrar a continuacion algunas estadisticas
    # No juntamos los dataframes de test porque no queremos saber nada de ellos, de momento
    df_train = append_series_to_dataframe(df_train_X, df_train_Y, column_name = "critical_temp")

    print("Mostramos las estadisticas de los datos de entrenamiento estandarizados")
    explore_training_set(df_train)
    wait_for_user_input()

    print(f"Tamaño df_train_X: {df_train_X.shape}")
    print(f"Tamaño df_train_Y: {df_train_Y.shape}")

    print("==> Calculamos algunos baselines haciendo entrenamiento")
    print("--> Baseline de entrenar linear regression sin aplicar PCA")
    linear_regresion = linear_model.LinearRegression()
    linear_regresion.fit(df_train_X_original.to_numpy(), df_train_Y.to_numpy())
    show_results(linear_regresion.coef_, df_train_X_original, df_train_Y, df_test_X_original, df_test_Y)

    print("--> Baseline de entrenar linear regression al aplicar PCA, pero sin transformaciones polinomicas")
    linear_regresion = linear_model.LinearRegression()
    linear_regresion.fit(df_train_X.to_numpy(), df_train_Y.to_numpy())
    show_results(linear_regresion.coef_, df_train_X, df_train_Y, df_test_X, df_test_Y)

    print("==> Aplicamos Cross Validation -> Primer paso")
    show_cross_validation_step1(df_train_X, df_train_Y, df_train_X_original)

    print("==> Hacemos la transformacion a los datos que hemos escogido en CV")
    poly = PolynomialFeatures(3)
    df_train_X = pd.DataFrame(poly.fit_transform(df_train_X))

    print("==> Aplicamos Cross Validation -> Segundo paso")
    show_cross_validation_step2(df_train_X, df_train_Y)


