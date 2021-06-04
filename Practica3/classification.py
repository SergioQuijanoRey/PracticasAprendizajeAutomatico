"""
Author:
    - Sergio Quijano Rey
    - sergioquijano@correo.ugr.es
Practica 3 - Problema de clasificacion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn import manifold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ==================== FICHERO CORE ====================
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

# ======================================================

# Parametros globales del programa
#===============================================================================

# Fichero con los datos del problema
file_path = "./datos/Sensorless_drive_diagnosis.txt"

# Columna de las etiquetas
label_col = 48

# Cargado de los datos
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

    # Columnas separadas por espacios en blanco
    # No hay una primera fila con los nombres de las columnas
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return df

def explore_training_set(df, show_plot = True):
    """
    Muestra caracteristicas relevantes del dataset de entrenamiento

    Parameters:
    ===========
    df: dataframe del que queremos realizar la exploracion
        No debe contener datos de test, pues no queremos visualizarlos
    show_plot: indica si queremos mostrar graficos

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

    if show_plot is True:
        print("Gráfica con el balanceo de los datos")
        plot_balanced_classes(df_train_Y)

def split_data(df_X, df_Y, test_percentage):
    """
    Separamos el dataset en un conjunto de entrenamiento y en otro conjunto de test

    Parameters:
    ===========
    df_X: dataframe en el que tenemos almacenados todos los datos de entrada
    df_Y: dataframe en el que tenemos almacenados las etiquetas
    test_percentage: tanto por uno que queremos que sea asignado al conjunto de test

    Returns:
    ========
    df_train_X, df_train_Y: dataframe con los datos de entrenamiento (entrada y etiquetas)
    df_test_X, df_train_Y: dataframe con los datos de test (entrada y etiquetas)
    """

    # Stratify para que las clases queden balanceadas en entrenamiento / test
    df_train_X, df_test_X, df_train_Y, df_test_Y = train_test_split(df_X, df_Y, test_size = test_percentage, shuffle = True, stratify = df_Y)
    return df_train_X, df_train_Y, df_test_X, df_test_Y

def split_dataset_into_X_and_Y(df):
    """
    Tenemos un dataframe con las variables dependientes y las etiquetas. Esta funcion los
    separa en un dataframe para cada tipo de variable

    Parameters:
    ===========
    df: dataframe con los datos

    Returns:
    ========
    df_X: dataframe con las variables dependientes
    df_Y: dataframe con las etiquetas
    """

    return df.loc[:, df.columns != 48], df[48]

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

def plot_balanced_classes(df_train_Y):
    """
    Grafico de barras para ver si las clases estan desbalanceadas o no

    Parameters:
    ===========
    df_train_Y: dataframe con las etiquetas de los datos de entrenamiento
    """

    # Es mas facil emplear numpy para este conteo
    labels = df_train_Y.to_numpy()

    # Tomamos los valores presentes en labels y el numero de veces que aparece
    unique, elements_in_current_class = np.unique(labels, return_counts = True)

    # Mostramos la grafica
    plt.title("Balanceo de las clases en el training set")
    plt.xlabel("Clase")
    plt.ylabel("Numero de ejemplos en la clase")
    plt.bar(unique, elements_in_current_class)
    plt.show()
    wait_for_user_input()

# Preprocesado de los datos
#===============================================================================
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

def standarize_dataset(train_df, test_df):
    """
    Estandariza el dataset, usando solo la informacion de los datos de entrenamiento. A los datos de
    test se les aplica la misma transformacion. Notar que no se esta usando informacion de la
    muestra de test para aplicar la estandarizacion. Pero pasamos el conjunto de test para aplicar
    la misma trasnformacion a estos datos

    Si no queremos standarizar las columnas de salida, separar antes los dataframes y pasar solo
    la matriz de datos de entrada!!

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

# Cross validation
#===============================================================================
def show_cross_validation(df_train_X, df_train_Y):
    """
    Lanza cross validation y muestra los resultados obtenidos
    Moveremos el conjunto de entrenamiento (sin PCA y sus transformaciones, PCA y sus
    trasnformaciones) y los modelos considerados: ajuste lineal sin regularizador, o regularizacion
    LASSO o Ridge

    Parameters:
    ===========
    df_train_X: dataframe con los datos de entrada, a los que hemos aplicado PCA
    df_train_Y: dataframe con los datos de salida
    """

    # Valores del parametro de regularizacion.
    C_values = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]

    # Modelos que vamos a validar
    logistic_reg = LogisticRegression(penalty='l2')
    models = ["LogReg"]

    # Transformaciones polinomiales que vamos a realizar de los datos
    # Sin PCA no queremos hacer tantas transformaciones
    transforms = [1, 2]

    # Cross validation <- 10 fold, con shuffle de los datos (puede introducir variabilidad en los resultados)
    cv = KFold(n_splits=10, shuffle=True)

    # Hacemos cross validation sobre los datos
    # Support Vector machines no haran caso a las transformaciones de los datos, esto lo conseguiremos
    # usando distintos kernels
    for model in models:
        for order in transforms:
            for C in C_values:
                # Seleccionamos el modelo a emplear
                mdl = None
                if model == "LogReg":
                    mdl = LogisticRegression(penalty='l2', C=C, multi_class="multinomial")

                # Transformamos los datos de entrada con polinomios
                # Notar que por defecto intruduce la columna del bias
                poly = PolynomialFeatures(order)
                df_modified_X = pd.DataFrame(poly.fit_transform(df_train_X))

                scores = cross_val_score(mdl, df_modified_X.to_numpy(), df_train_Y.to_numpy(), scoring='accuracy', cv=cv, n_jobs=-1)
                print(f"Model {model}, pol_order: {order}, C: {C}:")
                print(f"\tMedia: {np.mean(scores)}")
                print(f"\tMinimo: {np.min(scores)}")
                print(f"\tMaximo: {np.max(scores)}")

    # Iteramos sobre SVMS. Las transformaciones seran segun los kernels especificados
    # El kernel lineal lo ponemos aparte porque acepta un parametro adicional: degree
    kernels = ['linear', 'rbf']

    # Con SVM podemos emplear ordenes mayores gracias al kernel trick
    transforms = [1, 2, 3]

    # Primero kernel polinomial, que es el que acepta degree como parametro
    for order in transforms:
        for C in C_values:
            # ovr: one versus rest -> Estrategia para clasificacion multiclase
            model = SVC(C=C, kernel = 'poly', degree = order, decision_function_shape="ovr")
            scores = cross_val_score(model, df_train_X.to_numpy(), df_train_Y.to_numpy(), scoring='accuracy', cv=cv, n_jobs=-1)
            print(f"Model {model} Linear, pol_order: {order}, C: {C}:")
            print(f"\tMedia: {np.mean(scores)}")
            print(f"\tMinimo: {np.min(scores)}")
            print(f"\tMaximo: {np.max(scores)}")

    # Iteramos sobre el resto de kernels
    for kernel in kernels:
        for C in C_values:
            # ovr: one versus rest -> Estrategia para clasificacion multiclase
            model = SVC(C=C, kernel=kernel, decision_function_shape="ovr")
            scores = cross_val_score(model, df_train_X.to_numpy(), df_train_Y.to_numpy(), scoring='accuracy', cv=cv, n_jobs=-1)
            print(f"Model {model} {kernel}, C: {C}:")
            print(f"\tMedia: {np.mean(scores)}")
            print(f"\tMinimo: {np.min(scores)}")
            print(f"\tMaximo: {np.max(scores)}")

    # Paramos la ejecucion hasta que el usuario pulse una tecla
    print_bar(car = "-")
    wait_for_user_input()

def show_results(model, df_train_X, df_train_Y, df_test_X, df_test_Y):
    """
    Muestra los resultados del entrenamiento

    Parameters:
    ===========
    model: modelo sklearn que queremos evaluar
    df_train_X: conjunto de entrenamiento, variables de entrada
    df_train_Y: conjunto de entrenamiento, etiquetas
    df_test_X: conjunto de test, variables de entrada
    df_test_Y: conjunto de test, etiquetas
    """

    # Computamamos las predicciones en la muestra de entrenamiento
    predictions = model.predict(df_train_X.to_numpy())

    # Calculamos el error cuadratico medio en la muestra
    in_sample_acc = accuracy_score(predictions, df_train_Y.to_numpy())
    in_sample_f1 = f1_score(predictions, df_train_Y.to_numpy(), average='macro')
    in_sample_prec = precision_score(predictions, df_train_Y.to_numpy(), average="macro")
    in_sample_conf = confusion_matrix(predictions, df_train_Y.to_numpy())

    # Computamos las predicciones en el test y calculamos Etest
    predictions = model.predict(df_test_X.to_numpy())
    test_sample_acc = accuracy_score(predictions, df_test_Y.to_numpy())
    test_sample_f1 = f1_score(predictions, df_test_Y.to_numpy(), average="macro")
    test_sample_prec = precision_score(predictions, df_test_Y.to_numpy(), average = "macro")
    test_sample_conf = confusion_matrix(predictions, df_test_Y.to_numpy())

    # Mostramos los resultados
    print("Resultados del entrenamiento")
    print_bar()
    print(f"--> Accuracy en la muestra: {in_sample_acc}")
    print(f"--> Precision en la muestra: {in_sample_prec}")
    print(f"--> F1 en la muestra: {in_sample_f1}")
    print(f"--> Matriz de confusion en la muestra:\n{in_sample_conf}")
    print("")
    print(f"--> Accuracy en el test: {test_sample_acc}")
    print(f"--> Precision en el test: {test_sample_prec}")
    print(f"--> F1 en el test: {test_sample_f1}")
    print(f"--> Matriz de confusion en el test:\n{test_sample_conf}")
    wait_for_user_input()

# Funcion principal
#===============================================================================
if __name__ == "__main__":
    print("==> Estableciendo semilla inicial aleatoria")
    np.random.seed(123456789)

    print("==> Carga de los datos")
    df = load_data_csv(file_path)
    print(f"Tamaño de todo el dataset: {len(df)}")
    wait_for_user_input()

    print("==> Separamos los datos de entrenamiento de las etiquetas")
    df_X, df_Y = split_dataset_into_X_and_Y(df)

    print("==> Separamos training y test")
    # 20% del dataset a test
    df_train_X, df_train_Y, df_test_X, df_test_Y = split_data(df_X, df_Y, 0.2)
    print(f"Tamaño del set de entrenamiento: {len(df_train_X)}")
    print(f"Tamaño del set de test: {len(df_test_X)}")
    wait_for_user_input()

    # Mostramos las caracteristicas de los datos
    print("==> Exploracion de los datos de entrenamiento")

    # Junto los dos dataframes para trabajar con algo mas de comodidad
    df_train = append_series_to_dataframe(df_train_X, df_train_Y, column_name=[label_col])

    # Realizamos la exploracion de los datos
    explore_training_set(df_train)

    print("==> Procesamiento de los datos")

    print("--> Borrando outliers")
    prev_len = len(df_train_X)
    # No borramos los outliers en la variable de salida. Precisamente son los valores que nos
    # interesan en la aplicacion practica
    df_train_original = df_train # Para mas tarde comparar entre el df limpiado y sin limpiar
    df_train = remove_outliers(df_train, times_std_dev = 4.0, output_cols = [48])
    print(f"Tamaño tras la limpieza de outliers del train_set: {len(df_train)}")
    print(f"Numero de filas eliminadas: {prev_len - len(df_train)}")
    print(f"Porcentaje de filas eliminadas: {float(prev_len - len(df_train)) / float(prev_len) * 100.0}%")
    wait_for_user_input()

    # Como df_train_X, df_train_Y quedan desactualizados, los actualizo
    df_train_X, df_train_Y = split_dataset_into_X_and_Y(df_train)

    print("--> Estandarizando el conjunto de datos")
    df_train_X, df_test_X = standarize_dataset(df_train_X, df_test_X)
    print("Conjunto de entrenamiento tras normalizar")
    explore_training_set(df_train_X, show_plot=False)
    wait_for_user_input()

    # Aplicamos PCA
    print("--> Aplicando PCA a los datos")
    df_train_X, df_test_X = apply_PCA(df_train_X, df_test_X, explained_variation=0.99)

    print("--> Dataset despues de la transformacion PCA:")
    explore_training_set(df_train_X, show_plot = False)

    print("--> Estandarizando dataset PCA")
    # Notar que la variable de salida temperatura critica no la estamos estandarizando
    df_train_X, df_test_X = standarize_dataset(df_train_X, df_test_X)

    # Juntamos de nuevo los dos dataframes en uno solo, para mostrar a continuacion algunas estadisticas
    # No juntamos los dataframes de test porque no queremos saber nada de ellos, de momento
    df_train = append_series_to_dataframe(df_train_X, df_train_Y, column_name = "critical_temp")

    print("Mostramos las estadisticas de los datos de entrenamiento estandarizados")
    explore_training_set(df_train)
    wait_for_user_input()

    print("==> Aplicamos Cross Validation")
    # ESTA COMENTADO PORQUE CONSUME DEMASIADO TIEMPO DE COMPUTO
    #  show_cross_validation(df_train_X, df_train_Y)

    print("==> Entrenando sobre todo el conjunto de datos")

    print("--> Transformando con monomios de hasta grado 2")
    poly = PolynomialFeatures(2)
    df_train_X = pd.DataFrame(poly.fit_transform(df_train_X))
    df_test_X = pd.DataFrame(poly.fit_transform(df_test_X))

    # Parametros escogidos:
    C = 1.2
    lgr = LogisticRegression(penalty='l2', C=C, multi_class="multinomial")

    # Entrenamos sobre toda la poblacion de entrenamiento
    print("--> Ajustando los datos de entrenamiento")
    lgr.fit(df_train_X, df_train_Y)

    print("--> Mostrando resultados")
    show_results(lgr, df_train_X, df_train_Y, df_test_X, df_test_Y)

