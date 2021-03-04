from sklearn.model_selection import train_test_split
import numpy as np

def run():
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
    print("Resultado en forma matricial")
    print(f"El training queda: {training}")
    print("")
    print(f"El test queda: {test}")
    print("")


    # Ahora lo pruebo pero con los datos por separado
    # Para ello me quedo con todos los indices de fila y fijo la columna 0 o 1
    print("Ahora separamos el dataset")
    X = data_set[:,0]
    Y = data_set[:,1]
    print(f"Dataset: {data_set}")
    print(f"X: {X}")
    print(f"Y: {Y}")

    # Separo los datos con la funcion programada
    X_training, X_test, Y_training, Y_test = split_data_set_splitted(X, Y)

    # Muestro el resultado de separar los datos
    print("Resultado en forma separada")
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

if __name__ == "__main__":
    print("Lanzando solo el ejercicio 2")
    print("Si quieres lanzar todos los ejercicios, ejecuta:")
    print("\tpython main.py")
    run()
