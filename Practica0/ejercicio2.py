from sklearn.model_selection import train_test_split

def run():
    """Corre las operaciones necesarias para resolver el problema del ejercicio 2"""

    print("Ejecutando ejercicio 2")

    # Genero un data_set en formato matricial de pruebas para comprobar que la
    # funcion de separacion hace lo que tiene que hacer
    data_set = [[x, x] for x in range(10)]

    # Separo los datos
    training, test = split_data_set(data_set)

    # Muestro el resultado de separar los datos
    print(f"El training queda: {training}")
    print("")

    print(f"El test queda: {test}")

def split_data_set(data_set):
    """
    Dado un data set en formato matricial lo separa en un 75% para training y un 25%
    para test. Para conservar los elementos de cada clase en test y training se
    mezclan aleatoriamente los datos

    Para encontrar esta funcionalidad, he leido esta pagina de la documentacion
    oficila de scikit learn:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    # Uso la funcion de scikitlearn para separar el data_set
    # Esta funcion por defecto mezcla los datos para asegurar la representacion
    # de los datos en los dos subconjuntos
    training, test = train_test_split(data_set, train_size = 0.75, test_size = 0.25)
    return training, test
