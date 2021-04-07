"""
Practica 1
Sergio Quijano Rey
sergioquijano@correo.ugr.es
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # Para hacer graficas en 3D
from matplotlib import cm               # Para cambiar el color del grafico 3D

# Variables globales
#===============================================================================
# Como etiquetamos los datos
label5 = 1
label1 = -1

# Funciones Auxiliares / Comunes
#===============================================================================
def wait_for_user_input():
    input("Pulse ENTER para continuar...")

def readData(file_x, file_y):
    """
    Lee los datos almacenados en ficheros de numpy
    Codigo COPIADO completamente de la plantilla dada por los profesores
    Los datos de entrada X se guardan en file_x
    Los datos de salida o etiquetas Y se guardan en file_y
    """

    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
            if datay[i] == 5 or datay[i] == 1:
                    if datay[i] == 5:
                            y.append(label5)
                    else:
                            y.append(label1)
                    x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y


def simula_unif(N, d, size):
    """
    Simula N datos en un cuadrado [-size, size] ^ 2
    Codigo COPIADO completamente de la plantilla dada por los profesores

    Parameters:
    ===========
    N: numero de datos a generar
    d: numero de coordenadas de cada dato
    size: los datos se generan en el intervalo [-size, size]
    """
    return np.random.uniform(-size,size,(N,d))

# Funciones para mostrar graficas
#===============================================================================
def __birds_eye_loss_plot_not_show(loss_function, lower_x: float = -1, upper_x: float = 1, lower_y: float = -1, upper_y: float = 1, points_pers_axis: int = 1000):
    """
    Para no repetir codigo en birds_eye_loss_plot y birds_eye_gradient_descent

    Hace el plot de la funcion de error en vista de pajaro, pero no la muestra (para
    poder añadir puntos sobre dicha grafica)

    Consulto la funcion de la grafica de la documentacion oficial de matplotlib:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
    y de aqui resuelvo mi duda sobre como saber que magnitud representa cada color:
        https://matplotlib.org/3.3.4/gallery/images_contours_and_fields/contourf_demo.html
    """

    # Valores de las variables independientes
    X_values = np.linspace(lower_x, upper_x, points_pers_axis)
    Y_values = np.linspace(lower_y, upper_y, points_pers_axis)

    # Matriz con los valores de los errores segun las dos variables independientes
    # Hago un bucle for sobre la matriz inicializada a ceros porque no encuentro
    # otra forma de hacer el mapeo que busco
    loss_values_matrix = np.zeros(shape = (points_pers_axis, points_pers_axis))
    for x_index in range(0, len(X_values)):
        for y_index in range(0, len(Y_values)):
            # Hago los indices invertidos porque el primer indice mueve la fila
            # (mueve la direccion vertical) y el segundo indice mueve la columna
            # (mueve la direccion horizontal)
            loss_values_matrix[y_index][x_index] = loss_function(X_values[x_index], Y_values[y_index])

    # Mostramos la grafica de los puntos
    plt.title("Funcion de error")
    plt.xlabel("Eje X de la funcion de error")
    plt.ylabel("Eje Y del a funcion de error")
    plt.contourf(X_values, Y_values, loss_values_matrix)

    # Para poder ver las magnitudes que representa cada color en la grafica
    plt.colorbar()

def birds_eye_loss_plot(loss_function, lower_x: float = -1, upper_x: float = 1, lower_y: float = -1, upper_y: float = 1, points_pers_axis: int = 1000):
    """
    Muestro la grafica del error con un codigo de colores en 2D
    Para poder estudiar lo que pasa cuando corremos ciertos algoritmos de descenso del gradiente de forma intuitiva

    """

    # Genera el grafico
    __birds_eye_loss_plot_not_show(loss_function, lower_x, upper_x, lower_y, upper_y, points_pers_axis)

    # Como no vamos a hacer mas manipulaciones, lo muestra
    plt.show()
    wait_for_user_input()

def birds_eye_gradient_descent(loss_function, solution_at_iteration, lower_x: float = -1, upper_x: float = 1, lower_y: float = -1, upper_y: float = 1, points_pers_axis: int = 1000):
    """
    Muestra la grafica de como tomamos puntos solucion junto a la vista de pajaro de la funcion de error
    No corre el algoritmo, asi que el procedimiento es correr primero el algoritmo
    y despues pasar los puntos a la funcion, para evitar repetir demasiados calculos
    Repite gran parte del codigo de birds_eye_loss_plot

    El primer punto lo pinta con una cruz blanca, para saber que es el punto de partida
    El ultimo punto lo pinta con una cruz negra, para saber que es el punto de llegada
    """

    # Genera el grafico y realizamos mas manipulaciones a partir de este punto
    __birds_eye_loss_plot_not_show(loss_function, lower_x, upper_x, lower_y, upper_y, points_pers_axis)

    # Separo las coordenadas de las soluciones para poder mostrarlas correctamente
    # En este caso, como los datos vienen de la forma (x, y), no tengo que hacer
    # el cambio de indices que si haciamos con la matriz
    solution_x_values = solution_at_iteration[:, 0]
    solution_y_values = solution_at_iteration[:, 1]

    # Me quedo con el primer y ultimo punto para pintarlos de otros colores
    first_x = solution_x_values[0]
    last_x = solution_x_values[-1]
    first_y = solution_y_values[0]
    last_y = solution_y_values[-1]

    # No los borros para pintarlos de forma normal, pero con una cruz por encima
    # para distinguirlos

    # Añadimos la grafica de los puntos solucion, como puntos rojos gordos: "ro"
    plt.plot(solution_x_values, solution_y_values, "ro")

    # Pinto los puntos inicial y final con una cruz por encima
    # Uso este formato para especificar colores porque no se cual es el codigo
    # de caracter para estos dos colores
    plt.plot(first_x, first_y, "x", c="white")
    plt.plot(last_x, last_y, "x", c="black")

    plt.show()
    wait_for_user_input()

def plot_3d_gradient_descent(loss_function, solutions, x_lower: float, x_upper: float, y_lower: float, y_upper: float, points_pers_axis: int = 100):
    """
    Muestra en tres dimensiones la funcion de error junto a los puntos de
    A partir del codigo dado por los profesores, ligeramente modificado
    Le paso a mano los extremos de los valores de x e y para tener mas control sobre la grafica

    El color distinto lo saco de la documentacion oficial:
        https://matplotlib.org/3.1.1/gallery/mplot3d/surface3d.html
    """


    # Genero los valores de los ejes X e Y segun los parametros dados
    X = np.linspace(x_lower, x_upper, points_pers_axis)
    Y = np.linspace(y_lower, y_upper, points_pers_axis)
    X, Y = np.meshgrid(X, Y)

    # Valores del eje vertical segun la funcion de perdida dada como parametro
    Z = loss_function(X, Y)

    # Para generar  una grafica en 3d
    fig = plt.figure()
    ax = Axes3D(fig)

    # Genero la grafica en 3d a partir de los datos ya calculados
    # Cambio el color gracias al enlace que he referenciado
    # Añado alpha para que los puntos que pinte encima de la grafica se vean bien
    surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap=cm.coolwarm, alpha = 0.7)
    ax.plot(x_lower, y_lower, loss_function(x_lower, y_lower), 'r*', markersize = 10)

    # Añado titulo y leyenda a la grafica
    ax.set(title="Iteraciones del gradiente descendente junto a la funcion de error")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss(X, Y)')

    # Ponemos los puntos del gradiente descendente sobre la grafica de error
    for solution in solutions:
        # Para levantar algo los puntos sobre la superficie y que se vean mejor
        epsilon = 0.5

        plt.plot(solution[0], solution[1], loss_function(solution[0], solution[1]) + epsilon, "ko")

    plt.show()
    wait_for_user_input()

def scatter_plot_with_classes(data, classes, target_names, feature_names, title, ignore_first_column: bool = True, show: bool = True):
    """
    Hacemos un scatter plot de puntos con dos coordeandas que estan etiquetados en distintos grupos

    Mucho del codigo esta tomado de la practica anterior, en concreto, de hacer
    el plot de iris dataset

    Parameters:
    ===========
    data: coordeandas de los distintos puntos
    classes: etiquetas numericas de las clases a la que pertenencen los datos
    target_names: nombres que le doy a cada una de las clases
    feature_names: nombre de los ejes de coordenadas que le queremos dar al grafico
    ignore_first_column: indica si tenemos que ignorar o no la primera columna
                         Esto pues en algunos casos la primera columna de la matriz de datos
                         tiene todo unos para representar el sumando del termino independiente
                         en las ecuaciones lineales
    show: indica si queremos mostrar o no la grafica
          Nos puede interesar no mostrar la grafica para añadir nuevos elementos
          a esta grafica sin tener que repetir codigo

    data y classes ya en un tipo de dato numpy para poder operar con ellos
    """

    # Tomo las coordenadas de la matriz de datos, es decir, separo coordenadas
    # x e y de una matriz de datos que contiene pares de coordenadas
    x_values = data[:, 0]
    y_values = data[:, 1]

    # Cuando ignore_first_column = True, para poder operar con la matriz X,
    # su primera columna es todo unos (que representan los terminos independientes
    # en las operaciones matriciales). Para estas graficas logicamente no nos interesa esa columna
    if ignore_first_column is True:
        x_values = data[:, 1]
        y_values = data[:, 2]

    # Colores que voy a utilizar para cada una de las clases
    # Rojo para un numero, azul para otro color.
    colormap = ['red', 'blue']

    # Separacion de indices. Con esto, consigo la lista de los indices de la
    # clase i-esima, cada uno en un vector distinto. Esto lo necesitare para
    # colorear cada clase de un color y ponerle de label el nombre de la planta

    # Separo los indices correspondientes a la clase del numero 1 y la clase del
    # numero 5
    first_class_indexes = np.where(classes == -1)
    second_class_indexes = np.where(classes == 1)

    # Asi puedo referirme a la primera clase como splitted_indixes[0] en vez
    # de usar el nombre de la variable (para acceder a los indices en el siguiente
    # bucle)
    splitted_indixes = [first_class_indexes, second_class_indexes]

    # Tomo estos elementos para hacer graficas elaboradas
    fig, ax = plt.subplots()

    # Itero sobre las clases
    for index, target_name in enumerate(target_names):

        # Tomo las coordenadas de la clase index-esima
        current_x = x_values[splitted_indixes[index]]
        current_y = y_values[splitted_indixes[index]]

        # Muestro la clase index-esima, con su color y su etiqueta correspondiente
        # Ponemos alpha para apreciar donde se acumulan muchos datos (que seran
        # zonas mas oscuras que aquellas en las que no hay acumulaciones)
        ax.scatter(current_x, current_y, c=colormap[index], label=target_name, alpha = 0.6)

    # Titulo para la grafica
    plt.title(title)

    # Tomo los titulos de las caracteristicas y los asigno al grafico
    # Tomo la idea de: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
    x_legend = feature_names[0]
    y_legend = feature_names[1]
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    # Muestro la grafica en caso de que show = True
    if show is True:
        plt.show()
        wait_for_user_input()

def plot_classification_predictions(data, labels, weights, feature_names, title: str = "Grafica de predicciones"):
    """
    Mostramos un scatter plot de los datos segun las predicciones que hagamos. Si
    predecimos correctamente la etiqueta, se pinta el punto en un color gris. Si
    se falla la prediccion, se pinta en un color rojo

    Parameters:
    ===========
    data: los datos de entrada sobre los que predecimos
    labels: los verdaderos valores que deberiamos predecir
    weights: los pesos que representan la funcion lineal de clasificacion
    feature_names: el nombre de las caracteristicas en base a las que hacemos
                   las predicciones
    """
    # Tomo las coordenadas de la matriz de datos, es decir, separo coordenadas
    # x e y de una matriz de datos que contiene pares de coordenadas
    # Para poder operar con la matriz X, su primera columna es todo unos (que representan
    # los terminos independientes en las operaciones matriciales). Para estas
    # graficas logicamente no nos interesa esa columna
    x_values = data[:, 1]
    y_values = data[:, 2]

    # Funcion de prediccion
    lineal = get_lineal(weights)

    # Predicciones sobre el conjunto de datos
    # Solo me quedo con el signo, porque no me interesa saber el grado en el que
    # acierto o fallo, solo si acierto o fallo
    predictions = [np.sign(lineal(x)) for x in data]

    # Serapo los indices en indices de puntos que hemos predicho correctamente
    # e indices de puntos mal predichos
    good_predicion_indexes = np.where(predictions == labels)
    bad_prediction_indexes = np.where(predictions != labels)

    # Gris para los puntos bien predichos, rojo para los puntos mal predichos
    colormap = ['grey', 'red']

    # Nombre que vamos a poner en la leyenda
    target_names = ['Puntos BIEN predichos', 'Puntos MAL predichos']

    # Asi puedo referirme a la clase de puntos bien predichos como splitted_indixes[0]
    # (para acceder a los indices en el siguiente bucle)
    splitted_indixes = [good_predicion_indexes, bad_prediction_indexes]

    # Tomo estos elementos para hacer graficas elaboradas
    fig, ax = plt.subplots()

    # Itero sobre las clases
    for index, target_name in enumerate(target_names):

        # Tomo las coordenadas de la clase index-esima
        current_x = x_values[splitted_indixes[index]]
        current_y = y_values[splitted_indixes[index]]

        # Muestro la clase index-esima, con su color y su etiqueta correspondiente
        # Ponemos alpha para apreciar donde se acumulan muchos datos (que seran
        # zonas mas oscuras que aquellas en las que no hay acumulaciones)
        ax.scatter(current_x, current_y, c=colormap[index], label=target_name, alpha = 0.6)

    # Titulo para la grafica
    plt.title(title)

    # Tomo los titulos de las caracteristicas y los asigno al grafico
    # Tomo la idea de: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_iris_scatter.html
    x_legend = feature_names[0]
    y_legend = feature_names[1]
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    plt.show()
    wait_for_user_input()

def plot_error_evolution(error_at_iteration, title = "Evolucion del error", x_label = "Iteraciones", y_label = "Error"):
    """
    Muestra la grafica de evolucion del error con el paso de algun tipo de iteraciones
    Con 'un tipo de iteraciones' me refiero a que podemos estar indicando error
    por cada iteracion, por cada recorrido del minibatch, por cada epoch...
    Sea cual sea el 'tipo de iteracion', esta grafica es la misma

    Parameters:
    ===========
    error_at_iteration: el error en cada 'tipo de iteracion'
    title: titulo de la grafica
    x_label: para indicar el 'tipo de iteracion'
    y_label: por si queremos especificar algo sobre la medida el error
    """

    # Generamos los datos necesarios
    Y = error_at_iteration
    X = np.arange(0, len(Y))

    # Mostramos el grafico
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(X, Y)
    plt.show()
    wait_for_user_input()

def plot_frontier_line(data, labels, weights):
    """
    Muestra un scatter plot de los datos, coloreados por su categoria, y la
    recta que separa los datos segun la clasificacion que representa los pesos
    dados como parametro

    Parameters:
    ===========
    data: datos de entrada sobre los que se quiere hacer la separacion
    labels: etiquetas de los datos, para pintar los colores de las clasificaciones verdaderas
    weights: representa la funcion lineal que clasifica
    """

    # Reutilizamos el codigo para realizar el scatter plot
    # Para que no haga show y podamos añadir mas elementos, hacemos show = False
    scatter_plot_with_classes(
        data = data,
        classes = labels,
        target_names = ["Digito 1", "Digito 5"],
        feature_names = ["Intensidad", "Simetria"],
        title = "Linea de frontera de nuestro clasificador",
        show = False
    )

    # Recta que queda al igualar y = 0 y despejar x2 de la ecuacion:
    # y = w0 + x1 w1 + x2 w2
    # Para ello tiene que ocurrir que w2 != 0

    # Comprobacion de seguridad
    if weights[2] == 0:
        print("\t[Err] El segundo peso es nulo, no se puede mostrar esta linea")
        print("\tPor tanto, no mostramos nada")
        return

    # Ecuacion de la recta anteriormente descrita
    line = lambda x: (-weights[0] - weights[1] * x) * (1 / weights[2])

    # Tomamos los valores de abscisa sobre los que vamos a mapear la recta
    x_values = data[:, 1]
    y_values = line(x_values)

    # Ahora añadimos la linea que separa los datos
    plt.plot(x_values, y_values)

    # Ahora si que hacemos show de las dos graficas compuestas
    plt.show()
    wait_for_user_input()


def scatter_plot(x_values, y_values, title = "Scatter Plot Simple", x_label= "Eje X", y_label = "Eje Y"):
    """
    Grafico simple tipo scatter plot
    No tenemos clases que separar asi que pintamos todos los puntos del mismo color

    Parameters:
    ===========
    x_values: valores de la coordenada x de los datos
    y_values: valores de la coordenada y de los datos
    title: titulo del grafico
    x_label: etiqueta para el eje x
    y_label: etiqueta para el eje y
    """


    # Tomo estos elementos para hacer graficas elaboradas
    fig, ax = plt.subplots()

    # Muestro el scatter plot de los datos
    # Añado alpha por si los datos se acumulan unos sobre otros, para que esto
    # sea facilmente visible
    ax.scatter(x_values, y_values, c="grey", alpha = 0.6)

    # Titulo para la grafica
    plt.title(title)

    # Añado las leyendas en los ejes
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Muestro el grafico
    plt.show()
    wait_for_user_input()

# Algoritmos
#===============================================================================
def gradient_descent(starting_point, loss_function, gradient, learning_rate: float = 0.001, max_iterations: int = 100_000, target_error: float = None, verbose: bool = False):
    """
    Implementa el algoritmo de batch gradient descent

    Si verbose == true, entonces guardamos los errores y soluciones obtenidas en cada iteracion
    Si target_error == None, consumimos todas las iteraciones sin tener en cuenta el error
       target_error != None, paramos cuando estamos por debajo de dicha cota pasada como argument
    """

    # Defino esta funcion dentro para no ensuciar el resto del codigo
    # Devuelve si tenemos que parar (True) de iterar por estar por debajo de la cota de error
    # Claramente, si target_error == None siempre devuelve False
    def stop_because_error(current_solution):
        if target_error == None:
            return False

        return loss_function(current_solution[0], current_solution[1]) < target_error

    # Solucion de partida a partir de la cual iterar
    current_solution = starting_point
    current_iteration = 0

    # Guardamos errores y soluciones de las iteraciones cuando verbose == True
    error_at_iteration = None
    solution_at_iteration = None
    if verbose == True:
        error_at_iteration = [loss_function(current_solution[0], current_solution[1])]
        solution_at_iteration = [starting_point]

    while current_iteration < max_iterations and stop_because_error(current_solution) == False:
        # Calculamos la siguiente solucion usando el gradiente
        current_solution = current_solution - learning_rate * gradient(current_solution[0], current_solution[1])

        current_iteration = current_iteration + 1

        if verbose == True:
            error_at_iteration.append(loss_function(current_solution[0], current_solution[1]))
            solution_at_iteration.append(current_solution)

    return current_solution, current_iteration, np.array(error_at_iteration), np.array(solution_at_iteration)

def gradient_descent_and_plot_error(starting_point, loss_function, gradient, learning_rate: float = 0.001, max_iterations: int = 100_000, target_error: float = None):
    """
    Ejecutamos el gradiente descendente y mostramos la grafica de la evolucion del error
    Para no repetir muchas veces el mismo codigo en el que lanzamos el algoritmo y
    mostramos la grafica de la evolucion del error
    """

    print(f"Corriendo el algoritmo de descenso del gradiente para eta = {learning_rate}")
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent(starting_point, loss_function, gradient, learning_rate, max_iterations, target_error, verbose = True)

    # Mostramos la grafica de descenso del error
    Y = error_at_iteration
    X = np.arange(0, len(Y))

    plt.title(f"Evolucion del error para eta = {learning_rate}")
    plt.xlabel("Iteracion")
    plt.ylabel("Error")
    plt.plot(X, Y)
    plt.show()
    wait_for_user_input()

    # Por si necesitamos realizar otras operaciones con los resultados
    return weights, iterations, error_at_iteration, solution_at_iteration

def pseudo_inverse(data_matrix, label_vector):
    """
    Calcula los pesos de la regresion lineal a partir del algoritmo de la pseudo inversa

    Se puede hacer con numpy.pinv pero lo he dejado asi porque no me da fallos

    Parameters:
    ===========
    data_matrix: matriz numpy con los datos de entrada. X en notacion de los apuntes
    label_vector: array numpy con los datos de salida. Y en notacion de los apuntes

    Returns:
    ========
    weights: array numpy con los pesos del hiperplano que mejor se ajusta al modelo
    """

    # Para simplificar la formula que devolvemos (que el return sea mas compacto)
    # En los parametros dejo los nombres como estan para que sea mas explicito
    # el significado que tienen
    X = data_matrix
    Y = label_vector

    return np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

def stochastic_gradient_descent(data, labels, starting_solution, learning_rate: float = 0.001, batch_size: int = 1, max_minibatch_iterations: int = 200, target_error: float = None, verbose: bool = False):
    """
    Implementa el algoritmo de Stochastic Gradient Descent

    A diferencia del programado para Gradient Descent, usamos datos etiquetados
    como parametros de entrada en vez de una funcion de perdida con su gradiente
    calculado analiticamente, por lo que vamos a calular numericamente la funcion
    de error (error cuadratico medio) como se indican en las transparencias de teoria

    Parameters:
    ===========
    data: datos de entrada sobre los que queremos hacer predicciones
    labels: verdaderos valores que queremos predecir. Pueden representar etiquetas
            de una categoria para clasificacion o valores reales para regresion.
            Gracias a las etiquetas podemos calcular aproximadamente el gradiente
            del error para una solucion iterativa concreta
    starting_solution: np.array del que parte las soluciones iterativas
    learning_rate: tasa de aprendizaje
    max_minibatch_iterations: maximo numero de iteraciones
                              Por iteracion entendemos cada vez que modificamos los
                              pesos de la solucion iterativa (ie. cada recorrido de
                              un minibatch)
    target_error: error por debajo del cual dejamos de iterar
                  Puede ser None para indicar que no comprobemos el error para dejar de iterar
    verbose: indica si queremos que se guarden metricas en cada epoch
    """

    # Si verbose == True, guardamos algunas metricas parciales durante el proceso

    # Error que tenemos en cada epoca (en cada iteracion seria algo excesivo, asi
    # como en cada minibatch). Pero si que guardamos en cada iteracion sobre el
    # minibatech
    error_at_epoch = None
    error_at_minibatch = None
    if verbose is True:
        error_at_epoch = []
        error_at_minibatch = []

    # Establecemos la solucion actual (que vamos a ir modificando) a la solucion
    # inicial dada
    current_solution = starting_solution

    # Para controlar current_minibatch_iterations < max_minibatch_iterations de
    # forma comoda, porque tenemos dos bucles for
    current_minibatch_iterations = 0

    while current_minibatch_iterations < max_minibatch_iterations:
        # Generamos los minibatches a partir de los datos de entrada
        # Trabajamos por comodidad y eficiencia con indices, como se indica en la funcion
        mini_batches_index_groups = get_minibatches(data, batch_size)

        # Iteramos en los minibatches
        for mini_batches_indixes in mini_batches_index_groups:
            # Tomo los datos y etiquetas asociadas a los indices de este minibatch
            minibatch_data = data[mini_batches_indixes]
            minibatch_labels = labels[mini_batches_indixes]

            # Calculo la aproximacion al gradiente con estos datos
            minibatch_approx_gradient = calculate_gradient_from_data(minibatch_data, minibatch_labels, current_solution)

            # Actualizo la solucion con este minibatch
            current_solution = current_solution - learning_rate * minibatch_approx_gradient

            # Añadimos el error sobre la iteracion del minibatch
            if verbose is True:
                # Tomamos la solucion como un array, no como una matriz de una unica fila
                # pues esto provoca fallos en otras funciones (como la del calculo del error)
                tmp_solution = current_solution
                if(len(np.shape(current_solution)) == 2):
                    tmp_solution = current_solution[0]

                error_at_minibatch.append(clasiffication_mean_square_error(data, labels, tmp_solution))

            # Hemos hecho una pasada completa al minibatch, aumentamos el contador
            # y comprobamos si hemos superado el maximo (tenemos que hacer esta
            # compobracion por estar en un doble bucle)
            current_minibatch_iterations += 1
            if current_minibatch_iterations >= max_minibatch_iterations:
                break

        # Comprobamos si hemos alcanzado el error objetivo para dejar de iterar
        if target_error is not None:
            # Tomamos la solucion como un array, no como una matriz de una unica fila
            # pues esto provoca fallos en otras funciones (como la del calculo del error)
            tmp_solution = current_solution
            if(len(np.shape(current_solution)) == 2):
                tmp_solution = current_solution[0]

            if clasiffication_mean_square_error(data, labels, tmp_solution) < target_error:
                break

        # Añadimos el error en este epoch
        if verbose is True:
            # Tomamos la solucion como un array, no como una matriz de una unica fila
            # pues esto provoca fallos en otras funciones (como la del calculo del error)
            tmp_solution = current_solution
            if(len(np.shape(current_solution)) == 2):
                tmp_solution = current_solution[0]

            error_at_epoch.append(clasiffication_mean_square_error(data, labels, tmp_solution))


    # Devolvemos la solucion como un array, no como una matriz de una unica fila
    # pues esto provoca fallos en otras funciones (como la del calculo del error)
    if(len(np.shape(current_solution)) == 2):
        current_solution = current_solution[0]

    return current_solution, error_at_epoch, error_at_minibatch

def get_minibatches(data, batch_size: int):
    """
    Dados unos datos de entrada, mezcla los datos y los devuelve en subconjuntos
    de batch_size elementos. Realmente devolvemos un array de conjuntos de indices
    que representan este mezclado y empaquetado, pues asi es mas facil de operar
    (no alteramos los datos de entrada y no tenemos que considerar como quedarian
    ordenadas las etiquetas asociadas a los datos) y mas eficiente (trabajamos
    con indices, no con datos multidimensionales)

    Paramters:
    ==========
    data: matriz de datos de entrada que queremos mezclar y agrupar
    batch_size: tamaño de los paquetes en los que agrupamos los datos

    Returns:
    ========
    indixes: array de conjuntos de indices (array tambien) que representa la operacion
             descrita anteriormente
    """

    # Los indices de todos los datos de entrada
    # Me quedo con las filas porque indican el numero de datos con los que trabajamos
    # Las columnas indican el numero de caracteristicas de cada dato
    all_indixes = np.arange(start = 0, stop = np.shape(data)[0])

    # Mezclo estos indices antes de dividirlos en minibatches
    np.random.shuffle(all_indixes)

    # Array de conjuntos de indices (array de arrays)
    grouped_indixes = []

    # Agrupamos los indices que ya han sido mezclados en los minibatches
    last_group = []
    for value in all_indixes:

        # El ultimo minibatch no esta completo, podemos añadir un nuevo punto
        if len(last_group) < batch_size:
            last_group.append(value)

        # El minibatch esta completo, asi que hay que hay que añadirlo al grupo
        # de minibatches y reiniciar el grupo
        if len(last_group) == batch_size:
            grouped_indixes.append(last_group)
            last_group = []

    return np.array(grouped_indixes)

def calculate_gradient_from_data(data, labels, weights):
    """
    Calculamos el valor del gradiente a partir de los datos etiquetados y los
    pesos que representan la solucion actual que predice los valores

    Parameters:
    ===========
    data: datos de entrada que se usan para predecir
    labels: valores reales que debemos predecir correctamente
    weights: pesos que representan la funcion lineal que predice

    """
    # Inicializamos la aproximacion al gradiente con ceros y con el shape de la
    # solucion que generamos con estos datos (columnas que nos indican las caracteristicas
    # de las soluciones, no filas que nos indican el numero de datos)
    gradient = np.zeros((1, np.shape(data)[1]))

    # Funcion lineal que representan los pesos dados
    lineal = get_lineal(weights)

    # Aproximamos el gradiente linealmente con la formula dada en las transparencias
    for value, label in zip(data, labels):
        curr_err = value * (lineal(value) - label)
        gradient = gradient + curr_err

    return gradient * (2 / len(data))

# Ejercicio 1
#===============================================================================
def ejercicio1_apartado2():
    # Definimos la funcion de coste
    # Me quedo con el cuerpo del cuadrado porque lo usaremos en las derivadas parciales
    # Usamos np.float64 para forzar el uso de flotantes de 64 bits
    inside = lambda u, v: np.float64(np.power(u, 3) * np.exp(v - 2) - 2.0 * np.power(v, 2) * np.exp(-u))
    E = lambda u, v: np.float64(np.power(inside(u, v), 2))

    # Derivadas parciales, cuya expresion hemos calculado a mano y cuyo procedimiento
    # esta reflejado en el documento
    # De nuevo, usamos np.float64 para forzar el uso de flotantes de 64 bits
    dEu = lambda u, v: np.float64(2 * inside(u, v) * (3 * np.power(u, 2) * np.exp(v - 2) + 2 * np.power(v, 2) * np.exp(-u)))
    dEv = lambda u, v: np.float64(2 * inside(u, v) * (np.power(u, 3) * np.exp(v - 2) - 4 * v * np.exp(-u)))

    # Gradiente de la funcion de coste
    gradient = lambda u, v: np.array([dEu(u, v), dEv(u, v)])

    # Mostramos la expresion de la funcion de coste y la expresion del gradiente
    print("La funcion de coste es:")
    print("\t(u³ * exp(v−2) −2v² exp(−u))²")
    print("La derivada parcial respecto de u es:")
    print("\t2 * (u³ * exp(v−2) −2v² exp(−u)) * (3u² * exp(v-2) + 2v²exp(-u)")
    print("La derivada parcial respecto v es:")
    print("\t2* (u³ * exp(v−2) −2v² exp(−u)) * (u³ * exp(v-2) -4vexp(-u))")
    print("")
    wait_for_user_input()

    # Mostramos la grafica de la funcion de error
    print("Mostrando grafica de la funcion de error")
    birds_eye_loss_plot(E, -5, 5, -5, 5, 100)
    print("")

    # Parametros para el gradiente descendente
    learning_rate = 0.1
    max_iterations = 10000000000
    target_error = 1e-14
    starting_point = np.array([1.0, 1.0])

    # Lanzamos el descenso y mostramos la grafica de evolucion del error
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, E, gradient, learning_rate, max_iterations, target_error)

    # Mostramos algunos datos numericos del resultado
    print("Resultados:")
    print(f"\tNumero de iteraciones: {iterations}")
    print(f"\tPesos encontrados: {weights}")
    wait_for_user_input()

    # Cuantas iteraciones tarda el algoritmo en obtener un valor de E inferior a 10e-14
    # Hacemos indixes[0] para quedarnos con la lista de indices que devuelve, e
    # indixes[0][0] para quedarnos con el primer elemento de dicha lista
    indixes = np.where(error_at_iteration < 10e-14)
    first_index_under_error = indixes[0][0]
    print(f"La primera iteracion en la que el error esta por debajo de 10e-14 es: {first_index_under_error} (contando desde cero)")
    print(f"Las primeras coordenadas que estan por debajo de ese error: {solution_at_iteration[first_index_under_error]}")
    print("")
    wait_for_user_input()

    # Mostramos la grafica de como han avanzados las soluciones junto a la funcion de error
    print("Mostrando como han avanzado las soluciones junto a la funcion de error")
    birds_eye_gradient_descent(E, solution_at_iteration, 0.95, 1.2, 0.90, 1.05, 100)
    print("")

def ejercicio1_apartado3():
    # Funcion de perdida que nos dan y calculo sus derivadas parciales
    # No muestro por pantalla las expresiones porque el enunciado no lo pide
    f = lambda x, y: np.power(x + 2.0, 2.0) + 2.0 * np.power(y - 2.0, 2.0) + 2 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    dfx = lambda x, y: 2.0 * (x + 2.0) + 4.0 * np.pi * np.sin(2.0 * np.pi * y) * np.cos(2.0 * np.pi * x)
    dfy = lambda x, y: 4.0 * (y - 2.0) + 4.0 * np.pi * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    gradient = lambda x, y: np.array([dfx(x, y), dfy(x, y)])

    # Muestro la funcion de error porque tuve algunos problemas con las graficas
    # de errores (no bajaba el error de forma consistente) y necesitaba visualizar
    # la forma de la funcion. El error estaba en un signo de la derivada mal pasasdo a ordenador
    print("Mostrando la grafica de la funcion de error")
    birds_eye_loss_plot(f, -5, 5, -5, 5, 100)
    print("")

    # Parametros para el gradiente descendiente
    # No se especifica el error que hay que alcanzar asi que lo pongo a None
    # para que se consuman las iteraciones
    learning_rate = 0.01
    max_iterations = 50
    target_error = None
    starting_point = np.array([-1.0,1.0])

    # Lanzamos el descenso y mostramos la grafica con los resultados
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)
    print("")

    # Mostramos la grafica de como avanza el algoritmo
    print("Mostramos como avanza el algoritmo")
    birds_eye_gradient_descent(f, solution_at_iteration, -1.5, -0.9, 0.5, 1.5, 100)
    print("")

    # Realizamos el mismo proceso pero modificando el valor del learning rate
    learning_rate = 0.1
    weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)

    # La grafica del error fluctua demasiado, lo cual me parece raro
    # Por tanto, cobra mas sentido mostrar esta grafica
    print("Grafica del error rara, porque tiene muchas fluctuaciones, mostramos como ha avanzado el algoritmo")
    birds_eye_gradient_descent(f, solution_at_iteration, -5, 5, -5, 5, 100)
    print("")
    print("La fluctuacion se debe a que, con un learning_rate tan algo, nos salimos de los optimos locales")
    wait_for_user_input()

    # Ahora buscamos los errores minimos y valores de la solucion cuando partimos
    # desde distintos valores de partida

    # Como no se especifica nada en el enunciado del ejercicio, establezco el
    # learning rate, el numero maximo de iteraciones y el target_error
    learning_rates = [0.01, 0.1]
    max_iterations = 1000
    target_error = None

    # Muestro explicitamente que estos valores los he fijado yo al no tener indicaciones
    print("Fijo los siguientes parametros para el gradiente descendente:")
    print(f"\tlearning_rates (probamos con mas de uno): {learning_rates}")
    print(f"\tmax_iterations: {max_iterations}")
    print(f"\ttarget_error: {target_error}")
    print("")
    wait_for_user_input()

    # Vector con los puntos de partida que se nos piden
    # Para recorrerlo comodamente en un bucle y evitar repitir codigo
    starting_points = [
        [-0.5, -0.5],
        [1, 1],
        [2, 1],
        [-2, 1],
        [-3, 3],
        [-2, 2]
    ]

    for starting_point in starting_points:
        for learning_rate in learning_rates:
            # Lo pasamos a un array de numpy
            starting_point = np.array(starting_point)

            # Mostramos la grafica de descenso del error, que tambien nos devuelve
            # toda la informacion del proceso
            weights, iterations, error_at_iteration, solution_at_iteration = gradient_descent_and_plot_error(starting_point, f, gradient, learning_rate, max_iterations, target_error)
            print(f"Resultados para starting_point {starting_point} y learning_rate {learning_rate}")
            print(f"\t(x, y): {weights}")
            print(f"\tError final: {error_at_iteration[-1]}")
            wait_for_user_input()

            # Mostramos la traza de soluciones junto a la gráfica del error
            print("Mostrando la grafica de las soluciones junto al error")
            birds_eye_gradient_descent(f, solution_at_iteration, starting_point[0] - 2, starting_point[0] + 2, starting_point[1] - 2, starting_point[1] + 2, 30)
            print("")
            wait_for_user_input()

            # Mostramos la traza en tres dimensiones, porque las graficas
            # en dos dimensiones no nos dan informacion suficiente
            print("Mostrando la grafica de las soluciones (3d) junto al error")
            plot_3d_gradient_descent(f, solution_at_iteration, starting_point[0] - 2, starting_point[0] + 2, starting_point[1] - 2, starting_point[1] + 2, 30)
            print("")
            wait_for_user_input()

def ejercicio1():
    print("Ejecutando ejercicio 1")

    print("Apartado 2)")
    print("=" * 80)
    ejercicio1_apartado2()
    print("")

    print("Apartado 3)")
    print("=" * 80)
    ejercicio1_apartado3()
    print("")

# Ejercicio 2
#===============================================================================

def get_clasifficator(weights):
    """
    Dados los pesos de un modelo lineal, devuelve la funcion que clasifica linealmente
    Pasar la matriz de datos adecuada a la funcion que se devuelve segun el vector
    de representancion que se este usando. En la mayoria de casos el vector
    de representacion es (1, x1, x2), pero en el ultimo ejercicio se usa el vector
    (1, x1, x2, x1 * x2, x1**2, x2**2)
    """
    return lambda X: np.sign(np.dot(X, weights))

def get_lineal(weights):
    """
    Dados los pesos de un modelo lineal, devuelve la funcion lineal que se esta representando
    Pasar la matriz de datos adecuada a la funcion que se devuelve segun el vector
    de representancion que se este usando. En la mayoria de casos el vector
    de representacion es (1, x1, x2), pero en el ultimo ejercicio se usa el vector
    (1, x1, x2, x1 * x2, x1**2, x2**2)
    """

    # X es la matriz de datos sobre la que hacemos una prediccion
    return lambda X: np.dot(X, weights.T)

def clasiffication_error(data, labels, weights):
    """
    Dados unos datos etiquetados y unos pesos que representan una funcion lineal
    de clasificacion, calcula el error que se comete. Sirve tanto para calcular
    el error dentro de la muestra, como para calcular el error fuera de la muestra,
    segun los valores de data y labels pasados como parametro

    Usamos el error de clasificacion como medida del error

    Parameters:
    ===========
    data: los datos de entrada sobre los que predecimos
    labels: los verdaderos valores a predecir
    weights: los pesos que representan la funcion lineal de clasificacion
    """

    # Funcion lineal que representan los pesos
    lineal = get_lineal(weights)

    # Recorremos sobre los datos de entrada y las etiquetas reales de esos datos
    error = 0
    for (current_input, current_label) in zip(data, labels):
        error += max(0, -current_label * lineal(current_input))

    return error

def clasiffication_mean_square_error(data, labels, weights):
    """
    Dados unos datos etiquetados y unos pesos que representan una funcion lineal
    de clasificacion, calcula el error que se comete. Sirve tanto para calcular
    el error dentro de la muestra, como para calcular el error fuera de la muestra,
    segun los valores de data y labels pasados como parametro

    Usamos el error error cuadratico medio como medida de error

    Parameters:
    ===========
    data: los datos de entrada sobre los que predecimos
    labels: los verdaderos valores a predecir
    weights: los pesos que representan la funcion lineal de clasificacion
    """
    # Funcion lineal que representan los pesos
    lineal = get_lineal(weights)

    # Recorremos sobre los datos de entrada y las etiquetas reales de esos datos
    error = 0
    for (current_input, current_label) in zip(data, labels):
        error += (current_label - lineal(current_input))**2

    return error / len(labels)

def classification_porcentual_error(data, labels, weights):
    """
    Dados unos datos etiquetados y unos pesos que representan una funcion lineal
    de clasificacion, calcula el error porcentual (porcentaje de fallos / total
    de predicciones)que se comete. Sirve tanto para calcular el error dentro de
    la muestra, como para calcular el error fuera de la muestra, segun los valores
    de data y labels pasados como parametro


    Parameters:
    ===========
    data: los datos de entrada sobre los que predecimos
    labels: los verdaderos valores a predecir
    weights: los pesos que representan la funcion lineal de clasificacion
    """

    # Funcion lineal que representan los pesos
    lineal = get_lineal(weights)

    # Recorremos sobre los datos de entrada y las etiquetas reales de esos datos
    number_of_bad_predictions = 0
    for (current_input, current_label) in zip(data, labels):
        # Prediccion del input actual
        prediction = np.sign(lineal(current_input))

        # Hemos fallado en esta prediccion
        if prediction != current_label:
            number_of_bad_predictions += 1

    # Devolvemos el porcentaje de malas predicciones / total de predicciones
    return (number_of_bad_predictions / len(labels)) * 100




def ejercicio2_apartado1():

    # Leemos los datos de entrenamiento y de test
    X, Y = readData('datos/X_train.npy', 'datos/y_train.npy')
    X_test, Y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

    # Mostramos la grafica de los digitos
    print("Grafica de los datos")
    scatter_plot_with_classes(X, Y, ["Digito 1", "Digito 5"], ["Intensidad", "Simetria"], "Grafica de los datos de entrada")
    plt.show()

    # Calculamos la regresion lineal con la pseudo inversa
    # Calculamos tambien el error cometido para mostrar todos los resultados de golpe
    # Error tanto en la muestra comop fuera de la muestra, y ademas error de
    # clasificacion como error cuadratico medio
    print("Calculamos los pesos de la regresion lineal usando el algoritmo de pseudo inversa")
    weights = pseudo_inverse(X, Y)
    error_in_sample = clasiffication_error(X, Y, weights)
    error_out_sample = clasiffication_error(X_test, Y_test, weights)
    mean_square_error_in_sample = clasiffication_mean_square_error(X, Y, weights)
    mean_square_error_out_sample = clasiffication_mean_square_error(X_test, Y_test, weights)
    porcentual_error_in_sample = classification_porcentual_error(X, Y, weights)
    porcentual_error_out_sample = classification_porcentual_error(X_test, Y_test, weights)
    print(f"\tLos pesos obtenidos son: {weights}")
    print(f"\tEl error de clasficacion en la muestra Ein es: {error_in_sample}")
    print(f"\tEl error de clasificacion fuera de la muestra Eout es: {error_out_sample}")
    print(f"\tEl error cuadratico medio en la muestra Ein es: {mean_square_error_in_sample}")
    print(f"\tEl error cuadratico medio fuera de la muestra Eout es: {mean_square_error_out_sample}")
    print(f"\tEl error porcentual en la muestra Ein es: {porcentual_error_in_sample}%")
    print(f"\tEl error porcentual fuera de la muestra Eout es: {porcentual_error_out_sample}%")
    print("")
    wait_for_user_input()

    # Mostramos la recta que separa los datos
    print("Mostrando la recta que separa los datos sobre los datos de training")
    plot_frontier_line(X, Y, weights)
    print("")

    # Mostramos la grafica de nuestro modelo. En gris, pintaremos los valores
    # que se han predicho correctamente. En rojo, los valores en los que falla
    # la predicción
    # Mostramos esta grafica tanto para la muestra como para el dataset de test
    print("Mostrando grafica de predicciones en la muestra de entrenamiento")
    plot_classification_predictions(X, Y, weights, feature_names= ["Intensidad", "Simetria"], title = "Resultados en la muestra")
    wait_for_user_input()
    print("Mostrando grafica de predicciones en el conjunto de datos de test")
    plot_classification_predictions(X_test, Y_test, weights, feature_names= ["Intensidad", "Simetria"], title = "Resultados en el dataset de test")
    print("")
    wait_for_user_input()

    # Calculamos la regresion lineal con Minibatch Stochastic Gradient Descent
    # Calculamos tambien el error cometido para mostrar todos los resultados de golpe
    # Error tanto en la muestra comop fuera de la muestra, y ademas error de
    # clasificacion como error cuadratico medio
    print("Calculamos los pesos de la regresion lineal usando el algoritmo Stochastic Gradient Descent general")
    # Parametros para gradient descent
    learning_rate = 0.01
    batch_size = 32
    max_minibatch_iterations = 200

    # Solucion inicial con todo ceros
    # Tomamos el numero de columnas para el tamaño de nuestro vector solucion, pues
    # las columnas indican la dimension de los datos, mientras que las filas indican
    # el numero de datos
    starting_solution = np.zeros(np.shape(X)[1])

    weights, error_at_epoch, error_at_minibatch = stochastic_gradient_descent(X, Y, starting_solution, learning_rate, batch_size, max_minibatch_iterations, target_error = 0.01, verbose = True)
    error_in_sample = clasiffication_error(X, Y, weights)
    error_out_sample = clasiffication_error(X_test, Y_test, weights)
    mean_square_error_in_sample = clasiffication_mean_square_error(X, Y, weights)
    mean_square_error_out_sample = clasiffication_mean_square_error(X_test, Y_test, weights)
    porcentual_error_in_sample = classification_porcentual_error(X, Y, weights)
    porcentual_error_out_sample = classification_porcentual_error(X_test, Y_test, weights)
    print(f"\tLos pesos obtenidos son: {weights}")
    print(f"\tEl error de clasficacion en la muestra Ein es: {error_in_sample}")
    print(f"\tEl error de clasificacion fuera de la muestra Eout es: {error_out_sample}")
    print(f"\tEl error cuadratico medio en la muestra Ein es: {mean_square_error_in_sample}")
    print(f"\tEl error cuadratico medio fuera de la muestra Eout es: {mean_square_error_out_sample}")
    print(f"\tEl error porcentual en la muestra Ein es: {porcentual_error_in_sample}%")
    print(f"\tEl error porcentual fuera de la muestra Eout es: {porcentual_error_out_sample}%")
    print("")
    wait_for_user_input()

    # Mostramos la recta que separa los datos
    print("Mostrando la recta que separa los datos sobre los datos de training")
    plot_frontier_line(X, Y, weights)
    print("")

    # Mostramos la grafica de nuestro modelo. En gris, pintaremos los valores
    # que se han predicho correctamente. En rojo, los valores en los que falla
    # la predicción
    # Mostramos esta grafica tanto para la muestra como para el dataset de test
    print("Mostrando grafica de predicciones en la muestra de entrenamiento")
    plot_classification_predictions(X, Y, weights, feature_names= ["Intensidad", "Simetria"], title = "Resultados en la muestra")
    wait_for_user_input()
    print("Mostrando grafica de predicciones en el conjunto de datos de test")
    plot_classification_predictions(X_test, Y_test, weights, feature_names= ["Intensidad", "Simetria"], title = "Resultados en el dataset de test")
    print("")
    wait_for_user_input()

    # Mostramos la evolucion del error por cada epoca
    print("Evolucion del error por cada epoca de entrenamiento")
    plot_error_evolution(
        error_at_epoch,
        title = f"Evolucion del error por EPOCH para eta = {learning_rate}, batch_size = {batch_size}",
        x_label = "Epoch"
    )
    print("")


    # Mostramos la evolucion del error por cada iteracion en minibatch
    print("Evolucion del error por cada iteracion sobre minibatch")
    plot_error_evolution(
        error_at_minibatch,
        title = f"Evolucion del error por Minibatch para eta = {learning_rate}, batch_size = {batch_size}",
        x_label = "Minibatch Iteration"
    )
    print("")

def generate_labels_for_simula_unif(data):
    """
    Genera las etiquetas tal y como se indica en el guion de practicas para los
    datos generados aleatoriamente

    Parameters:
    ===========
    data: la matriz de datos que queremos etiquetar
    """

    # Funcion de etiquetado, que no incluye el ruido
    # La dejo dentro de esta funcion porque creo que asi queda mas claro el objetivo
    # de esta funcion de etiquetado
    def label_function(x, y):
        # Funcion dada en el guion de la practica
        label = np.sign(np.power(x - 0.2, 2.0) + np.power(y, 2.0) - 0.6)
        return label

    # Devuelve el 10% de los indices aleatorios para que cambiemos los signos
    # La dejo dentro de esta funcion porque creo que asi queda mas claro el objetivo
    # de esta funcion
    def generate_noisy_indixes(data):
        # Los indices ordenados
        indixes = np.arange(len(data))

        # Mezclamos los indices
        np.random.shuffle(indixes)

        # Devolvemos el 10% de los datos
        ten_percent = int(0.1 * len(data))
        return indixes[0:ten_percent]


    # Etiquetamos los datos sin introducir ruido
    labels = label_function(data[:, 0], data[:, 1])

    # Indices aleatorios para cambiar el signo
    indixes_to_change = generate_noisy_indixes(data)

    # Introducimos el ruido sobre esos datos
    for index in indixes_to_change:
        labels[index] = labels[index] * (-1)

    return np.array(labels)

def experiment_linear(iterations: int = 1000):
    """
    Realiza el experimento indicado en el ejercicio 2 Apartado 2 Subapartado c)

    Parameters:
    ===========
    iterations: el numero de veces que repetimos el experimento

    Returns:
    ========
    mean_perc_error_in_sample: media de los errores percentuales del experimento en la muestra
    mean_perc_error_out_sample: media de los errores percentuales del experimento fuera de la muestra
    mean_mean_square_error_in_sample:media de los errores cuadraticos medios del experimento en la muestra
    mean_mean_square_error_out_sample:media de los errores cuadraticos medios del experimento fuera de la muestra
    """

    # Parametros para minibatch gradient descent
    batch_size = 32
    max_minibatch_iterations = 200
    learning_rate = 0.01

    # Errores que vamos a ir calculando para devolver su media
    mean_perc_error_in_sample = 0
    mean_perc_error_out_sample = 0
    mean_mean_square_error_in_sample = 0
    mean_mean_square_error_out_sample = 0

    for _ in range(iterations):
        # Generamos la muestra de entrenamiento de 1000 puntos en el cuadrado [-1, 1] x [-1, 1]
        # Generamos tambien las etiquetas del conjunto de entrenamiento
        X = simula_unif(1000, 2, 1)
        labels = generate_labels_for_simula_unif(X)

        # Ahora generamos la muestra de testing y sus etiquetas
        test_data = simula_unif(1000, 2, 1)
        test_labels = generate_labels_for_simula_unif(test_data)

        # Clasificamos estos datos usando regresion lineal
        # Para ello, necesitamos que la matriz de datos contenga una primera columna
        # de unos para representar que estamos usando el vector de caracteristicas
        # (1, x1, x2) (termino independiente en los sumandos de las ecuaciones lineales)

        # Añadimos la columna de unos al training
        number_of_rows = int(np.shape(X)[0])
        new_column = np.ones(number_of_rows)
        X = np.insert(X, 0, new_column, axis = 1)

        # Añadimos la columna de unos al testing
        number_of_rows = int(np.shape(test_data)[0])
        new_column = np.ones(number_of_rows)
        test_data = np.insert(test_data, 0, new_column, axis = 1)

        # Ceros con el numero de columnas de nuestra matriz de datos
        starting_solution = np.zeros(np.shape(X)[1])

        # Ejecutamos minibatch gradient descent y calculamos los errores
        # verbose = False porque no queremos la evolucion del error a traves de las
        # iteraciones, y asi el algoritmo va mas rapido
        weights, _, _ = stochastic_gradient_descent(X, labels, starting_solution, learning_rate, batch_size, max_minibatch_iterations, verbose = False)
        mean_perc_error_in_sample += classification_porcentual_error(X, labels, weights)
        mean_perc_error_out_sample += classification_porcentual_error(test_data, test_labels, weights)
        mean_mean_square_error_in_sample += clasiffication_mean_square_error(X, labels, weights)
        mean_mean_square_error_out_sample += clasiffication_mean_square_error(test_data, test_labels, weights)

    # Calculamos la media de los errores acumulados
    mean_perc_error_in_sample /= iterations
    mean_perc_error_out_sample /= iterations
    mean_mean_square_error_in_sample /= iterations
    mean_mean_square_error_out_sample /= iterations

    # Devolvemos estas medias
    return mean_perc_error_in_sample, mean_perc_error_out_sample, mean_mean_square_error_in_sample, mean_mean_square_error_out_sample

def experiment_non_linear(iterations: int = 1000, max_minibatch_iterations = 200):
    """
    Realiza el mismo experimento que en experiment_linear, pero usando otro vector
    de caracteristicas: (1, x1, x2, x1*x2, x1**2, x2**2)

    Parameters:
    ===========
    iterations: el numero de veces que repetimos el experimento
    max_minibatch_iterations: numero maximo de iteraciones sobre minimbatch, por defecto 200

    Returns:
    ========
    mean_perc_error_in_sample: media de los errores percentuales del experimento en la muestra
    mean_perc_error_out_sample: media de los errores percentuales del experimento fuera de la muestra
    mean_mean_square_error_in_sample:media de los errores cuadraticos medios del experimento en la muestra
    mean_mean_square_error_out_sample:media de los errores cuadraticos medios del experimento fuera de la muestra
    """

    # Parametros para minibatch gradient descent
    batch_size = 32
    learning_rate = 0.01

    # Errores que vamos a ir calculando para devolver su media
    mean_perc_error_in_sample = 0
    mean_perc_error_out_sample = 0
    mean_mean_square_error_in_sample = 0
    mean_mean_square_error_out_sample = 0

    for _ in range(iterations):
        # Generamos la muestra de entrenamiento de 1000 puntos en el cuadrado [-1, 1] x [-1, 1]
        # Generamos tambien las etiquetas del conjunto de entrenamiento
        X = simula_unif(1000, 2, 1)
        labels = generate_labels_for_simula_unif(X)

        # Ahora generamos la muestra de testing y sus etiquetas
        test_data = simula_unif(1000, 2, 1)
        test_labels = generate_labels_for_simula_unif(test_data)

        # Clasificamos estos datos usando regresion lineal
        # Usamos el vector de caracteristicas (1, x1, x2, x1*x2, x1**2, x2**2)

        # Añadimos la columna de unos al training
        number_of_rows = int(np.shape(X)[0])
        new_column = np.ones(number_of_rows)
        X = np.insert(X, 0, new_column, axis = 1)

        # Añadimos la tercera columna x1 * x2
        third_col = X[:, 1] * X[:, 2]
        X = np.insert(X, 3, third_col, axis = 1)

        # Añadimos la cuarta columna x1**2
        fourth_col = X[:, 1] * X[:, 1]
        X = np.insert(X, 4, fourth_col, axis = 1)

        # Añadimos la quinta columna x2**2
        last_col = X[:, 2] * X[:, 2]
        X = np.insert(X, 5, last_col, axis = 1)

        # Aplicamos la misma modificacion a la matriz de testing
        number_of_rows = int(np.shape(test_data)[0])
        new_column = np.ones(number_of_rows)
        test_data = np.insert(test_data, 0, new_column, axis = 1)

        third_col = test_data[:, 1] * test_data[:, 2]
        test_data = np.insert(test_data, 3, third_col, axis = 1)

        fourth_col = test_data[:, 1] * test_data[:, 1]
        test_data = np.insert(test_data, 4, fourth_col, axis = 1)

        last_col = test_data[:, 2] * test_data[:, 2]
        test_data = np.insert(test_data, 5, last_col, axis = 1)

        # Ceros con el numero de columnas de nuestra matriz de datos
        starting_solution = np.zeros(np.shape(X)[1])

        # Ejecutamos minibatch gradient descent y calculamos los errores
        # verbose = False porque no queremos la evolucion del error a traves de las
        # iteraciones, y asi el algoritmo va mas rapido
        weights, _, _ = stochastic_gradient_descent(X, labels, starting_solution, learning_rate, batch_size, max_minibatch_iterations, verbose = False)
        mean_perc_error_in_sample += classification_porcentual_error(X, labels, weights)
        mean_perc_error_out_sample += classification_porcentual_error(test_data, test_labels, weights)
        mean_mean_square_error_in_sample += clasiffication_mean_square_error(X, labels, weights)
        mean_mean_square_error_out_sample += clasiffication_mean_square_error(test_data, test_labels, weights)

    # Calculamos la media de los errores acumulados
    mean_perc_error_in_sample /= iterations
    mean_perc_error_out_sample /= iterations
    mean_mean_square_error_in_sample /= iterations
    mean_mean_square_error_out_sample /= iterations

    # Devolvemos estas medias
    return mean_perc_error_in_sample, mean_perc_error_out_sample, mean_mean_square_error_in_sample, mean_mean_square_error_out_sample



def ejercicio2_apartado2():
    # Generamos la muestra de entrenamiento de 1000 puntos en el cuadrado [-1, 1] x [-1, 1]
    print("Generando la muestra de datos de entrenamiento y mostrando su gracica")
    X = simula_unif(1000, 2, 1)

    # Separo las coordedas x1 de las coordenadas x2 (o equivalentemente (x, y))
    scatter_plot(X[:, 0], X[:, 1], title = "Datos generados aleatoriamente")

    # Etiquetamos los puntos tal y como se indica en el guion de practicas
    # Una vez etiquetados, mostramos los puntos ya etiquetados, usando colores
    # ignore_first_column porque en este caso no trabajamos con una matriz de datos
    # con la primera columna de unos para representar el sumando del termino independiente,
    # como si pasa con otras matrices de la practica
    labels = generate_labels_for_simula_unif(X)
    print("Mostrando etiquetado de los datos previamente generados")
    scatter_plot_with_classes(X, labels, ["Puntos positivos", "Puntos negativos"],  ["Valor en X", "Valor en Y"], title = "Etiquetado de los datos generados", ignore_first_column = False)
    print("")

    # Clasificamos estos datos usando regresion lineal, como en el anterior apartado
    # Para ello, necesitamos que la matriz de datos contenga una primera columna
    # de unos para representar que estamos usando el vector de caracteristicas
    # (1, x1, x2) (termino independiente en los sumandos de las ecuaciones lineales)

    # Añadimos la columna de unos
    number_of_rows = int(np.shape(X)[0])
    new_column = np.ones(number_of_rows)
    X = np.insert(X, 0, new_column, axis = 1)


    # Lanzamos minibatch stochastic gradient descent
    # Usamos los parametros del ejercicio anterior porque nos han funcionado muy bien
    batch_size = 32
    max_minibatch_iterations = 200
    learning_rate = 0.01

    # Ceros con el numero de columnas de nuestra matriz de datos
    starting_solution = np.zeros(np.shape(X)[1])

    # Ejecutamos el algoritmo y calculamos el error in sample (no tenemos muestra
    # de datos de testing)
    print("Lanzando Stochastic Gradient Descent")
    weights, error_at_epoch, error_at_minibatch = stochastic_gradient_descent(X, labels, starting_solution, learning_rate, batch_size, max_minibatch_iterations, verbose = True)
    error_in_sample = clasiffication_error(X, labels, weights)
    mean_square_error_in_sample = clasiffication_mean_square_error(X, labels, weights)
    porcentual_error_in_sample = classification_porcentual_error(X, labels, weights)
    print(f"\tLos pesos obtenidos son: {weights}")
    print(f"\tEl error de clasficacion en la muestra Ein es: {error_in_sample}")
    print(f"\tEl error cuadratico medio en la muestra Ein es: {mean_square_error_in_sample}")
    print(f"\tEl error porcentual en la muestra Ein es: {porcentual_error_in_sample}%")
    print("")
    wait_for_user_input()

    # Mostramos la evolucion del error
    print("Mostrando la evolucion del error por iteracion en minibatch")
    plot_error_evolution(error_at_minibatch, title = "Error por cada iteracion sobre el minibatch", x_label = "Error por iteracion de minibatch")
    print("")

    # Mostramos los puntos en los que falla el clasificador
    print("Mostrando los puntos en los que falla el clasificador entrenado")
    plot_classification_predictions(X, labels, weights, feature_names = ["Valor de X", "Valor de Y"], title = "Puntos en los que fallamos en la prediccion")

    # Ahora realizamos el experimento 1000 veces, como se indica en el guion,
    # simulando datos en cada iteracion, tanto de training como de test
    print("Lanzamos el experimento 1000 veces, usando el vector de caracteristicas (1, x1, x2)")
    mean_perc_error_in_sample, mean_perc_error_out_sample, mean_mean_square_error_in_sample, mean_mean_square_error_out_sample = experiment_linear(1000)

    print("Los valores medios de los errores del experimento tras 1000 iteraciones son:")
    print(f"\tError medio porcentual DENTRO de la muestra: {mean_perc_error_in_sample}%")
    print(f"\tError medio porcentual FUERA de la muestra: {mean_perc_error_out_sample}%")
    print(f"\tError medio cuadratico medio DENTRO de la muestra: {mean_mean_square_error_in_sample}")
    print(f"\tError medio cuadratico medio FUERA de la muestra: {mean_mean_square_error_out_sample}")
    print("")
    wait_for_user_input()

    # Ahora realizamos el mismo experimento pero con un vector de caracteristicas
    # no lineal, como se indica en el guion de practicas
    print("Lanzamos el experimento 1000 veces, usando el vector de caracteristicas (1, x1, x2, x1*x2, x1**2, x2**2)")
    mean_perc_error_in_sample, mean_perc_error_out_sample, mean_mean_square_error_in_sample, mean_mean_square_error_out_sample = experiment_non_linear(1000)

    print("Los valores medios de los errores del experimento tras 1000 iteraciones son:")
    print(f"\tError medio porcentual DENTRO de la muestra: {mean_perc_error_in_sample}%")
    print(f"\tError medio porcentual FUERA de la muestra: {mean_perc_error_out_sample}%")
    print(f"\tError medio cuadratico medio DENTRO de la muestra: {mean_mean_square_error_in_sample}")
    print(f"\tError medio cuadratico medio FUERA de la muestra: {mean_mean_square_error_out_sample}")
    print("")
    wait_for_user_input()

    # Lanzamos el algoritmo para un ejemplo unicamente, y con ello mostramos
    # algunas graficas
    # Para ello lo primero que tenemos que hacer es transformar la matriz de datos
    # Partimos de un nuevo conjunto de datos al que le añadimos las nuevas columnas

    # Generamos la muestra de entrenamiento de 1000 puntos en el cuadrado [-1, 1] x [-1, 1]
    # Generamos tambien las etiquetas del conjunto de entrenamiento
    X = simula_unif(1000, 2, 1)
    labels = generate_labels_for_simula_unif(X)

    # Ahora generamos la muestra de testing y sus etiquetas
    test_data = simula_unif(1000, 2, 1)
    test_labels = generate_labels_for_simula_unif(test_data)

    # Clasificamos estos datos usando regresion lineal
    # Usamos el vector de caracteristicas (1, x1, x2, x1*x2, x1**2, x2**2)

    # Añadimos la columna de unos al training
    number_of_rows = int(np.shape(X)[0])
    new_column = np.ones(number_of_rows)
    X = np.insert(X, 0, new_column, axis = 1)

    # Añadimos la tercera columna x1 * x2
    third_col = X[:, 1] * X[:, 2]
    X = np.insert(X, 3, third_col, axis = 1)

    # Añadimos la cuarta columna x1**2
    fourth_col = X[:, 1] * X[:, 1]
    X = np.insert(X, 4, fourth_col, axis = 1)

    # Añadimos la quinta columna x2**2
    last_col = X[:, 2] * X[:, 2]
    X = np.insert(X, 5, last_col, axis = 1)

    # Aplicamos la misma modificacion a la matriz de testing
    number_of_rows = int(np.shape(test_data)[0])
    new_column = np.ones(number_of_rows)
    test_data = np.insert(test_data, 0, new_column, axis = 1)

    third_col = test_data[:, 1] * test_data[:, 2]
    test_data = np.insert(test_data, 3, third_col, axis = 1)

    fourth_col = test_data[:, 1] * test_data[:, 1]
    test_data = np.insert(test_data, 4, fourth_col, axis = 1)

    last_col = test_data[:, 2] * test_data[:, 2]
    test_data = np.insert(test_data, 5, last_col, axis = 1)

    # Ceros con el numero de columnas de nuestra matriz de datos
    starting_solution = np.zeros(np.shape(X)[1])

    # Ahora ejecutamos minibatch gradient descent y calculamos los errores
    # verbose = False porque no queremos la evolucion del error a traves de las
    # iteraciones, y asi el algoritmo va mas rapido
    print("Lanzando Stochastic Gradient Descent para un caso concreto, con vector de caracteristicas no linelaes")

    # El valor de max_minibatch_iterations lo hemos ido variando para realizar el
    # analisis plasmado en la memoria. Ha pasado por los valores 200, 400, 600 y 800
    weights, error_at_epoch, error_at_minibatch = stochastic_gradient_descent(X, labels, starting_solution, learning_rate, batch_size, max_minibatch_iterations=800, verbose = True)
    error_in_sample = clasiffication_error(X, labels, weights)
    mean_square_error_in_sample = clasiffication_mean_square_error(X, labels, weights)
    porcentual_error_in_sample = classification_porcentual_error(X, labels, weights)
    print(f"\tLos pesos obtenidos son: {weights}")
    print(f"\tEl error de clasficacion en la muestra Ein es: {error_in_sample}")
    print(f"\tEl error cuadratico medio en la muestra Ein es: {mean_square_error_in_sample}")
    print(f"\tEl error porcentual en la muestra Ein es: {porcentual_error_in_sample}%")
    print("")
    wait_for_user_input()

    # Mostramos la evolucion del error
    print("Mostrando la evolucion del error por iteracion en minibatch")
    plot_error_evolution(error_at_minibatch, title = "Error por cada iteracion sobre el minibatch", x_label = "Error por iteracion de minibatch")
    print("")

    # Mostramos los puntos en los que falla el clasificador
    print("Mostrando los puntos en los que falla el clasificador entrenado")
    plot_classification_predictions(X, labels, weights, feature_names = ["Valor de X", "Valor de Y"], title = "Puntos en los que fallamos en la prediccion")

    # Repetimos el experimento mil veces, pero cambiando el valor de max_minibatch_iterations
    # pues explorando dicho parametro vemos que se puede mejorar mucho mas el error
    print("Repitiendo el experimento pero cambiando el valor de max_minibatch_iterations a 800")
    mean_perc_error_in_sample, mean_perc_error_out_sample, mean_mean_square_error_in_sample, mean_mean_square_error_out_sample = experiment_non_linear(1000, max_minibatch_iterations = 800)

    print("Los valores medios de los errores del experimento tras 1000 iteraciones y habiendo cambiado max_minibatch_iterations a 800 son:")
    print(f"\tError medio porcentual DENTRO de la muestra: {mean_perc_error_in_sample}%")
    print(f"\tError medio porcentual FUERA de la muestra: {mean_perc_error_out_sample}%")
    print(f"\tError medio cuadratico medio DENTRO de la muestra: {mean_mean_square_error_in_sample}")
    print(f"\tError medio cuadratico medio FUERA de la muestra: {mean_mean_square_error_out_sample}")
    print("")
    wait_for_user_input()

def ejercicio2():
    print("Ejecutando ejercicio 2")

    print("Apartado 1)")
    print("=" * 80)
    ejercicio2_apartado1()
    print("")

    print("Apartado 2)")
    print("=" * 80)
    ejercicio2_apartado2()


# Ejercicio Bonus
#===============================================================================

# Corremos todos los ejercicios
#===============================================================================
if __name__ == "__main__":
    # Establecemos la semilla aleatoria para trabajar con resultados reproducibles
    np.random.seed(123456789)

    ejercicio1()
    ejercicio2()
