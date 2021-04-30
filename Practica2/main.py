"""
Practica 2: Aprendizaje Automatico
Sergio Quijano Rey, sergioquijano@correo.ugr.es
Enlaces usados:
    [1]: https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray
    [2]: https://glowingpython.blogspot.com/2012/01/how-to-plot-two-variable-functions-with.html
"""

import numpy as np
import matplotlib.pyplot as plt

# Para contar los numeros de apariciones de cierto elemento en un array numpy
import collections

# Funciones auxiliares
#===================================================================================================


def wait_for_user_input():
    """Esperamos a que el usuario pulse una tecla para continuar con la ejecucion"""
    input("Pulse ENTER para continuar...")


def get_straight_line(a, b):
    """Devuelve la funcion recta de la forma a*x + b"""
    return lambda x: a * x + b

def get_straight_line_from_implicit(weights):
    """Devuelve la recta y = ax + b a partir de los pesos que representan una funcion lineal
    implicita ax + by + c = label, haciendo label = 0

    En concreto, tenemos la funcion implicita en x, y: w0 + w1 x + w2 y = label
    """
    return lambda x: (1 / weights[2]) * (-weights[0] - x * weights[1])

def get_frontier_function(weights):
    """
    Dado un clasificador lineal de la forma w0 + w1x + w2y, devuelve la recta y = f(x) frontera del
    clasificador lineal

    Parameters:
    ===========
    weights: pesos del clasificador. np.array de tres elementos
    """
    return lambda x: (1.0 / weights[2]) * (-weights[0] - weights[1] * x)

def add_colum_of_ones(matrix):
    """
    Dada una matriz, devuelve la misma matriz a la que añadimos una columna de unos. Esta columna de
    unos se coloca en la primera columna de la matriz. Este añadido se realiza para representar el
    termino independiente de una combinacion lineal

    Parameters:
    ===========
    matrix: la matriz a la que añadimos la nueva columna
            No se modifica

    Returns:
    ========
    new_matrix: la matriz con la nueva columna
    """

    # Para evita que la matriz original se modifique
    new_matrix = matrix

    # Tomamos el numero de filas que tiene la matriz
    number_of_rows = int(np.shape(new_matrix)[0])

    # Generamos la nueva columna de unos
    new_column = np.ones(number_of_rows)

    # Añadimos la columna de unos
    new_matrix = np.insert(new_matrix, 0, new_column, axis = 1)

    return new_matrix


# Valores de las etiquetas
#===================================================================================================
label_pos = 1
label_neg = -1

# Variables globales
#===================================================================================================
global_dataset = None
global_labels = None
global_lin_coeffs = None
global_noisy_labels = None

# Funciones dadas por los profesores
#===================================================================================================


def simula_unif(N, dim, rango):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    media = 0
    out = np.zeros((N, dim), np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    """Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores"""
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1)  # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b


def readData(file_x, file_y, digits, labels):
    """
    Funcion dada por los profesores para leer un fichero de datos
    Funcion COPIADA COMPLETAMENTE de la plantilla dada por los profesores

    Parameters:
    file_x: fichero con las caracteristicas de los digitos
    file_y: fichero con las etiquetas de los puntos
    digits: clases que queremos extraer. Vector con dos elemento indicando los digitos que queremos
            extraer del fichero
    labels: etiquetas que queremos asignar a los dos digitos

    Returns:
    ========
    x: matriz con las caracteristicas de los datos. Incluye la columna de unos para representar el
       termino independiente en las combinaciones lineales
    y: vector con las etiquetas de los datos
    """
    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
    for i in range(0, datay.size):
        if datay[i] == digits[0] or datay[i] == digits[1]:
            if datay[i] == digits[0]:
                y.append(labels[0])
            else:
                y.append(labels[1])
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y

# Graficos
#===================================================================================================


def scatter_plot(x_values, y_values, title="Scatter Plot Simple", x_label="Eje X", y_label="Eje Y"):
    """
    Grafico simple tipo scatter plot, sin colorear segun clases porque no tenemos clases
    De nuevo, no tenemos clases que separar asi que pintamos todos los puntos del mismo color

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
    ax.scatter(x_values, y_values, c="grey", alpha=0.6)

    # Titulo para la grafica
    plt.title(title)

    # Añado las leyendas en los ejes
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Muestro el grafico
    plt.show()
    wait_for_user_input()


def scatter_plot_with_classes(data, classes, target_names, feature_names, title, ignore_first_column: bool = False, show: bool = True, canvas=None):
    """
    Hacemos un scatter plot de puntos con dos coordenadas que estan etiquetados en distintos grupos

    Parameters:
    ===========
    data: coordenadas de los distintos puntos
    classes: etiquetas numericas de las clases a la que pertenencen los datos
    target_names: nombres que le doy a cada una de las clases
    feature_names: nombre de los ejes de coordenadas que le queremos dar al grafico
    ignore_first_column: indica si tenemos que ignorar o no la primera columna
                         Esto pues en algunos casos la primera columna de la matriz de datos
                         tiene todo unos para representar el sumando del termino independiente
                         en las ecuaciones lineales. En esta practica, no va a ser el caso
    show: indica si queremos mostrar o no la grafica
          Nos puede interesar no mostrar la grafica para añadir nuevos elementos
          a esta grafica sin tener que repetir codigo
    canvas: fig, ax que se obtienen con plt.subplots(). Cuando vale None, realizamos la llamada
            dentro de la funcion, generando asi una grafica desde cero. Cuando no valen None, usamos
            los parametros pasados. Este ultimo caso se usa para llamar primero a esta funcion, y
            despues dibujar sobre el grafico que esta funcion genere sin que se separe en dos graficos
            distintos. Es claro que para que tenga buen comportamiento, canvas != None => show == false

    data y classes ya en un tipo de dato numpy para poder operar con ellos

    Un ejemplo de canvas != None es hacer un contourf de una funcion y despues, encima del contourf,
    hacer el scatterplot. Si hiciesemos el scatterplot primero, el contourf, al estar encima, taparia
    por completo los puntos pintados.

    De nuevo, si tomamos fig, ax dentro de esta funcion buscando hacer lo anterior, generariamos dos
    graficos por separado, uno con el scatter y otro con el contourf, y este no es el comportamiento
    que esperamos
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

    # Tomo los elementos para dibujar la grafica elaborada
    fig, ax = None, None

    # Los genero desde dentro porque el caller no ha pasado valor
    if canvas is None:
        fig, ax = plt.subplots()

    # Los tomo desade parametro porque el caller si que ha pasado valor
    else:
        fig, ax = canvas[0], canvas[1]

    # Itero sobre las clases
    for index, target_name in enumerate(target_names):

        # Tomo las coordenadas de la clase index-esima
        current_x = x_values[splitted_indixes[index]]
        current_y = y_values[splitted_indixes[index]]

        # Muestro la clase index-esima, con su color y su etiqueta correspondiente
        # Ponemos alpha para apreciar donde se acumulan muchos datos (que seran
        # zonas mas oscuras que aquellas en las que no hay acumulaciones)
        ax.scatter(current_x, current_y,
                   c=colormap[index], label=target_name, alpha=0.6)

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


def scatter_plot_with_classes_and_labeling_function(data, classes, target_names, feature_names, title, labeling_function, ignore_first_column = False):
    """
    Hacemos un scatter plot de puntos con dos coordenadas que estan etiquetados a partir de una
    funcion de etiquetado que mostramos en la grafica. Notar que las etiquetas pueden tener ruido,
    por lo que la grafica puede mostrar puntos mal etiquetados segun la funcion dada
    Ademas, mostramos la funcion que ha sido usada para etiquetar.

    Se usa esta funcion cuando podemos expresar y = f(x) de forma global, pues recordar que las
    funciones de etiquetado son funciones de dos variables implicitas

    Parameters:
    ===========
    data: coordenadas de los distintos puntos
    classes: etiquetas numericas de las clases a la que pertenencen los datos
    target_names: nombres que le doy a cada una de las clases
    feature_names: nombre de los ejes de coordenadas que le queremos dar al grafico
    title: titulo que le queremos poner a la grafica
    labeling_function: funcion de etiquetado que ha generado la parte deterministica del etiquetas
                       (recordar que las etiquetas pueden tener ruido)
                       Debe ser una funcion de una variable despejada para el valor de y, es decir,
                       en la forma y = f(x), globalmente
    ignore_first_column: si queremos ignorar una primera columna de unos añadida para representar el
                         termino independiente de un modelo lineal
    """

    # Usamos la funcion que hace scatter plot de los datos etiquetados
    # Hacemos show = False para que no se muestre la grafica, porque queremos seguir haciendo
    # modificaciones sobre esta
    scatter_plot_with_classes(
        data, classes, target_names, feature_names, title, ignore_first_column = ignore_first_column, show=False)

    # Establecemos la columna que representa el valor x
    # Depende de si tenemos una primera columna de unos o no
    x_col = 0
    if ignore_first_column == True:
        x_col = 1

    # Tomamos el valor minimo y maximo en el eje de ordenadas
    # Los escalamos un poco por encima para que la grafica de la funcion quede bien V
    lower_x = np.amin(data[:, x_col])
    upper_x = np.amax(data[:, x_col])
    lower_x = lower_x * 1.1
    upper_x = upper_x * 1.1

    # Generamos un mapeado de los valores para dibujar la grafica de la funcion
    resolution = 1000  # Numero de puntos que vamos a usar para dibujar la grafica de la funcion
    x_values = np.linspace(lower_x, upper_x, resolution)
    y_values = labeling_function(x_values)

    # Generamos la grafica
    plt.plot(x_values, y_values)

    # Mostramos la grafica
    plt.show()
    wait_for_user_input()

def scatter_plot_with_classes_and_labeling_region(data, classes, target_names, feature_names, title, labeling_function):
    """
    Hacemos un scatter plot de puntos con dos coordenadas que estan etiquetados a partir de una
    funcion de etiquetado que mostramos en la grafica. Notar que las etiquetas pueden tener ruido,
    por lo que la grafica puede mostrar puntos mal etiquetados segun la funcion dada

    Mostramos las regiones positivas y negativas de la funcion. Lo hacemos asi porque la idea de
    esta funcion es ser usada cuando labeling_function no tenga un despeje de una variable en funcion
    de la otra variable global (las funciones de etiquetado son funciones implicitas)

    Podriamos trabajar con los despejes locales del teorema de la funcion implicita, pero al ser
    funciones locales da mucho trabajo a la hora de generar las graficas

    Parameters:
    ===========
    data: coordenadas de los distintos puntos
    classes: etiquetas numericas de las clases a la que pertenencen los datos
    target_names: nombres que le doy a cada una de las clases
    feature_names: nombre de los ejes de coordenadas que le queremos dar al grafico
    title: titulo que le queremos poner a la grafica
    labeling_function: funcion de etiquetado que ha generado la parte deterministica del etiquetas
                       (recordar que las etiquetas pueden tener ruido)
                       Debe ser una funcion de dos variables de la que no podamos obtener una funcion
                       despejada global de la forma xi = f(xj). En caso de poder realizar este calculo
                       deberiamos usar scatter_plot_with_classes_and_labeling_function

    La idea para pintar una funcion real-valuada de dos variables con un contourf la tomo de [2]
    """

    # Tomamos el valor minimo y maximo en el eje de ordenadas
    # Los escalamos un poco por encima para que la grafica de la funcion quede bien
    lower_x = np.amin(data[:, 0])
    upper_x = np.amax(data[:, 0])
    lower_y = np.amin(data[:, 1])
    upper_y = np.amax(data[:, 1])
    lower_x = lower_x * 1.1
    upper_x = upper_x * 1.1
    lower_y = lower_y * 1.1
    upper_y = upper_y * 1.1

    # Generamos un mapeado de los valores para dibujar la grafica de la funcion
    resolution = 1000  # Numero de puntos que vamos a usar para dibujar la grafica de la funcion
    x_values = np.linspace(lower_x, upper_x, resolution)
    y_values = np.linspace(lower_y, upper_y, resolution)

    # Generamos una matriz con estos dos arrays para poder evaluar una funcion de dos variables
    # sobre todas las combinaciones (como si estuviesemos haciendo un producto cartesiano)
    X, Y = np.meshgrid(x_values, y_values)

    # Tomamos los valores de la funcion en lo anterio
    Z = np.sign(labeling_function(X, Y))

    # Tomamos los elementos para generar las graficas elaboradas
    fig, ax = plt.subplots()

    # Mostramos la grafica de la region de separacion
    plt.contourf(X, Y, Z)

    # Ahora generamos la grafica de scatter plot
    scatter_plot_with_classes(
        data=data,
        classes=classes,
        target_names=target_names,
        feature_names=feature_names,
        title=title,
        ignore_first_column=False,
        show=False,
        canvas=[fig, ax]  # Para evitar que se generen dos graficos distintos
    )

    # Mostramos la grafica compuesta
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

def plot_misclassified_classification_predictions(data, labels, labeling_function, feature_names, title: str = "Grafica de predicciones", ignore_first_column = False):
    """
    Mostramos un scatter plot en el que visualizamos que datos se han clasificado mal. Si predecimos
    correctamente la etiqueta, se pinta el punto en un color gris. Si se falla la prediccion, se
    pinta en un color rojo

    Parameters:
    ===========
    data: los datos de entrada sobre los que predecimos
    labels: los verdaderos valores que deberiamos predecir
    labeling_function: la funcion que se usa para clasificar
    feature_names: el nombre de las caracteristicas en base a las que hacemos
                   las predicciones
    ignore_first_column: si queremos ignorar una primera columna de unos añadida para representar el
                         termino independiente de un modelo lineal
    """
    # Tomo las coordenadas de la matriz de datos, es decir, separo coordenadas
    # x e y de una matriz de datos que contiene pares de coordenadas
    # Segun si ignoramos o no la primera columna, accedemos a unas posiciones u otras
    x_col, y_col = 0, 1
    if ignore_first_column == True:
        x_col, y_col = 1, 2

    x_values = data[:, x_col]
    y_values = data[:, y_col]

    # Predicciones sobre el conjunto de datos a partir de la funcion de etiquetado dada
    predictions = [labeling_function(x) for x in data]

    # Separo los indices en indices de puntos que hemos predicho correctamente
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

# Algoritmos
#===================================================================================================
def perceptron_learning_algorithm(dataset, labels, max_iterations, init_solution, verbose = False):
    """
    Algoritmo de aprendizaje para perceptron

    Parameters:
    ===========
    dataset: conjunto de puntos que vamos a clasificar. Cada punto tiene sus coordenadas en una fila
             de la matriz
    labels: vector de etiquetas
    max_iterations: numero maximo de iteraciones
                    Por iteracion consideramos una pasada completa sobre todos los datos
    init_solution: vector de pesos iniciales que representan la solucion inicial
    verbose: indica si queremos devolver mas datos de los estrictamente necesarios
             Al tener que evaluar el error, hace que el algoritmo corra mas lento

    Returns:
    ========
    current_solution: la solucion que se alcanza al final del proceso
    current_iteration: numero de iteraciones necesarias para alcanzar la solucion. Iteraciones sobre
                       todos los datos (por tanto, EPOCHs consumidas)
    error_at_iteration: error en cada iteracion (iteracion sobre todos los datos, sobre EPOCH).
                        Solo cuando verbose == True
    """

    # Valores iniciales para el algoritmo
    current_solution = init_solution
    current_iteration = 0
    error_at_iteration = [] # Se usa cuando verbose == True

    # Para controlar si debemos parar de iterar al haber encontrado una solucion que respeta todo
    # el etiquetado
    full_pass_without_changes = False

    # Iteramos sobre el algoritmo
    while full_pass_without_changes == False and current_iteration < max_iterations:
        full_pass_without_changes = True

        # Iteramos sobre todos los puntos del dataset junto a las correspondientes etiquetas
        for point, label in zip(dataset, labels):

            # A partir de los pesos, obtenemos la funcion de clasificacion
            perceptron = get_perceptron(current_solution)

            # Comprobamos que estamos etiquetando bien este punto
            if perceptron(point) != label:
                # Actualizamos los pesos y la funcion que representa
                current_solution = current_solution + label * point
                perceptron = get_perceptron(current_solution)

                # Estamos cambiando un dato, asi que esta pasada sobre los datos no es limpia
                full_pass_without_changes = False

        # Aumentamos la iteracion, pues hemos hecho una pasada completa sobre los datos
        current_iteration = current_iteration + 1

        # Añadimos el error en esta iteracion sobre epoch
        if verbose == True:
            curr_err = percentage_error(dataset, labels, current_solution)
            error_at_iteration.append(curr_err)


    # Devolvemos los resultados en el caso verbose
    if verbose == True:
        return current_solution, current_iteration, error_at_iteration

    # Devolvemos los resultados sin datos adicionales
    return current_solution, current_iteration

def get_perceptron(weights):
    """
    Devuelve la funcion de clasificacion perceptron representada por los pesos dados

    Parameters:
    ===========
    weights: pesos que queremos convertir a perceptron. Deben ser un np.array para poder operar
             con estos comodamente
    """
    return lambda x: np.sign(np.dot(weights.T, x))

def percentage_error(dataset, labels, weights):
    """
    Calcula el error porcentual de un perceptron

    Parameters:
    ===========
    dataset: conjunto de datos
    labels: etiquetado real de los datos
    weights: pesos que definen el perceptron del cual queremos calcular el error
    """
    # Tomamos el perceptron representado por los pesos dados
    perceptron = get_perceptron(weights)

    # Cantidad de puntos mal etiquetados
    misclassified_count = 0

    # Iteramos los puntos junto a sus etiquetas
    for point, label in zip(dataset, labels):
        if perceptron(point) != label:
            misclassified_count += 1

    # Devolvemos el porcentaje (en tantos por uno)
    return misclassified_count / len(dataset)

def percentage_error_function(dataset, labels, classifier):
    """
    Devuelve el porcentaje de puntos mal clasificados usando un clasificador generico (tenemos otras
    funciones para calcular porcentajes de clasificadores concretos como PLA o LGR)

    Parameters:
    ===========
    dataset: conjunto de datos
    labels: etiquetas verdaderas de los datos
    classifier: funcion de clasificado. Debe admitir como entrada un punto del dataset y devolver
                un valor que tenga sentido respecto a los labels que estamos pasando como paramtro

    Returns:
    ========
    El porcentaje de puntos mal clasificados, EN TANTOS POR UNO
    """

    # Cantidad de puntos mal etiquetados
    misclassified_count = 0

    # Iteramos los puntos junto a sus etiquetas
    for point, label in zip(dataset, labels):
        if classifier(point) != label:
            misclassified_count += 1

    # Devolvemos el porcentaje (en tantos por uno)
    return misclassified_count / len(dataset)

def stochastic_gradient_descent(data, labels, starting_solution, learning_rate: float = 0.001, batch_size: int = 1, max_minibatch_iterations: int = 200, target_error: float = None, target_epoch_delta: float = None, gradient_function = None, error_function = None, verbose: bool = False):

    """
    Implementa el algoritmo de Stochastic Gradient Descent, con minibatches

    Parameters:
    ===========
    data: datos de entrada sobre los que queremos hacer predicciones
    labels: verdaderos valores que queremos predecir. Pueden representar etiquetas
            de una categoria para clasificacion o valores reales para regresion.
            Gracias a las etiquetas podemos calcular el gradiente del error usando los datos de la
            muestra dada
    starting_solution: np.array del que parte las soluciones iterativas
    learning_rate: tasa de aprendizaje
    max_minibatch_iterations: maximo numero de iteraciones
                              Por iteracion entendemos cada vez que modificamos los pesos de la
                              solucion iterativa (ie. cada recorrido de un minibatch). No confundir
                              con el numero maximo de epochs
                              Si max_minibatch_iterations = None, no lo tenemos en cuenta
    target_error: error por debajo del cual dejamos de iterar
                  Puede ser None para indicar que no comprobemos el error para dejar de iterar
                  El error se comprueba en cada EPOCH completo, no sobre una iteracion concreta de
                  minibatches
    target_epoch_delta: variacion de las soluciones entre dos epochs consecutivas por debajo del
                        cual queremos estar.
                        Si es None, no lo tenemos en cuenta
                        Notar que target_error mide que estemos por debajo de un umbral de error.
                        Mientras que esta medida no tiene en cuenta valores del error, sino la
                        diferencia entre dos soluciones consecutivas
    gradient_function: funcion que toma la matriz de datos, el vector de etiquetas y el vector de
                       pesos actuales, y computa el gradiente usando los datos de la muestra
    error_function: funcion que toma la matriz de datos, el vector de etiquetas y el vector de pesos
                    actuales, y computa el error
    verbose: indica si queremos que se guarden metricas en cada epoch

    Returns:
    ========
    current_solution: solucion que se alcanza
    iterations_consumed: cuantas iteraciones de minibatch se han consumido en el proceso
                         Si verbose == True, se pueden calcular usando error_at_minibatch. Pero
                         cuando verbose == False, necesitamos devolver este valor
    epochs_consumed: cuantas iteraciones de epochs (pasadas con minibatches sobre todos los datos)
                     hemos consumido. Misma discusion que iterations_consumed y verbose
    error_at_epoch: error en cada EPOCH <-- Cuando verbose == True
    error_at_minibatch: error en cada iteracion sobre minibatch <-- Cuando verbose == True
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

    # Para llevar la cuenta de cuantos epochs hemos consumido
    current_epoch_iterations = 0

    # Si no tenemos max_minibatch_iterations iteramos sin control de contador
    # En otro caso, acotamos el numero maximo de iteraciones
    while max_minibatch_iterations is None or current_minibatch_iterations < max_minibatch_iterations:
        # Empezamos una nueva epoca
        current_epoch_iterations += 1

        # Generamos los minibatches a partir de los datos de entrada
        # Trabajamos por comodidad y eficiencia con indices, como se indica en la funcion
        mini_batches_index_groups = get_minibatches(data, batch_size)

        # Para calcular la diferencia de soluciones entre dos epochs consecutivos
        solution_at_last_epoch = current_solution

        # Iteramos en los minibatches
        for mini_batches_indixes in mini_batches_index_groups:
            # Tomo los datos y etiquetas asociadas a los indices de este minibatch
            minibatch_data = data[mini_batches_indixes]
            minibatch_labels = labels[mini_batches_indixes]

            # Calculo la aproximacion al gradiente con estos datos
            minibatch_sample_gradient = gradient_function(minibatch_data, minibatch_labels, current_solution)

            # Actualizo la solucion con este minibatch
            current_solution = current_solution - learning_rate * minibatch_sample_gradient

            # Añadimos el error sobre la iteracion del minibatch
            if verbose is True:
                # Tomamos la solucion como un array, no como una matriz de una unica fila
                # pues esto provoca fallos en otras funciones (como la del calculo del error)
                tmp_solution = current_solution
                if(len(np.shape(current_solution)) == 2):
                    tmp_solution = current_solution[0]

                error_at_minibatch.append(error_function(data, labels, tmp_solution))

            # Hemos hecho una pasada completa al minibatch, aumentamos el contador
            # y comprobamos si hemos superado el maximo (tenemos que hacer esta
            # compobracion por estar en un doble bucle)
            # Hacemos la comporbacion solo si max_minibatch_iterations no es None
            current_minibatch_iterations += 1
            if max_minibatch_iterations is not None and current_minibatch_iterations >= max_minibatch_iterations:
                break

        # Comprobamos si hemos alcanzado el error objetivo para dejar de iterar
        if target_error is not None:
            # Tomamos la solucion como un array, no como una matriz de una unica fila
            # pues esto provoca fallos en otras funciones (como la del calculo del error)
            tmp_solution = current_solution
            if(len(np.shape(current_solution)) == 2):
                tmp_solution = current_solution[0]

            if error_function(data, labels, tmp_solution) < target_error:
                break

        # Comprobamos si hemos alcanzado la variacion entre soluciones objetivo
        if target_epoch_delta is not None:
            # Calculamos la variacion con la solucion de la epoca anterior
            curr_delta = np.linalg.norm(current_solution - solution_at_last_epoch)

            # Realizamos la comprobacion
            if curr_delta < target_epoch_delta:
                break


        # Añadimos el error en este epoch
        if verbose is True:
            # Tomamos la solucion como un array, no como una matriz de una unica fila
            # pues esto provoca fallos en otras funciones (como la del calculo del error)
            tmp_solution = current_solution
            if(len(np.shape(current_solution)) == 2):
                tmp_solution = current_solution[0]

            error_at_epoch.append(error_function(data, labels, tmp_solution))


    # Devolvemos la solucion como un array, no como una matriz de una unica fila
    # pues esto provoca fallos en otras funciones (como la del calculo del error)
    if(len(np.shape(current_solution)) == 2):
        current_solution = current_solution[0]

    # Hacemos renaming de la variable para que quede mas explicito lo que estamos devolviendo
    iterations_consumed = current_minibatch_iterations
    epochs_consumed = current_epoch_iterations

    return current_solution, iterations_consumed, epochs_consumed, error_at_epoch, error_at_minibatch

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

def sigmoid(x):
    """Funcion sigmoide, que se usa extensivamente en logistic regression"""
    return 1.0 / (1.0 + np.exp(-x))

def logistic_gradient(dataset, labels, solution):
    """
    Gradiente para la regresion logistica usando una muestra de datos

    Parmeters:
    ==========
    dataset: conjunto de entrada de datos
    labels: etiquetas sobre los datos
    solution: pesos que representan el clasificador logistico solucion. Solucion respecto de la cual
              calculamos el gradiente
    """

    # Sumatoria recorriendo los puntos y sus etiquetas
    gradient = 0.0
    for point, label in zip(dataset, labels):
        signal = np.dot(solution.T, point)
        gradient += (label * point) / (1 + np.exp(label * signal))

    # Devolvemos la sumatoria por -1 / N
    return -gradient / len(dataset)

def logistic_error(dataset, labels, solution):
    """
    Error para la regresion logistica usando una muestra de datos. Es el error que vamos a optimizar,
    el que se deriva de maximizar el likelihood. No es tan intuitivo como el error porcentual, que
    se puede tomar llamando a percentage_logistic_error

    Parmeters:
    ==========
    dataset: conjunto de entrada de datos
    labels: etiquetas sobre los datos
    solution: pesos que representan el clasificador logistico solucion. Solucion de la que calculamos
              el error
    """
    err = 0.0

    # Iteramos los puntos con sus etiquetas para calcular la parte del sumatorio
    for point, label in zip(dataset, labels):
        err += (1 + np.exp(-label * np.dot(solution.T, point)))

    # Devolvemos la media de la anterior suma
    return err / len(dataset)

def get_logistic_classifier(weights):
    """
    Devuelve el clasificador asociado a los pesos que representan una solucion de regresion logistica

    La función sigmoide sobre la señal de entrada da un valor en [0, 1] que representa la probabilidad
    de que el dato pertenezca al label_pos. Por tanto, 1 - sigmoid(signal) es la probabilidad de que
    el dato pertenezca a label_neg.

    Para que el valor sigmoid(signal) sea interpretado en clasificacion, devolvemos label_pos si
    sigmoid esta por encima de un umbral. Devolvemos label_neg en otro caso

    Lo mas logido es definir el umbral como 0.5
    """
    # Cota que debe superarse para considerarse etiquetado positivo o negativo
    threshold = 0.5

    # Funcion que vamos a devolver
    def classifier(x):
        # Calculamos la probabilidad de ser de la clase label_pos
        signal = np.dot(weights.T, x)
        probability = sigmoid(signal)

        if probability < threshold:
            return label_neg
        else:
            return label_pos

    # Devolvemos la funcion de clasificacion
    return classifier

def percentage_logistic_error(dataset, labels, weights):
    """
    Calcula el porcentaje de puntos mal clasificados por regresion logistica

    Paramters:
    ==========
    dataset: conjunto de datos
    labels: etiqutas sobre los datos
    weights: pesos que representan el clasificador de regresion lineal

    Returns:
    ========
    percentage: porcentaje de puntos mal clasificados
    """

    # Tomamos el clasificador
    classifier = get_logistic_classifier(weights)

    # Miramos la cantidad de puntos mal clasificados
    missclasified = 0
    for point, label in zip(dataset, labels):
        if classifier(point) != label:
            missclasified += 1

    # Devolvemos la media de la suma
    percentage = missclasified / len(dataset)
    return percentage

def perceptron_learning_algorithm_pocket(dataset, labels, max_iterations, init_solution):
    """
    Algoritmo de aprendizaje para perceptron, modificacion Pocket (en cada iteracion guardamos la
    mejor solucion hasta el momento)

    Parameters:
    ===========
    dataset: conjunto de puntos que vamos a clasificar. Cada punto tiene sus coordenadas en una fila
             de la matriz
    labels: vector de etiquetas
    max_iterations: numero maximo de iteraciones sobre EPOCHS
    init_solution: vector de pesos iniciales que representan la solucion inicial

    Returns:
    ========
    best_solution: la solucion que se alcanza al final del proceso
    current_iteration: numero de iteraciones necesarias para alcanzar la solucion
    error_at_iteration: error en cada iteracion sobre datos, no sobre EPOCH
                        Al estar lanzando PLA-Pocket, es el error asociado a la mejor funcion
                        encontrada hasta el momento
                        No hacemos error por cada EPOCH porque como en Pocket tenemos que estar
                        evaluando el error dato a dato, y la grafica es monotona (se visualiza por
                        tanto bien), consideramos mas conveniente devolver error por cada iteracion
                        de dato

    Como vamos a estar usando el error para best_solution, no tiene sentido distinguir si queremos
    verbose o no: siempre usamos modo 'verbose'
    """

    # Valores iniciales para el algoritmo
    current_solution = init_solution
    best_solution = init_solution
    current_iteration = 0
    error_at_iteration = []

    # Error de la mejor solucion hasta el momento
    # Esto nos evita calcular el error una y otra vez sobre la misma solucion mejor hasta el momento
    best_solution_error = percentage_error(dataset, labels, best_solution)

    # Iteramos sobre el algoritmo hasta alcanzar error cero o hasta agotar las iteraciones
    while best_solution_error > 0 and current_iteration < max_iterations:

        # Iteramos sobre todos los puntos del dataset junto a las correspondientes etiquetas
        for point, label in zip(dataset, labels):

            # A partir de los pesos, obtenemos la funcion de clasificacion
            perceptron = get_perceptron(current_solution)

            # Comprobamos que estamos etiquetando bien este punto
            if perceptron(point) != label:
                # Actualizamos los pesos y la funcion que representa
                current_solution = current_solution + label * point
                perceptron = get_perceptron(current_solution)

                # Hemos cambiado el valor de los pesos y hemos obtenido una mejor solucion
                curr_err = percentage_error(dataset, labels, current_solution)
                if curr_err < best_solution_error:
                    best_solution = current_solution
                    best_solution_error = curr_err

                # Añadimos el error de la mejor solucion
                error_at_iteration.append(best_solution_error)

        # Aumentamos la iteracion, pues hemos hecho una pasada completa sobre los datos
        current_iteration = current_iteration + 1

        if current_iteration % 10 == 0:
            print(f"--> Iteracion {current_iteration}")

    # Devolvemos los resultados sin datos adicionales
    return best_solution, current_iteration, error_at_iteration

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


# Ejercicio 1
#===================================================================================================


def ejercicio1():
    """Codigo que lanza todos los apartados del primer ejercicio"""

    # Fijamos la semilla para no depender tanto de la aleatoriedad y conseguir resultados
    # reproducibles
    np.random.seed(123456789)

    print("==> Lanzando ejercicio 1")
    print("=" * 80)

    # Primer apartado
    ejercicio1_apartado1()

    # Segundo apartado
    ejercicio1_apartado2()


def ejercicio1_apartado1():
    """Codigo que lanza la tarea del primer apartado del primer ejercicio"""
    print("--> Ejercicio 1 Apartado 1")

    # Parametros de la tarea pedida
    number_of_points = 50   # Numero de datos
    dimensions = 2          # Dimensiones de cada dato
    lower = -50             # Extremo inferior del intervalo en cada coordenada
    upper = 50              # Extremo superior del intervalo en cada coordenada
    lower_sigma = 2         # Extremo inferior para el valor de la desviacion
    upper_sigma = 7         # Extremo superior para el valor de la desviacion

    # Generamos los dos conjuntos de datos
    uniform_dataset = simula_unif(
        number_of_points,
        dimensions,
        rango=[lower, upper]
    )
    gauss_dataset = simula_gaus(
        number_of_points,
        dimensions,
        sigma=[lower_sigma, upper_sigma]
    )

    # Mostramos los dos datasets obtenidos
    # Hacemos dataset[:, 0], dataset[:, 1] para quedarnos con todos los datos de la primera y
    # segunda columna en arrays distintos. Esto es, separamos las coordenadas x1 y x2
    print("Dataset generado con una distribucion uniforme")
    scatter_plot(uniform_dataset[:, 0], uniform_dataset[:, 1],
                 f"Scatter Plot de la distribucion uniforme de {number_of_points} puntos en el rango [{lower}, {upper}]")

    print("Dataset generado con una distribucion gaussiana")
    scatter_plot(gauss_dataset[:, 0], gauss_dataset[:, 1],
                 f"Scatter Plot de la distribucion gaussiana de {number_of_points} puntos con sigma en [{lower_sigma, upper_sigma}]")


def generate_labels_with_random_straight_line(dataset, lower, upper, ignore_first_column = False):
    """
    Genera las etiquetas para una muestra de datos de dos dimensiones usando el signo de la distancia
    a la recta simulada (a partir del codigo de los profesores). Tambien devuelve la recta que
    ha sido usada para el etiquetado

    Parameters:
    ===========
    dataset: conjunto de datos que queremos etiquetar
    lower: extremo inferior del dominio de los datos
    upper: extremo superior del dominio de los datos

    Returns:
    ========
    labels: np.array en el dominio {-1, 1} con el etiquetado
    line_coeffs: los coeficientes de la recta que ha sido usada para generar el etiquetado
    ignore_first_column: si queremos ignorar una primera columna de unos añadida para representar el
                         termino independiente de un modelo lineal
    """

    # Recta simulada que nos servira para etiquetar los datos y funciones para etiquetar
    a, b = simula_recta(intervalo=[lower, upper])

    # Recta que generamos aleatoriamente
    def f(x): return a * x + b

    # Distancia de un punto a la recta aleatoria y funcion de etiquetado
    def distance(x, y): return y - f(x)

    # Usamos la funcion generica de etiquetado para generar el etiquetado
    # Devolvemos tambien los coeficientes tomados
    return generate_labels_with_function(dataset, distance, ignore_first_column), [a, b]


def generate_labels_with_function(dataset, labeling_function, ignore_first_column = False):
    """
    Genera las etiquetas para una muestra de datos de dos dimensiones usando el signo de una funcion
    de etiquetado dada

    Parameters:
    ===========
    dataset: conjunto de datos que queremos etiquetar, con dos coordenadas
    labeling_function: funcion de etiquetado usada, con parametros de entrada x e y
    ignore_first_column: si queremos ignorar una primera columna de unos añadida para representar el
                         termino independiente de un modelo lineal

    Returns:
    ========
    labels: np.array en el dominio {-1, 1} con el etiquetado
    """

    # Funcion de etiquetado que vamos a usar
    labeling = lambda x,y: np.sign(labeling_function(x, y))

    # Etiquetamos la muestra
    labels = []

    for data_point in dataset:

        # Descomponemos las coordenadas
        # Tenemos que controlar si ignoramos o no la primera columna de la matriz
        x, y = data_point[0], data_point[1]
        if ignore_first_column == True:
            x, y = data_point[1], data_point[2]

        # Tomamos el signo de la distancia a la recta
        label = labeling(x, y)

        # Añadimos la etiqueta
        labels.append(label)

    return np.array(labels)


def change_labels(labels, percentage):
    """
    Cambiamos un porcentaje dado de las etiquetas. Las etiquetas que se cambian se escogen en orden
    aleatorio. La cantidad de cambios no es aleatoria, pues la fija el percentage de forma
    deterministica. Se cambia un percentage de etiquetas positivas y otro percentage de etiquetas
    negativas

    Parameters:
    ===========
    labels: etiquetas que vamos a modificar. Debe ser un array de numpy
    percentage: porcentaje en tantos por uno de etiquetas que vamos a cambiar, tanto negativas como
                positivas

    Returns:
    ========
    new_labels: nuevo etiquetado como ya se ha indicado
    """

    # Comprobacion de seguridad
    if percentage < 0 or percentage > 1:
        raise Exception("El porcentaje debe estar en el intervalo [0, 1]")

    # Copiamos las etiquetas
    new_labels = labels.copy()

    # Contamos el numero de positivos y negativos
    # La idea de usar collections.Counter viene del post en StackOverflow: [1]
    counter = collections.Counter(labels)
    number_of_positives = counter.get(label_pos)
    number_of_negatives = counter.get(label_neg)

    # Comprobacion de seguridad. Cuando no hay etiquetas de algun tipo, en vez de devolver un 0
    # se devuelve un None
    if number_of_positives is None:
        number_of_positives = 0
    if number_of_negatives is None:
        number_of_negatives = 0

    # Tomamos el numero de elementos a cambiar usando el porcentage
    num_positives_to_change = int(number_of_positives * percentage)
    num_negatives_to_change = int(number_of_negatives * percentage)

    # Tomamos un vector de posiciones de etiquetas negativas y otro de etiquetas positivas
    positive_index = []
    negative_index = []

    # Recorremos elemetos, pero tambien indices, porque lo que nos interesa guardar son los indices
    # de los dos tipos de etiquetas
    for index, label in enumerate(labels):
        if label == label_pos:
            positive_index.append(index)
        elif label == label_neg:
            negative_index.append(index)
        else:
            raise Exception(
                f"Una etiqueta no tiene valor en {-1, 1}. El valor encontrado fue {label}")

    # Como el cambio debe ser aleatorio, hacemos shuffle de los indices
    np.random.shuffle(positive_index)
    np.random.shuffle(negative_index)

    # Modificamos las etiquetas negativas
    for i in range(num_negatives_to_change):
        # Indice en el vector de etiquetas que debemos cambiar
        # Lo recorremos secuencialmente porque anteriormente hemos hecho un shuffle
        index_to_change = negative_index[i]

        # Cambiamos ese indice por una etiqueta positiva
        new_labels[index_to_change] = label_pos

    # Modificamos las etiquetas positivas
    for i in range(num_positives_to_change):
        # Indice en el vector de etiquetas que debemos cambiar
        # Lo recorremos secuencialmente porque anteriormente hemos hecho un shuffle
        index_to_change = positive_index[i]

        # Cambiamos ese indice por una etiqueta positiva
        new_labels[index_to_change] = label_neg

    return new_labels


def ejercicio1_apartado2():

    """Codigo que lanza la tarea del segundo apartado del primer ejercicio"""
    print("--> Ejercicio 1 Apartado 2")

    print("Subapartado a)")
    # Generamos el dataset que se nos indica para este apartado
    number_of_points = 100  # Numero de datos
    dimensions = 2          # Dimensiones de cada dato
    lower = -50             # Extremo inferior del intervalo en cada coordenada
    upper = 50              # Extremo superior del intervalo en cada coordenada
    dataset = simula_unif(number_of_points, dimensions, rango=[lower, upper])

    # Generamos las etiquetas como se indica
    labels, line_coeffs = generate_labels_with_random_straight_line(
        dataset,
        lower,
        upper
    )

    # Guardamos los datos en una variable global para poder usarlos en el siguiente ejercicio
    # El keyword global se necesita para que el cambio se realice
    # De otro formo, se crea una variable local que hace sombra a la variable globaL
    global global_dataset
    global global_labels
    global global_lin_coeffs
    global_dataset = dataset
    global_labels = labels
    global_lin_coeffs = line_coeffs

    # Mostramos el etiquetado de los datos junto a la recta que se ha usado para etiquetar
    print("Mostramos los datos generados y el etiquetado realizado a partir de una linea aleatoria")

    # Tomamos la funcion de etiquetado, que ya esta despejada para el valor de y
    labeling_function = get_straight_line(line_coeffs[0], line_coeffs[1])

    # Mostramos la grafica de etiquetado deterministico
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        target_names=["Signo Negativo", "Signo positivo"],
        feature_names=["Eje X", "Eje Y"],
        title="Datos etiquetados por la linea aleatoria",
        labeling_function=labeling_function
    )

    print("Subapartado b)")
    print("Cambiamos aleatoriamente el 10% de las etiquetas")
    # Modificamos el 10% de las etiquetas positivas y el 10% de las etiquetas negativas, escogiendo
    # los elementos a cambiar de forma aleatoria (sin seguir ningun orden sobre las etiquetas a modificar)
    changed_labels = change_labels(labels, 0.1)

    # Guardamos las variables modificadas para el siguiente ejercicio
    global global_noisy_labels
    global_noisy_labels = changed_labels

    # Mostramos de nuevo la grafica, con etiquetas sonoras y la funcion que fue usada para el
    # etiquetado deterministico
    print("Mostramos el etiquetado cambiado y la recta de clasificacion original")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        changed_labels,
        target_names=["Signo Negativo", "Signo positivo"],
        feature_names=["Eje X", "Eje Y"],
        title=f"Datos etiquetados con ruido y recta de clasificacion original",
        labeling_function=labeling_function
    )



    print("Subapartado c)")
    # Generamos las funciones de etiquetado
    f1 = lambda x,y: (x - 10.0)*(x-10.0) + (y-20)*(y-20) - 400
    f2 = lambda x,y: 0.5 * (x + 10.0)*(x+10.0) + (y-20)*(y-20) - 400
    f3 = lambda x,y: 0.5 * (x - 10.0)*(x-10.0) - (y+20)*(y+20) - 400
    f4 = lambda x,y: y - 20.0 * x*x - 5.0*x + 3.0
    labeling_functions = [f1, f2, f3, f4]

    # Usando estas funciones de etiquetado, mostramos las regiones de clasificacion y los puntos,
    # que LLEVAN EL ETIQUETADO ORIGINAL a partir de la recta aleatoria y la introduccion de ruido
    # sintetico
    for index, labeling_function in enumerate(labeling_functions):

        # Mostramos el etiquetado original del dataset junto con las regiones de clasificacion
        # de las nuevas funciones dadas. En el titulo de la grafica mostramos el error porcentual
        # cometido
        print(f"Mostrando el etiquetado con ruido de la funcion {index+1}-esima")
        percentage_error = 100 * percentage_error_function(dataset, changed_labels, lambda point: np.sign(labeling_function(point[0], point[1])))
        scatter_plot_with_classes_and_labeling_region(
            data=dataset,
            classes=changed_labels,
            target_names=["Signo Negativo", "Sigo positivo"],
            feature_names=["Eje X", "Eje Y"],
            title=f"Puntos etiquetados con ruido por la funcion f{index} -- {percentage_error}% mal clasificado",
            labeling_function=labeling_function
        )

    print("Subapartado c) -- Experimento alternativo")
    # Generamos las funciones de etiquetado
    f1 = lambda x,y: (x - 10.0)*(x-10.0) + (y-20)*(y-20) - 400
    f2 = lambda x,y: 0.5 * (x + 10.0)*(x+10.0) + (y-20)*(y-20) - 400
    f3 = lambda x,y: 0.5 * (x - 10.0)*(x-10.0) - (y+20)*(y+20) - 400
    f4 = lambda x,y: y - 20.0 * x*x - 5.0*x + 3.0
    labeling_functions = [f1, f2, f3, f4]

    # Realizamos el mismo experimento que en apartado anterior pero usando las nuevas funciones de
    # etiquetado
    for index, labeling_function in enumerate(labeling_functions):
        # Generamos el etiquetado con la funcion que pasamos
        deterministic_labels = generate_labels_with_function(
            dataset,
            labeling_function
        )

        # Modificamos aleatoriamente algunas etiquetas
        noisy_labels = change_labels(deterministic_labels, 0.1)

        # Mostramos la grafica de etiquetado de las funciones junto a la funcion clasificadora
        # Ademas, mostramos el porcentaje de puntos mal clasificados
        print(f"Mostrando el etiquetado con ruido de la funcion {index+1}-esima")
        percentage_error = 100 * percentage_error_function(dataset, noisy_labels, lambda point: np.sign(labeling_function(point[0], point[1])))
        scatter_plot_with_classes_and_labeling_region(
            data=dataset,
            classes=noisy_labels,
            target_names=["Signo Negativo", "Sigo positivo"],
            feature_names=["Eje X", "Eje Y"],
            title=f"Puntos etiquetados con ruido por la funcion f{index} -- {percentage_error}% mal clasificado",
            labeling_function=labeling_function
        )

# Ejercicio 2
#===================================================================================================
def ejercicio2():
    """Codigo que lanza todas las tareas del segundo ejercicio"""

    # Fijamos la semilla para no depender tanto de la aleatoriedad y conseguir resultados
    # reproducibles
    np.random.seed(123456789)

    print("==> Lanzando ejercicio 2")

    # Lanzamos el primer apartado
    ejercicio2_apartado1()

    # Lanzamos el segundo apartado
    ejercicio2_apartado2()

def PLA_experiment(dataset, labels, max_iterations = 1e5, repetitions = 10, init_type: str = "zero"):
    """
    Lanza el experimento consistente en lanzar un numero de veces dado el algoritmo PLA sobre un
    dataset. El usuario especifica el tipo de vector inicial a usar

    Parameters:
    ===========
    dataset: conjunto de datos a etiquetar
    labels: etiquetas del conjunto de datos
    max_iterations: numero maximo de iteraciones
    repetitions: numero de veces que repetimos el experimento
    init_type: tipo de vector inicial que queremos usar. Los valores pueden ser:
                    - "zero"
                    - "random"

    Returns:
    ========
    final_errors: vector con los errores alcanzados en cada repeticion del algoritmo
    consumed_iterations: iteraciones consumidas por cada repeticion del algoritmo

    Es muy facil a partir de estor vectores calcular los valores medios que se nos piden, como
    hago en el codigo del segundo ejercicio
    """
    # Experimento para vector inicial aleatorio
    # Valores que guardamos para promediar
    final_errors = []
    consumed_iterations = []

    for _ in range(10):

        # Seleccionamos el tipo de vector inicial
        init_solution = None
        if init_type == "zero":
            init_solution = np.zeros(len(dataset[0]))
        elif init_type == "random":
            init_solution = np.random.rand(len(dataset[0]))
        else:
            raise Exception("Tipo de solucion inicial no valida")


        # Calculamos todos los valores para la inicializacion aleatoria
        curr_sol, curr_cons_it = perceptron_learning_algorithm(dataset, labels, max_iterations, init_solution, verbose = False)
        curr_err = percentage_error(dataset, labels, curr_sol)

        # Guardamos el valor
        final_errors.append(curr_err)
        consumed_iterations.append(curr_cons_it)

    return final_errors, consumed_iterations

def ejercicio2_apartado1():
    """Codigo que lanza las tareas del apartado primero del segundo ejercicio"""
    print("--> Lanzando apartado 1")
    print("--> Subapartado 1")

    # Tomamos los datos generados en el primer ejercicio, apartado 2 subapartado a
    dataset = global_dataset
    labels = global_labels
    lin_coeffs = global_lin_coeffs

    # Añadimos la columna de unos al dataset, para representar el termino independiente de la
    # combinacion lineal
    dataset = add_colum_of_ones(dataset)

    # Mostramos el dataset con el que trabajamos y el etiquetado generado
    print("Dataset y etiquetado con el que trabajamos en este apartado:")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        ["Valor positivo", "Valor negativo"],
        ["Eje X", "Eje Y"],
        "Clasificacion de los datos usando una recta",
        get_straight_line(lin_coeffs[0], lin_coeffs[1]),
        ignore_first_column = True
    )

    # Parametros del algoritmo
    max_iterations = 5000   # No se especifica, pero lo coloco como medida de seguridad
                            # Cuando los datos son linealmente separables, es extremadamente
                            # improbable alcanzar este numero de iteraciones sobre EPOCHS

    # Lanzamos las diez repeticiones con vector inicial cero y mostramos los resultados
    print("Lanzamos 10 repeticiones del experimento, para vector inicial cero")
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="zero"
    )

    # Mostramos los resultados
    print("Resultado de las 10 iteraciones -- Vector Inicial Cero:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Valor medio de los errores: {sum(final_errors) / len(final_errors) * 100}%")
    print(f"\t-> Desviacion tipica de los errores: {np.std(final_errors)}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    print(f"\t-> Desviacion tipica del numero de iteraciones: {np.std(consumed_iterations)}")
    wait_for_user_input()

    # Experimento para vector inicial aleatorio
    print("Lanzamos 10 repeticiones del experimento, para vector inicial aleatorio")
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="random"
    )

    # Mostramos los resultados
    print("Resultado de las 10 iteraciones -- Vector Inicial aleatorio:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Valor medio de los errores: {sum(final_errors) / len(final_errors) * 100}%")
    print(f"\t-> Desviacion tipica de los errores: {np.std(final_errors)}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    print(f"\t-> Desviacion tipica del numero de iteraciones: {np.std(consumed_iterations)}")
    wait_for_user_input()

    # Ahora vamos a lanzar dos ejecuciones individuales, con vector inicial cero y vector inicial
    # aleatorio, y mostramos las graficas para poder añadirlas a la memoria y tener una pequeña
    # intuicion sobre el comportamiento del algoritmo sobre este dataset

    # Lanzamos el algoritmo con vector inicial cero a solas para mostrar las graficas
    print("-> Mostrando una unica ejecucion para la solucion inicial zero")
    zero_solution = np.zeros_like(dataset[0])
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorithm(dataset, labels, max_iterations, zero_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Evolución del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

    # Mostramos como clasifica nuestra solucion
    print("Mostrando el clasificador obtenido")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        ["Valor positivo", "Valor negativo"],
        ["Eje X", "Eje Y"],
        "Clasificacion de los datos usando una recta",
        get_frontier_function(perceptron_weights),
        ignore_first_column = True
    )

    # Hacemos lo mismo pero para una solucion inicial aleatoria
    print("-> Mostrando una unica ejecucion para la solucion inicial aletoria")
    rand_solution = np.random.rand(len(dataset[0]))
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorithm(dataset, labels, max_iterations, rand_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Evolución del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

    # Mostramos como clasifica nuestra solucion
    print("Mostrando el clasificador obtenido")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        ["Valor positivo", "Valor negativo"],
        ["Eje X", "Eje Y"],
        "Clasificacion de los datos usando una recta",
        get_frontier_function(perceptron_weights),
        ignore_first_column = True
    )

    print("--> Subapartado 2")
    print("Repetimos el experimento usando etiquetas con ruido y vectores iniciales zero y aleatorio del ejercicio 1.2.b)")
    # Repetimos el experimento con los datos del ejercicio 1 Apartado 2 Subapartado b
    # Tomamos los datos guardados en una variable global
    noisy_labels = global_noisy_labels

    # Lanzamos las diez repeticiones con vector inicial cero y mostramos los resultados
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        noisy_labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="zero"
    )
    print("Resultado de las 10 iteraciones -- Vector Inicial Cero:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Valor medio de los errores: {sum(final_errors) / len(final_errors) * 100}%")
    print(f"\t-> Desviacion tipica de los errores: {np.std(final_errors)}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    print(f"\t-> Desviacion tipica del numero de iteraciones: {np.std(consumed_iterations)}")
    wait_for_user_input()

    # Lanzamos las diez repeticiones con vector inicial aleatorio y mostramos los resultados
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        noisy_labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="random"
    )
    print("Resultado de las 10 iteraciones -- Vector inicial aleatorio:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Valor medio de los errores: {sum(final_errors) / len(final_errors) * 100}%")
    print(f"\t-> Desviacion tipica de los errores: {np.std(final_errors)}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    print(f"\t-> Desviacion tipica del numero de iteraciones: {np.std(consumed_iterations)}")
    wait_for_user_input()

    # Esta vez lanzamos una unica ejecucion de PLA para mostrar las graficas
    # Elegimos vector inicial PLA de forma arbitraria para mostrar la grafica
    print("-> Mostrando una unica ejecucion para la solucion inicial aletoria")
    rand_solution = np.random.rand(len(dataset[0]))
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorithm(dataset, noisy_labels, max_iterations, rand_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Evolución del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

    # Mostramos como clasifica nuestra solucion
    print("Mostrando el clasificador obtenido")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        noisy_labels,
        ["Valor positivo", "Valor negativo"],
        ["Eje X", "Eje Y"],
        "Clasificacion de los datos usando una recta",
        get_frontier_function(perceptron_weights),
        ignore_first_column = True
    )

def generate_dataset(number_of_points):
    """
    Simula el efecto de extraer number_of_points elementos de la poblacion. La poblacion es el
    intervalo [0,2] x [0,2] en el que los puntos se distribuyen con probabilidad uniforme. Por ello,
    para realizar la simulacion basta con generar dos numeros aleatorios en el intervalo [0, 2] y
    devolverlos como coordenadas
    """

    # Datos distribuidos uniformemente en [0, 1] x [0, 1]
    dataset = np.random.rand(number_of_points, 2)

    # Devolvemos los datos distribuidos uniformemente en [0, 2] x [0, 2]
    return dataset*2

def calculate_straight_line_from_two_points(first, second):
    """Dados dos puntos bidimensionales, devuelve la recta que pasa por esos dos puntos"""
    slope = (second[1] - first[1]) / (second[0] - first[0])
    offset = second[1] - slope * second[0]
    return lambda x: slope * x + offset


def ejercicio2_apartado2():
    """Codigo que lanza las tareas del segundo apartado del segundo ejercicio"""
    print("--> Apartado 2")

    # Generamos el dataset como se indica en el enunciado
    number_of_points = 100
    dataset = generate_dataset(number_of_points)

    # Generamos la recta de etiquetado a partir de dos puntos aleatorios de la poblacion
    two_pop_points = generate_dataset(2)
    random_line = calculate_straight_line_from_two_points(two_pop_points[0], two_pop_points[1])
    deterministic_labeling_function = lambda x, y: y - random_line(x) # Etiquetamos con la distancia a la recta

    # Añadimos la columna de unos a la matriz de datos para representar el termino independiente
    # en el sumando de la combinacion lineal
    dataset = add_colum_of_ones(dataset)

    # Generamos el etiquetado de los datos, y mostramos como queda la grafica de los puntos etiquetados
    # junto a la recta que los ha generado
    labels = generate_labels_with_function(dataset, deterministic_labeling_function, ignore_first_column=True)
    print("Mostrando etiquetado de la muestra de datos generada, a partir de una recta que pasa por dos puntos de la poblacion")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        ["Datos positivos", "Datos negativos"],
        ["Eje X", "Eje Y"],
        "Etiquetado de los datos con una recta aleatoria",
        random_line,
        ignore_first_column = True
    )

    # Lanzamos el algorimto de gradiente descendente para regresion logistica
    # Parametros del algoritmo
    init_solution = np.zeros_like(dataset[0])
    learning_rate = 0.01
    target_epoch_delta = 0.01
    batch_size = 1

    print("Lanzando Stochastic Gradient Descent para regresion logistica...")
    solution, _, _, error_at_epoch, error_at_minibatch_iteration = stochastic_gradient_descent(
        data = dataset,
        labels = labels,
        starting_solution = init_solution,
        learning_rate = learning_rate, batch_size = batch_size,
        max_minibatch_iterations = None,
        target_error = None,
        target_epoch_delta = target_epoch_delta,
        gradient_function = logistic_gradient,
        error_function = logistic_error,
        verbose = True
    )

    # Calculamos el porcentaje de puntos mal clasificados para mostrarlo con los resultados
    percentage_error = percentage_logistic_error(dataset, labels, solution) * 100

    # Mostramos los resultados
    print("-->Resultados del gradiente descendente")
    print(f"\t- Pesos obtenidos: {solution}")
    print(f"\t- Iteraciones consumidas: {len(error_at_minibatch_iteration * batch_size)}")
    print(f"\t- Error final alcanzado (error LGR): {error_at_minibatch_iteration[-1]}")
    print(f"\t- Error final alcanzado (porcentaje mal clasificado): {percentage_error}%")
    wait_for_user_input()

    # Mostramos el grafico de evolucion del error
    print("Mostrando el grafico de evolucion del error")
    plot_error_evolution(error_at_minibatch_iteration, title="Evolucion del error", x_label="Minibatch Iteration")

    # Mostramos la frontera de clasificación
    print("Mostrando la frontera de clasificacion obtenida")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        labels,
        ["Datos positivos", "Datos negativos"],
        ["Eje X", "Eje Y"],
        "Etiquetado de los datos con una recta aleatoria",
        get_straight_line_from_implicit(solution),
        ignore_first_column = True
    )


    # Mostramos la grafica de puntos mal clasificados
    print("Mostramos la grafica de puntos mal clasificados")
    plot_misclassified_classification_predictions(dataset, labels, get_logistic_classifier(solution), ["Eje X", "Eje Y"], ignore_first_column = True)

    # Calculamos el error fuera de la muestra generando otra muestra aleatoria de un alto numero de datos
    # etiquetandolos con la recta dada (lo que seria el etiquetado deterministico verdadero) y
    # calculando como fallamos en la muestra de test
    print("--> Generamos una muestra de datos de test para evaluar el error fuera de la muestra")
    size_of_test_sample = int(1e4)
    test_dataset = generate_dataset(size_of_test_sample)
    test_labels = generate_labels_with_function(test_dataset, deterministic_labeling_function)

    # Añadimos la columna de unos a la matriz de datos para representar el termino independiente
    # en el sumando de la combinacion lineal
    test_dataset = add_colum_of_ones(test_dataset)

    # Calculamos el error porcentual
    percentage_error = percentage_logistic_error(test_dataset, test_labels, solution) * 100
    print(f"\t- Porcentaje de puntos mal clasificados en el test: {percentage_error}%")
    wait_for_user_input()

    # Mostramos el grafico de puntos mal clasificados
    print("Mostrando puntos mal clasificados en el test sample")
    plot_misclassified_classification_predictions(
        test_dataset,
        test_labels,
        get_logistic_classifier(solution),
        ["Eje X", "Eje Y"],
        title = "Puntos mal clasificados en la muestra de test",
        ignore_first_column = True
    )

    # Una vez hecho esto, que incluiremos en la memoria, repetimos el experimento 100 veces
    print("--> Lanzamos 100 veces el experimento anterior")
    minibatch_iterations, epoch_iterations, percentage_error_at_test_sample = logistic_regresion_experiment(100)

    # Tomamos algunas estadisticas de los experimentos
    mean_minibatch = float(sum(minibatch_iterations)) / float(len(minibatch_iterations))
    mean_epoch = float(sum(epoch_iterations)) / float(len(minibatch_iterations))
    mean_err = sum(percentage_error_at_test_sample) / len(percentage_error_at_test_sample)
    dev_err = np.std(percentage_error_at_test_sample)

    # Mostramos las estadisticas
    print("--> Resultados del experimento")
    print(f"\t--> Media de iteraciones sobre minibatches consumidas: {mean_minibatch}")
    print(f"\t--> Media de iteraciones sobre epochs consumidas: {mean_epoch}")
    print(f"\t--> Media del porcentaje de puntos mal clasificados fuera de la muestra: {mean_err*100}%")
    print(f"\t--> Desviacion tipica del error fuera de la muestra: {dev_err}")
    wait_for_user_input()


def logistic_regresion_experiment(number_of_repetitions):
    """
    Lanzamos un numero dado de veces el experimento sobre regresion logistica. Al lanzar muchas veces
    el experimento, no mostramos mensajes por pantalla ni mostramos graficas

    Parameters:
    ===========
    number_of_repetitions: numero de veces que repetimos el experimento

    Returns:
    ========
    minibatch_iterations: iteraciones sobre minibatches necesarias en cada repeticion del experimento
                          para que SGD converja
    epoch_iterations: iteraciones sobre EPOCHS necesarias en cada repeticion para que SGD converja
    percentage_error_at_test_samle: porcentaje de puntos mal clasificados en las muestras de test
                                    de cada repeticion. Notar que en cada repeticion se genera una
                                    nueva muestra de testing
    """

    # Parametros del experimento
    learning_rate = 0.01
    target_epoch_delta = 0.01
    batch_size = 1

    # Valores que guardamos en el experimento
    # Iteraciones de minibatch necesarias para que converja SGD
    minibatch_iterations = []

    # Iteraciones de epoc necesarias para que converja SGD
    epoch_iterations = []

    # Error en el test_sample
    # No guardamos el error dentro de la muestra porque es menos interesante a la hora de ser
    # analizado. Seria interesante guardarlo en el caso en el que detectasemos problemas para que
    # el algoritmo convergiese en la muestra de entrenamiento
    percentage_error_at_test_samle = []

    for it in range(number_of_repetitions):
        # Generamos el dataset como se indica en el enunciado
        number_of_points = 100
        dataset = generate_dataset(number_of_points)

        # Generamos la recta de etiquetado a partir de dos puntos aleatorios de la poblacion
        two_pop_points = generate_dataset(2)
        random_line = calculate_straight_line_from_two_points(two_pop_points[0], two_pop_points[1])
        deterministic_labeling_function = lambda x, y: y - random_line(x) # Etiquetamos con la distancia a la recta

        # Añadimos la columna de unos a la matriz de datos para representar el termino independiente
        # en el sumando de la combinacion lineal
        dataset = add_colum_of_ones(dataset)

        # Generamos el etiquetado de los datos
        labels = generate_labels_with_function(dataset, deterministic_labeling_function, ignore_first_column=True)

        # Lanzamos el algorimto de gradiente descendente para regresion logistica
        init_solution = np.zeros_like(dataset[0])
        solution, minibatch_iterations_consumed, epoch_iterations_consumed, error_at_epoch, error_at_minibatch_iteration = stochastic_gradient_descent(
            data = dataset,
            labels = labels,
            starting_solution = init_solution,
            learning_rate = learning_rate, batch_size = batch_size,
            max_minibatch_iterations = None,
            target_error = None,
            target_epoch_delta = target_epoch_delta,
            gradient_function = logistic_gradient,
            error_function = logistic_error,
            verbose = False
        )

        # Añadimos los valores calculados
        minibatch_iterations.append(minibatch_iterations_consumed)
        epoch_iterations.append(epoch_iterations_consumed)

        # Calculamos el error fuera de la muestra generando otra muestra aleatoria de un alto numero de datos
        # etiquetandolos con la recta dada (lo que seria el etiquetado deterministico verdadero) y
        # calculando como fallamos en la muestra de test
        size_of_test_sample = int(1e4)
        test_dataset = generate_dataset(size_of_test_sample)
        test_labels = generate_labels_with_function(test_dataset, deterministic_labeling_function)

        # Añadimos la columna de unos a la matriz de datos para representar el termino independiente
        # en el sumando de la combinacion lineal
        test_dataset = add_colum_of_ones(test_dataset)

        # Calculamos el error porcentual y lo añadimos a los datos que devolvemos
        percentage_error = percentage_logistic_error(test_dataset, test_labels, solution)
        percentage_error_at_test_samle.append(percentage_error)

        # Mostramos el progreso del experimento
        if it % 10 == 0:
            print(f"\t--> Iteracion {it} finalizada")


    # Devolvemos los resultados del experimento
    return minibatch_iterations, epoch_iterations, percentage_error_at_test_samle

# Ejercicio Bonus
#===================================================================================================
def ejercicio_bonus():
    """Lanzamos todo el codigo para resolver la tarea extra"""

    # Fijamos la semilla para no depender tanto de la aleatoriedad y conseguir resultados
    # reproducibles
    np.random.seed(123456789)

    print("==> Lanzando ejercicio extra")

    print("--> Cargando los datos de entrenamiento y test")

    # Parametros para extraer los datos a partir de la funcion dada por los profesores
    learning_path_caracteristics = "./datos/X_train.npy"
    learning_path_labels = "./datos/y_train.npy"
    test_path_caracteristics = "./datos/X_test.npy"
    test_path_labels = "./datos/y_test.npy"
    digits = [4, 8]
    labels = [-1, 1]

    # Vamos a usar estos parametros para mostrar las graficas repetidamente
    target_names = ["Digito 4", "Digito 8"]
    feature_names = ["Intensidad promedio", "Simetria"]

    # Cargamos los datos de training
    learning_dataset, learning_labels = readData(learning_path_caracteristics, learning_path_labels, digits, labels)

    # Cargamos los datos de test
    test_dataset, test_labels = readData(test_path_caracteristics, test_path_labels, digits, labels)

    # Mostramos las graficas de los datos cargados
    print("--> Mostrando los datos cargados")
    print("Datos de entrenamiento")
    scatter_plot_with_classes(
        learning_dataset,
        learning_labels,
        target_names,
        feature_names,
        title="Datos de entrenamiento",
        ignore_first_column=True
    )

    print("Datos de testing")
    scatter_plot_with_classes(
        test_dataset,
        test_labels,
        target_names,
        feature_names,
        title="Datos de test",
        ignore_first_column=True
    )

    # Lanzamos el algoritmo de la pseudo inversa como punto inicial para PLA-POCKET
    print("--> Lanzando el algoritmo de la pseudoinversa")
    first_solution = pseudo_inverse(learning_dataset, learning_labels)


    # Mostramos los resultados de esta primera etapa
    print(f"\t- Solucion inicial: {first_solution}")
    print(f"\t- Error conseguido en la muestra: {percentage_error(learning_dataset, learning_labels, first_solution)}")
    print(f"\t- Error conseguido fuera de la muestra: {percentage_error(test_dataset, test_labels, first_solution)}")
    wait_for_user_input()

    # Lanzamos PLA-Pocket para mejorar la solucion obtenida
    print("--> Lanzando algoritmo PLA-Pocket")
    max_iterations = 50
    solution, iterations_consumed, error_at_iteration = perceptron_learning_algorithm_pocket(learning_dataset, learning_labels, max_iterations, first_solution)
    print(f"\t- Solucion: {solution}")
    print(f"\t- Iteraciones consumidas: {iterations_consumed}")
    print(f"\t- Error conseguido en la muestra: {error_at_iteration[-1]}")
    print(f"\t- Error conseguido fuera de la muestra: {percentage_error(test_dataset, test_labels, solution)}")
    wait_for_user_input()

    # Mostramos la grafica de evolucion de Ein
    print("--> Mostrando evolucion del error en la muestra durante el aprendizaje")
    plot_error_evolution(
        error_at_iteration,
        title="Evolucion del error en la muestra durante el aprendizaje",
        y_label="Error en la muestra"
    )

    # Para mostrar las siguientes graficas, necesitamos la funcion de etiquetado que representa
    # los pesos de la solucion obtenida
    labeling_function = get_straight_line_from_implicit(solution)

    # Mostramos la clasificacion con la linea tanto en la muestra de entrenamiento como en la
    # muestra de test
    print("--> Mostrando el etiquetado en los datos de entrenamiento")
    scatter_plot_with_classes_and_labeling_function(
        learning_dataset,
        learning_labels,
        target_names,
        feature_names,
        title="Clasificado de los datos de aprendizaje",
        labeling_function=labeling_function,
        ignore_first_column=True
    )

    print("--> Mostrando el etiquetado en los datos de test")
    scatter_plot_with_classes_and_labeling_function(
        test_dataset,
        test_labels,
        target_names,
        feature_names,
        title="Clasificado de los datos de aprendizaje",
        labeling_function=labeling_function,
        ignore_first_column=True
    )

# Funcion principal
#===================================================================================================
if __name__ == "__main__":

    # Lanzamos el primer ejercicio
    ejercicio1()

    # Lanzamos el segundo ejercicio
    ejercicio2()

    # Lanzamos el ejercicio extra
    ejercicio_bonus()
