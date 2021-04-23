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

def get_frontier_function(weights):
    """
    Dado un clasificador lineal de la forma w0 + w1x + w2y, devuelve la recta y = f(x) frontera del
    clasificador lineal

    Parameters:
    ===========
    weights: pesos del clasificador. np.array de tres elementos
    """
    return lambda x: (1.0 / weights[2]) * (-weights[0] - weights[1] * x)


# Valores de las etiquetas
#===================================================================================================
label_pos = 1
label_neg = -1

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
    Hacemos un scatter plot de puntos con dos coordeandas que estan etiquetados en distintos grupos

    Parameters:
    ===========
    data: coordeandas de los distintos puntos
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
    Hacemos un scatter plot de puntos con dos coordeandas que estan etiquetados a partir de una
    funcion de etiquetado que mostramos en la grafica. Notar que las etiquetas pueden tener ruido,
    por lo que la grafica puede mostrar puntos mal etiquetados segun la funcion dada
    Ademas, mostramos la recta que ha sido usada para etiquetar.

    Se usa esta funcion cuando podemos expresar y = f(x) de forma global, pues recordar que las
    funciones de etiquetado son funciones de dos variables implicitas

    Parameters:
    ===========
    data: coordeandas de los distintos puntos
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
    Hacemos un scatter plot de puntos con dos coordeandas que estan etiquetados a partir de una
    funcion de etiquetado que mostramos en la grafica. Notar que las etiquetas pueden tener ruido,
    por lo que la grafica puede mostrar puntos mal etiquetados segun la funcion dada

    Mostramos las regiones positivas y negativas de la funcion. Lo hacemos asi porque la idea de
    esta funcion es ser usada cuando labeling_function no tenga un despeje de una variable en funcion
    de la otra variable global (las funciones de etiquetado son funciones implicitas)

    Podriamos trabajar con los despejes locales del teorema de la funcion implicita, pero al ser
    funciones locales da mucho trabajo a la hora de generar las graficas

    Parameters:
    ===========
    data: coordeandas de los distintos puntos
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

# Algoritmos
# ===================================================================================================
def perceptron_learning_algorihtm(dataset, labels, max_iterations, init_solution, verbose = False):
    """
    Algoritmo de aprendizaje para perceptron

    Parameters:
    ===========
    dataset: conjunto de puntos que vamos a clasificar. Cada punto tiene sus coordenadas en una fila
             de la matriz
    labels: vector de etiquetas
    max_iterations: numero maximo de iteraciones
    init_solution: vector de pesos iniciales que representan la solucion inicial
    verbose: indica si queremos devolver mas datos de los estrictamente necesarios
             Al tener que evaluar el error, hace que el algoritmo corra mas lento

    Returns:
    ========
    current_solution: la solucion que se alcanza al final del proceso
    current_iteration: numero de iteraciones necesarias para alcanzar la solucion
    error_at_iteration: error en cada iteracion. Solo cuando verbose == True

    TODO -- añadir vector con la evolucion del error
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

                # Hemos cambiado el valor de los pesos, asi que tenemos que calcular el error de nuevo
                if verbose == True:
                    curr_err = percentage_error(dataset, labels, current_solution)
                    error_at_iteration.append(curr_err)
            else:
                # No cambia el valor de los pesos, asi que el error tampoco cambia
                # Esto nos ahorra bastante tiempo de computo
                if verbose == True:
                    if len(error_at_iteration) > 0:
                        last_error = error_at_iteration[-1]
                    # No tenemos un ultimo error (primer error que calculamos)
                    # Asi que tenemos que calcular a mano el error
                    else:
                        last_error = percentage_error(dataset, labels, current_solution)

                    # Añadimos el error que en la mayoria de los casos (salvo en el primer calculo
                    # de error), es un valor 'cacheado'
                    error_at_iteration.append(last_error)



            # Aumentamos la iteracion
            current_iteration = current_iteration + 1

            # Comprobamos si hemos agotado el maximo de iteraciones en este bucle interno
            if current_iteration >= max_iterations:
                break

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


# Ejercicio 1
#===================================================================================================


def ejercicio1():
    """Codigo que lanza todos los apartados del primer ejercicio"""
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

    Returns:
    ========
    labels: np.array en el dominio {-1, 1} con el etiquetado
    ignore_first_column: si queremos ignorar una primera columna de unos añadida para representar el
                         termino independiente de un modelo lineal
    """

    # Funcion de etiquetado que vamos a usar
    def labeling(x, y): return np.sign(labeling_function(x, y))

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

    # Mostramos de nuevo la grafica, con etiquetas sonoras y la funcion que fue usada para el
    # etiquetado deterministico
    print("Mostramos el etiquetado cambiado y la recta de clasificacion original")
    scatter_plot_with_classes_and_labeling_function(
        dataset,
        changed_labels,
        target_names=["Signo Negativo", "Signo positivo"],
        feature_names=["Eje X", "Eje Y"],
        title="Datos etiquetados con ruido y recta de clasificacion original",
        labeling_function=labeling_function
    )

    print("Subapartado c)")
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
        print(f"Mostrando el etiquetado con ruido de la funcion {index+1}-esima")
        scatter_plot_with_classes_and_labeling_region(
            data=dataset,
            classes=noisy_labels,
            target_names=["Signo Negativo", "Sigo positivo"],
            feature_names=["Eje X", "Eje Y"],
            title=f"Puntos etiquetados con ruido por la funcion f{index}",
            labeling_function=labeling_function
        )

# Ejercicio 2
# ===================================================================================================
def ejercicio2():
    """Codigo que lanza todas las tareas del segundo ejercicio"""

    print("==> Lanzando ejercicio 2")

    # Lanzamos el primer apartado
    ejercicio2_apartado1()

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
        # Tomamos una solucion aleatoria
        init_solution = np.random.rand(len(dataset[0]))

        # Calculamos todos los valores para la inicializacion aleatoria
        curr_sol, curr_cons_it = perceptron_learning_algorihtm(dataset, labels, max_iterations, init_solution, verbose = False)
        curr_err = percentage_error(dataset, labels, curr_sol)

        # Guardamos el valor
        final_errors.append(curr_err)
        consumed_iterations.append(curr_cons_it)

    return final_errors, consumed_iterations

def ejercicio2_apartado1():
    """Codigo que lanza las tareas del apartado primero del segundo ejercicio"""
    print("--> Lanzando apartado 1")
    print("--> Subapartado 1")

    # TODO -- tomar los mismos datos que en el ejercicio 1 Apartado 2a
    # Parametros de la tarea pedida
    number_of_points = 100      # Numero de datos
    dimensions = 2              # Dimensiones de cada dato
    lower = -50                 # Extremo inferior del intervalo en cada coordenada
    upper = 50                  # Extremo superior del intervalo en cada coordenada
    max_iterations = 1e5        # Numero maximo de iteraciones (no se especifica en el guion)

    # Generamos los dos conjuntos de datos
    dataset = simula_unif(
        number_of_points,
        dimensions,
        rango=[lower, upper]
    )

    # Añadimos una columna de unos para que esto represente el termino independiente en la combinacion
    # lineal. El codigo lo tomo de la practica anterior en la que teniamos que añadir columnas
    # a la matriz de entrada
    number_of_rows = int(np.shape(dataset)[0])
    new_column = np.ones(number_of_rows)
    dataset = np.insert(dataset, 0, new_column, axis = 1)

    # Generamos las etiquetas para estos datos
    labels, lin_coeffs = generate_labels_with_random_straight_line(dataset, lower, upper, ignore_first_column = True)

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
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
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
    print("Resultado de las 10 iteraciones -- Vector inicial aleatorio:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    wait_for_user_input()

    # Ahora vamos a lanzar dos ejecuciones individuales, con vector inicial cero y vector inicial
    # aleatorio, y mostramos las graficas para poder añadirlas a la memoria y tener una pequeña
    # intuicion sobre el comportamiento del algoritmo sobre este dataset

    # Lanzamos el algoritmo con vector inicial cero a solas para mostrar las graficas
    print("-> Mostrando una unica ejecucion para la solucion inicial zero")
    zero_solution = np.zeros_like(dataset[0])
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorihtm(dataset, labels, max_iterations, zero_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Iteracion del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

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
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorihtm(dataset, labels, max_iterations, rand_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Iteracion del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

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
    print("Repetimos el experimento usando etiquetas con ruido y vectores iniciales zero y aleatorio")
    # Repetimos el experimento con los datos del ejercicio 1 Apartado 2 Subapartado b
    # Para ello, modifcamos aleatoriamente el 10% de las etiquetas positivas y el 10% de las
    # etiquetas negativas
    noisy_labels = change_labels(labels, 0.1)

    # Lanzamos las diez repeticiones con vector inicial cero y mostramos los resultados
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        noisy_labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="zero"
    )
    print("Resultado de las 10 iteraciones -- Vector inicial cero:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    wait_for_user_input()

    # Lanzamos las diez repeticiones con vector inicial aleatorio y mostramos los resultados
    final_errors, consumed_iterations = PLA_experiment(
        dataset,
        noisy_labels,
        max_iterations = max_iterations,
        repetitions=10,
        init_type="randon"
    )
    print("Resultado de las 10 iteraciones -- Vector inicial aleatorio:")
    print(f"\t-> Errores finales(tanto por uno): {final_errors}")
    print(f"\t-> Iteraciones consumidas: {consumed_iterations}")
    print(f"\t-> Valor medio de iteraciones: {sum(consumed_iterations) / len(consumed_iterations)}")
    wait_for_user_input()

    # Esta vez lanzamos una unica ejecucion de PLA para mostrar las graficas
    # Elegimos vector inicial PLA de forma arbitraria para mostrar la grafica
    print("-> Mostrando una unica ejecucion para la solucion inicial aletoria")
    rand_solution = np.random.rand(len(dataset[0]))
    perceptron_weights, consumed_iterations, error_at_iteration = perceptron_learning_algorihtm(dataset, noisy_labels, max_iterations, rand_solution, verbose = True)
    print(f"Pesos del perceptron obtenidos: {perceptron_weights}")
    print(f"Iteraciones consumidas: {consumed_iterations}")
    print(f"Porcentaje mal clasificado: {percentage_error(dataset, labels, perceptron_weights) * 100}%")
    wait_for_user_input()

    # Mostramos la grafica de progreso del error
    print("Mostrando grafica de la evolucion del error")
    plot_error_evolution(error_at_iteration, "Iteracion del error por iteracion de PLA", "Iteraciones", "% mal clasificados")

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




# Funcion principal
# ===================================================================================================
if __name__ == "__main__":
    # Fijamos la semilla para no depender tanto de la aleatoriedad y conseguir resultados
    # reproducibles
    # TODO -- descomentar esto para fijar la semilla aleatoria
    # np.random.seed(123456789)

    # Lanzamos el primer ejercicio
    # TODO -- descomentar este codigo para lanzar el primer ejercicio
    #ejercicio1()

    # Lanzamos el segundo ejercicio
    ejercicio2()
