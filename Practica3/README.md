# Tercera práctica de la asignatura

## Dudas al profesor:

1. Mostrar una tabla tan grande es penalizable?
2. Stratify en regresión sería necesario? Porque al no estar trabajando con clases, no tiene sentido preocuparnos por que queden clases infrarepresentadas en test. Lo que si pueden quedar infrarrepresentados son los valores, pero estamos trabajanado con test_size == 4253, y por tanto, esta última situación parece improbable
3. Datos sin balancear => eso solo es aplicable a clasificacion? Ie. si tenemos datos que estan muy acumulados en [0, 10], y unos pocos en [10, 5], eso sería desabalanceo y por tanto digno de tratar de contrarestar?
4. Uso el comando `return df[(np.abs(stats.zscore(df)) < times_std_dev).all(axis=1)]` con scipy.stats para quitar los outliers
5. Quitar los outliers con z-score 4.0 me borra el 9% de los datos de entrenamiento, esto es de esperar? Es aceptable?
