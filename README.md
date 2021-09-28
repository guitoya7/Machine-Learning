# Machine-Learning-Twitter
Trabajo donde entrenamos un modelo predictivo que evalua dos a dos que elemento es mas influyente dentro de la red de tweeter y nos dice que variables son mas importantes para la evaluación del mismo.
# Pasos realizados en el análisis
### Tratamiento del dataset:
* Transformación conveniente a nivel unidades y distribución (transformación logarítmica)
* Análisis dimensional para la reducción de variables con coeficientes de correlación y análisis de componentes principales
### Modelaje
* Regresión lineal
* Árbol de clasificación
* Ensamblado de ambos modelos
* Modelo final con optimización de ambos modelos (Gridsearch de parámetros, ada boosting de la logística y gradient boosting classifier con árboles de clasificación)
### Conclusiones y análisis del error
* Remarcación de las principales variables con su importancia en el suceso a estudiar
* Reflexión sobre los puntos que hemos fallado en el análisis
