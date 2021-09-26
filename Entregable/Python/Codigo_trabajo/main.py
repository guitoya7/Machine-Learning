#############################################
#CODIGO UTILIZADO, IR CERRANDO LAS IMAGENES QUE APARECEN PARA QUE EL CÓDIGO VAYA CORRIENDO
#############################################

#CARGAMOS LAS FUNCIONES A UTILIZAR
from cargar_datos import cargar_datos as cd
from visualizar_datavs import visualizar_datavs as vd
from funar_variables import funar_variables as fv
from pca_analisis import pca_analisis as pa
from modelo_regresion_logistica import regresion_logistica as rl
from Arbol_regresion import arbol_de_regresion as ar
from csv_entregar import csv_entregar as ce
from Analisis_errores import Analisis_errores as ae
from ensambled import ensambles as ens
from modelo_definitivo import modelo_definitivo as mf
#IMPORTAMOS NUMPY, PANDAS Y PICKLE
import numpy as np
import pandas as pd
import pickle
#Cargamos el dataset con el entrenamiento
data= cd()[0]
#Aplicamos la transformación a logaritmo
log_data=vd()
#Aplicamos el estudio por correlación y limpia de variables
simple_data=fv(log_data)
#Aplicamos el analisis de componentes principales
simple_data=pa(simple_data)
#Modelo de regresión logístico
modelo_logistico=rl(simple_data)
#modelo de arbol de clasificación
modelo_arbol_de_regresion=ar(simple_data)
#Primer modelo ensamblado simple
modelo_conjunto=ens(simple_data)
#Modelo ensamblado habiendo optimizado cada uno de los dos modelos anteriores
modelo_df=mf(simple_data)
#Analisis de errores que comento en mi estudio
ae(simple_data,modelo_df,log_data)
#Cargamos el test que subiremos a Kaggle con la transformación que aplicamos al train ya realizada para no evaluar
#en ninguna función que hayamos creado ya que implican gráficas
test_data=cd()[1].apply(lambda x: np.log(x+1)).drop(['B_network_feature_1','A_network_feature_1','B_listed_count',
                                                     'A_listed_count','A_mentions_received','B_mentions_received',
                                                     'A_mentions_sent','B_mentions_sent','A_retweets_sent','B_retweets_sent',
                                                     'A_posts','B_posts'],axis=1)
#Genera 4 excel para subir al kaggle y probar el acierto de cada modelo
ce(test_data,modelo_logistico,modelo_arbol_de_regresion,modelo_conjunto,modelo_df)
#ANTES DE SUBIR CAMBIAR EL NOMBRE DE LAS COLUMNAS MANUAL
#AQUI GUARDAREMOS LOS MODELOS
filename = 'modelo_logistico'
pickle.dump(modelo_logistico, open(filename,'wb'))
filename = 'modelo_arbol_de_regresion'
pickle.dump(modelo_arbol_de_regresion, open(filename,'wb'))
filename = 'modelo_conjunto'
pickle.dump(modelo_conjunto, open(filename,'wb'))
filename = 'modelo_definitivo'
pickle.dump(modelo_df, open(filename,'wb'))