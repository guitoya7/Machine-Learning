import pandas as pd
import numpy as np
def csv_entregar(test_data,modelo1,modelo2,modelo3,modelo4):
    resultado = pd.DataFrame(modelo1.predict_proba(test_data))
    resultado = pd.DataFrame(resultado[resultado.columns[1]])
    resultado.index = np.arange(1, len(resultado) + 1)
    resultado.to_csv('resultado_1_logistica.csv')
    resultado = pd.DataFrame(modelo2.predict_proba(test_data))
    resultado = pd.DataFrame(resultado[resultado.columns[1]])
    resultado.index = np.arange(1, len(resultado) + 1)
    resultado.to_csv('resultado_2_arbolregresion.csv')
    resultado = pd.DataFrame(modelo3.predict_proba(test_data))
    resultado = pd.DataFrame(resultado[resultado.columns[1]])
    resultado.index = np.arange(1, len(resultado) + 1)
    resultado.to_csv('resultado_3_ensembled.csv')
    resultado = pd.DataFrame(modelo4.predict_proba(test_data))
    resultado = pd.DataFrame(resultado[resultado.columns[1]])
    resultado.index = np.arange(1, len(resultado) + 1)
    resultado.to_csv('resultado_4_modelo_definitivo.csv')

