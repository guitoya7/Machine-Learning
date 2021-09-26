import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler
def visualizar_datavs():
    dir_train=r'C:\Users\guito\PycharmProjects\Trabajo_redes_sociales\train.csv'
    df_train = pd.read_csv(dir_train)
    print('Dat:',df_train.describe())
    #MOSTRAMOS LOS DATOS TAL Y COMO LOS TENEMOS POR PARES:
    fig = plt.figure(figsize=(20, 30))
    for i in range(11):
        ax = plt.subplot(4, 3, i + 1)
        sns.distplot(df_train[df_train.columns.values[i + 1]]).set_title(
            df_train.columns.values[i + 1]+' ' + 'vs'+ ' ' + df_train.columns.values[i + 12])
        sns.distplot(df_train[df_train.columns.values[i + 12]])
    plt.show()
    #MOSTRAMOS LOS DATOS APLICANDO LOGARITMO
    df_train_log=df_train.drop('Choice',1).apply(lambda x: np.log(x+1))
    print('log', df_train_log.describe())
    df_train_log['Choice']=df_train['Choice']
    fig = plt.figure(figsize=(20,30))
    for i in range(11):
        ax=plt.subplot(4,3,i+1)
        sns.distplot(df_train_log[df_train.columns.values[i+1]]).set_title(df_train.columns.values[i+1]+'vs'+df_train.columns.values[i+12])
        sns.distplot(df_train_log[df_train.columns.values[i+12]])
    plt.show()
    #MOSTRAMOS LOS DATOS APLICANDO LA TRANSFORMACION STANDARD SCALER
    scaler = StandardScaler()
    scaler.fit(df_train.drop('Choice', 1))
    df_standard_scaler = scaler.transform(df_train.drop('Choice', 1))
    df_standard_scaler = pd.DataFrame(df_standard_scaler)
    df_standard_scaler['Choice'] = df_train['Choice']
    fig = plt.figure(figsize=(20, 30))
    for i in range(11):
        ax = plt.subplot(4, 3, i + 1)
        sns.distplot(df_standard_scaler[df_standard_scaler.columns.values[i]]).set_title(
            df_train.columns.values[i + 1] + 'vs' + df_train.columns.values[i + 12])
        sns.distplot(df_standard_scaler[df_standard_scaler.columns.values[i + 11]])
    plt.show()
    #MOSTRAMOS LOS DATOS APLICANDO LA TRANSFORMACION 1/(x+1)
    df_train_3 = df_train.drop('Choice', 1).apply(lambda x: 1 / (x + 1))
    df_train_3['Choice'] = df_train['Choice']
    fig = plt.figure(figsize=(20, 30))
    for i in range(11):
        ax = plt.subplot(4, 3, i + 1)
        sns.distplot(df_train_3[df_train_3.columns.values[i]])
        sns.distplot(df_train_3[df_train_3.columns.values[i + 11]]).set_title(
            df_train.columns.values[i + 1] + 'vs' + df_train.columns.values[i + 12])
    plt.show()
    return(df_train_log)
