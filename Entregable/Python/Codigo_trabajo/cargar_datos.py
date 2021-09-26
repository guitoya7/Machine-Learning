def cargar_datos():
    import pandas as pd
    dir_train=r'C:\Users\guito\PycharmProjects\Trabajo_redes_sociales\train.csv'
    dir_test=r'C:\Users\guito\PycharmProjects\Trabajo_redes_sociales\test.csv'
    df_train = pd.read_csv(dir_train)
    df_test=pd.read_csv(dir_test)
    return [df_train, df_test]
