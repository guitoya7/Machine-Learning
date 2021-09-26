import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import seaborn as sns
def regresion_logistica(data):
    data.drop(['Choice'], 1).hist()
    plt.show()
    X=data.drop(['Choice'],1)
    y=data['Choice']
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Choice'],1),data['Choice'],test_size=0.2, random_state=42)
    ####################################################################################################################
    #Validación cruzada
    ####################################################################################################################
    kf = KFold(n_splits=5)
    model = linear_model.LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    print("Media de cross_validation para ver su estabilidad", scores.mean())
    model.fit(X,y)
    coeff_df = pd.DataFrame(model.coef_.T,
                            X.columns,
                            columns=['Coefficient'])
    print('Coeficientes del modelo para importancia de variables ',coeff_df,model.intercept_)

    return model
