from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
def modelo_definitivo(Data):
    X = Data.drop(['Choice'], 1)
    y = Data['Choice']
    X_train, X_test, y_train, y_test = train_test_split(Data.drop(['Choice'], 1), Data['Choice'], test_size=0.2,
                                                        random_state=42)
    gbct = GradientBoostingClassifier(max_depth=4,
                                  n_estimators=100,
                                  random_state=42)
    gbct.fit(X_train, y_train)
    names=Data.columns
    scores = sorted(zip(map(lambda x: round(x, 4), gbct.feature_importances_), names), reverse=True)
    print(pd.DataFrame(scores, columns=['Score', 'Feature']))

    y_pred_gbct = gbct.predict(X_test)
    print('Árbol regresión después de gbct:',accuracy_score(y_test, y_pred_gbct))
    logistic_params = {'penalty': ['l2','none'],'C':np.logspace(0, 4, 10)}
    clf = GridSearchCV(estimator=LogisticRegression(max_iter=100),
                       param_grid=logistic_params,
                       n_jobs=-1,
                       cv=10)
    clf.fit(X_train,y_train)
    ada_clf = AdaBoostClassifier(base_estimator=clf.best_estimator_,
                                 n_estimators=100,
                                 learning_rate=0.5,
                                 random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print('Modelo logístico después de ajustarlo:',accuracy_score(y_test, y_pred))
    estimators = [('ada', ada_clf), ('gbct', gbct)]
    soft_voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    soft_voting_clf.fit(X_train, y_train)
    predictions = soft_voting_clf.predict(X_test)
    print('Modelo final ensamblado con soft voting:', accuracy_score(y_test, predictions))
    return soft_voting_clf
