from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
import matplotlib.pyplot as plt
# ver dtreeviz,graphviz
import sklearn
def arbol_de_regresion(data):
    data.drop(['Choice'], 1).hist()
    plt.show()
    X = data.drop(['Choice'], 1)
    y = data['Choice']
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Choice'], 1), data['Choice'], test_size=0.2,random_state=42)
    tree_params = {'max_depth': range(2, 11)}
    tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=17),
                             tree_params,
                             cv=5)
    tree_grid.fit(X_train, y_train)
    print("Best params", tree_grid.best_params_)
    print("Best score", tree_grid.best_score_)
    modelo_final=DecisionTreeClassifier(max_depth=4, random_state=17)
    modelo_final.fit(X_train, y_train)
    features_list=data.columns
    plt.figure(figsize=(22, 10))
    sklearn.tree.plot_tree(modelo_final,
                           feature_names=features_list,
                           class_names='actual',
                           fontsize=8);
    plt.show()
    return (modelo_final)




