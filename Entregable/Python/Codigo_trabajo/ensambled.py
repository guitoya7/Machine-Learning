from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,accuracy_score,confusion_matrix
def ensambles(Data):
    X = Data.drop(['Choice'], 1)
    y = Data['Choice']
    X_train, X_test, y_train, y_test = train_test_split(Data.drop(['Choice'], 1), Data['Choice'], test_size=0.2,random_state=42)
    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    estimators = [('lr', log_clf),('rf', rnd_clf)]
    soft_voting_clf = VotingClassifier(estimators = estimators, voting = 'soft')
    soft_voting_clf.fit(X_train, y_train)
    predictions = soft_voting_clf.predict(X_test)
    print('Puntuación de nuestro modelo ensamblado simple:',accuracy_score(y_test, predictions))
    return soft_voting_clf