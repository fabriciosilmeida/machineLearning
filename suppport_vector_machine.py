#Codigo de machine learning
from sklearn.svm import SVC

#Medidor de precisão
from sklearn.metrics import accuracy_score

def svm(X_train, Y_train, X_test, Y_test, predict):
    #Iniciando classificador
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    #Treinando modelo
    clf.fit(X_train, Y_train)
    #Precisão
    clfs = clf.predict(X_test)
    acuracy = int(accuracy_score(Y_test, clfs)*100)
    #Previsão
    predict = clf.predict([predict])
    #Mostrando resultado da predição
    print("Algoritimo: Support Vector Machine")
    print("Precisão do treinamento: {}%".format(acuracy))
    print("Previsão: {} \n".format(predict))    