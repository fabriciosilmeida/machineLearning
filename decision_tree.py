#Codigo de machine learning
from sklearn.tree import DecisionTreeClassifier

#Medidor de precisão
from sklearn.metrics import accuracy_score

def decisionTree(X_train, Y_train, X_test, Y_test, predict):
    #Iniciando classificador
    clf = DecisionTreeClassifier()
    #Treinando modelo
    clf.fit(X_train, Y_train)
    #Precisão
    clfs = clf.predict(X_test)
    acuracy = int(accuracy_score(Y_test, clfs)*100)
    #Previsão
    predict = clf.predict([predict])
    #Mostrando resultado da predição
    print("Algoritimo: Decision Tree")
    print("Precisão do treinamento: {}%".format(acuracy))
    print("Previsão: {} \n".format(predict))    