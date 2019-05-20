#Codigo de machine learning
from sklearn.naive_bayes import GaussianNB

#Medidor de precisão
from sklearn.metrics import accuracy_score

def naiveBayes(X_train, Y_train, X_test, Y_test, predict):
    #Iniciando classificador
    clf = GaussianNB()
    #Treinando modelo
    clf.fit(X_train, Y_train)
    #Precisão
    clfs = clf.predict(X_test)
    acuracy = int(accuracy_score(Y_test, clfs)*100)
    #Previsão
    predict = clf.predict([predict])
    #Mostrando resultado da predição
    print("")
    print("Algoritimo: Naive Bayes")
    print("Precisão do treinamento: {}%".format(acuracy))
    print("Previsão: {} \n".format(predict))    