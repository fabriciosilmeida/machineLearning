#Algoritimos de machine learning
from naive_bayes import naiveBayes
from suppport_vector_machine import svm
from decision_tree import decisionTree

#Base de dados
from data_base import X_train, Y_train, input_train, X_test, Y_test, input_test

#Dados para predição
predict = [-0.8, -1]

#Naive Bayes
naiveBayes(X_train, Y_train, X_test, Y_test, predict)

#Support Vector Machine
svm(X_train, Y_train, X_test, Y_test, predict)

#Decision Tree
decisionTree(X_train, Y_train, X_test, Y_test, predict)