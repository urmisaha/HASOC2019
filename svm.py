import pickle
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

Train_Y1 = pickle.load(open("TrainY1.pkl", "rb"))
Test_Y1 = pickle.load(open("TestY1.pkl", "rb"))
Train_Y2 = pickle.load(open("TrainY2.pkl", "rb"))
Test_Y2 = pickle.load(open("TestY2.pkl", "rb"))
Train_Y3 = pickle.load(open("TrainY3.pkl", "rb"))
Test_Y3 = pickle.load(open("TestY3.pkl", "rb"))

Train_X_Tfidf = pickle.load(open("TrainXTfidf.pkl", "rb"))
Test_X_Tfidf = pickle.load(open("TestXTfidf.pkl", "rb"))

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':('auto', 'scale')}
svr = svm.SVC(verbose=2)
SVM = GridSearchCV(svr, parameters, verbose=2)

# Task 1
SVM.fit(Train_X_Tfidf,Train_Y1)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

f = open("results/eng-results_SVM.txt", "w+")
# Use score functions
f.write("\nSVM Scores\n")
f.write("\nTask 1:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_SVM, Test_Y1)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y1, predictions_SVM)))
f.write("\nConfusion Matrix:\n")
f.write(str(confusion_matrix(Test_Y1, predictions_SVM)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y1, predictions_SVM)))

f.write("\n----------------------------------------------------------------------------\n")

# Task 2
SVM.fit(Train_X_Tfidf,Train_Y2)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use score functions
f.write("\nSVM Scores\n")
f.write("\nTask 2:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_SVM, Test_Y2)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y2, predictions_SVM)))
f.write("\nConfusion Matrix:\n")
f.write(str(confusion_matrix(Test_Y2, predictions_SVM)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y2, predictions_SVM)))

f.write("\n----------------------------------------------------------------------------\n")

# Task 3
SVM.fit(Train_X_Tfidf,Train_Y3)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use score functions
f.write("\nSVM Scores\n")
f.write("\nTask 3:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_SVM, Test_Y3)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y3, predictions_SVM)))
f.write("\nConfusion Matrix:\n")
f.write(str(confusion_matrix(Test_Y3, predictions_SVM)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y3, predictions_SVM)))

f.write("\n----------------------------------------------------------------------------\n")

f.close()