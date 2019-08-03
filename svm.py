import pickle
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

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
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

# Task 1
SVM.fit(Train_X_Tfidf,Train_Y1)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use score functions
print("SVM Scores")
print("Task 1:")
print("Accuracy")
print(accuracy_score(predictions_SVM, Test_Y1)*100)
print("Precision recall fscore")
print(precision_recall_fscore_support(Test_Y1, predictions_SVM))
print("Classification Report")
print(classification_report(Test_Y1, predictions_SVM))

print("----------------------------------------------------------------------------")

# Task 2
SVM.fit(Train_X_Tfidf,Train_Y2)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use score functions
print("SVM Scores")
print("Task 2:")
print("Accuracy")
print(accuracy_score(predictions_SVM, Test_Y2)*100)
print("Precision recall fscore")
print(precision_recall_fscore_support(Test_Y2, predictions_SVM))
print("Classification Report")
print(classification_report(Test_Y2, predictions_SVM))

print("----------------------------------------------------------------------------")

# Task 3
SVM.fit(Train_X_Tfidf,Train_Y3)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use score functions
print("SVM Scores")
print("Task 3:")
print("Accuracy")
print(accuracy_score(predictions_SVM, Test_Y3)*100)
print("Precision recall fscore")
print(precision_recall_fscore_support(Test_Y3, predictions_SVM))
print("Classification Report")
print(classification_report(Test_Y3, predictions_SVM))

print("----------------------------------------------------------------------------")