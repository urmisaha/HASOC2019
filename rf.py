import pickle
from sklearn.ensemble import RandomForestClassifier
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
classifier = RandomForestClassifier(n_estimators=50, criterion='entropy')

# Task 1
classifier.fit(Train_X_Tfidf,Train_Y1)

predictions_RF = classifier.predict(Test_X_Tfidf)

f = open("results/eng-results_RF.txt", "w+")

# Use score functions
f.write("\nRF Scores\n")
f.write("\nTask 1:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_RF, Test_Y1)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y1, predictions_RF)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y1, predictions_RF)))

f.write("\n----------------------------------------------------------------------------\n")

# Task 2
classifier.fit(Train_X_Tfidf,Train_Y2)

predictions_RF = classifier.predict(Test_X_Tfidf)

# Use score functions
f.write("\nRF Scores\n")
f.write("\nTask 2:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_RF, Test_Y2)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y2, predictions_RF)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y2, predictions_RF)))

f.write("\n----------------------------------------------------------------------------\n")

# Task 3
classifier.fit(Train_X_Tfidf,Train_Y3)

predictions_RF = classifier.predict(Test_X_Tfidf)

# Use score functions
f.write("\nRF Scores\n")
f.write("\nTask 3:\n")
f.write("\nAccuracy\n")
f.write(str(accuracy_score(predictions_RF, Test_Y3)*100))
f.write("\nPrecision recall fscore\n")
f.write(str(precision_recall_fscore_support(Test_Y3, predictions_RF)))
f.write("\nClassification Report\n")
f.write(str(classification_report(Test_Y3, predictions_RF)))

f.write("\n----------------------------------------------------------------------------\n")

f.close()