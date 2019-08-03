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