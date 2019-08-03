import pandas as pd
import numpy as np
import regex as re
import pickle
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

np.random.seed(500)

dataframe = pd.read_csv(r"english_dataset/english_dataset.tsv", delimiter='\t', encoding='latin-1')

# Preprocessing Data

# Step - a : Remove blank rows if any.
dataframe['text'].dropna(inplace=True)

# Some simple cleaning of data
dataframe['text'] = [re.sub('\d','0', entry) for entry in dataframe['text']]

# Modify URLs to <url>
dataframe['text'] = [re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", entry) for entry in dataframe['text']]

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
dataframe['text'] = [entry.lower() for entry in dataframe['text']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
dataframe['text']= [word_tokenize(entry) for entry in dataframe['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(tqdm(dataframe['text'])):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    dataframe.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y1, Test_Y1, Train_Y2, Test_Y2, Train_Y3, Test_Y3 = model_selection.train_test_split(dataframe['text_final'],dataframe['task_1'],dataframe['task_2'],dataframe['task_3'],test_size=0.3)

Encoder = LabelEncoder()

Train_Y1 = Encoder.fit_transform(Train_Y1)
Test_Y1 = Encoder.fit_transform(Test_Y1)
Train_Y2 = Encoder.fit_transform(Train_Y2)
Test_Y2 = Encoder.fit_transform(Test_Y2)
Train_Y3 = Encoder.fit_transform(Train_Y3)
Test_Y3 = Encoder.fit_transform(Test_Y3)

pickle.dump(Train_Y1, open("TrainY1.pkl", "wb"))
pickle.dump(Test_Y1, open("TestY1.pkl", "wb"))
pickle.dump(Train_Y2, open("TrainY2.pkl", "wb"))
pickle.dump(Test_Y2, open("TestY2.pkl", "wb"))
pickle.dump(Train_Y3, open("TrainY3.pkl", "wb"))
pickle.dump(Test_Y3, open("TestY3.pkl", "wb"))

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(dataframe['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

pickle.dump(Train_X_Tfidf, open("TrainXTfidf.pkl", "wb"))
pickle.dump(Test_X_Tfidf, open("TestXTfidf.pkl", "wb"))