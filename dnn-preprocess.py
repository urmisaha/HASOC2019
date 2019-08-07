import sys
import bz2
from collections import Counter
import re
import nltk
import numpy as np
import pandas
import pickle
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

dataframe = pandas.read_csv(r"english_dataset/english_dataset.tsv", delimiter='\t', encoding='latin-1')

# dataframe = dataframe.sample(frac=1).reset_index(drop=True)   # commented bcoz other statistical models are run without shuffling

Train_X, Test_X, Train_Y1, Test_Y1, Train_Y2, Test_Y2, Train_Y3, Test_Y3 = model_selection.train_test_split(dataframe['text'],dataframe['task_1'],dataframe['task_2'],dataframe['task_3'],test_size=0.3)
train_size = Train_X.shape[0]

train_sentences = Train_X.values
test_sentences = Test_X.values

Encoder = LabelEncoder()

Train_Y1 = Encoder.fit_transform(Train_Y1)
Test_Y1 = Encoder.fit_transform(Test_Y1)
Train_Y2 = Encoder.fit_transform(Train_Y2)
Test_Y2 = Encoder.fit_transform(Test_Y2)
Train_Y3 = Encoder.fit_transform(Train_Y3)
Test_Y3 = Encoder.fit_transform(Test_Y3)

print("Data load completed..")

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences

for i, sentence in enumerate(train_sentences):
    # The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence):  # Tokenizing the words
        words.update([word.lower()])  # Converting all the words to lowercase
        train_sentences[i].append(word)
    if i%100 == 0:
        print(str((i*100)/train_size) + "% done")
print("100% done")

# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 200  # The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

split_frac = 0.5 # 50% validation, 50% test
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_Y1, Test_Y1 = Test_Y1[:split_id], Test_Y1[split_id:]
val_Y2, Test_Y2 = Test_Y2[:split_id], Test_Y2[split_id:]
val_Y3, Test_Y3 = Test_Y3[:split_id], Test_Y3[split_id:]

pickle.dump(train_sentences, open('pickle_files/train_sentences.pkl', 'wb'))
pickle.dump(val_sentences, open('pickle_files/val_sentences.pkl', 'wb'))
pickle.dump(test_sentences, open('pickle_files/test_sentences.pkl', 'wb'))
pickle.dump(Train_Y1, open('pickle_files/Train_Y1.pkl', 'wb'))
pickle.dump(Train_Y2, open('pickle_files/Train_Y2.pkl', 'wb'))
pickle.dump(Train_Y3, open('pickle_files/Train_Y3.pkl', 'wb'))
pickle.dump(val_Y1, open('pickle_files/val_Y1.pkl', 'wb'))
pickle.dump(val_Y2, open('pickle_files/val_Y2.pkl', 'wb'))
pickle.dump(val_Y3, open('pickle_files/val_Y3.pkl', 'wb'))
pickle.dump(Test_Y1, open('pickle_files/Test_Y1.pkl', 'wb'))
pickle.dump(Test_Y2, open('pickle_files/Test_Y2.pkl', 'wb'))
pickle.dump(Test_Y3, open('pickle_files/Test_Y3.pkl', 'wb'))

pickle.dump(word2idx, open('pickle_files/word2idx.pkl', 'wb'))
pickle.dump(idx2word, open('pickle_files/idx2word.pkl', 'wb'))
