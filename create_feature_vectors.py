import sys
import pickle
import pandas
import numpy as np

model_type = sys.argv[1]
data_amount = sys.argv[2]
try:
    weighted_type = sys.argv[3]
except:
    weighted_type = ''

print("model_type = ", model_type)
print("data_amount = ", data_amount)
print("weighted_type = ", weighted_type)

# Train data
dataframe = pandas.read_csv("english_dataset/english_dataset.tsv", delimiter='\t', names=['sentence', 'task1', 'task2', 'task3'], header=None)

dataset = dataframe.values
X_train = dataset[1:,0]
Y1_train = dataset[1:,1]
Y2_train = dataset[1:,2]
Y3_train = dataset[1:,3]

X_train = process_reviews(X_train)

print("set(Y_train): {}".format(set(Y_train)))

# with open("aspect_term_list.pkl", "rb") as f:
#     aspect_term_list = pickle.load(f)

# with open("aspect_weights.pkl", "rb") as f:
#     aspect_weights = pickle.load(f)

# with open("aspect_term_mapping.pkl", "rb") as f:
#     aspect_term_mapping = pickle.load(f)

def softmax(l):
    return np.exp(l)/np.sum(np.exp(l))

# For different weights assigned to aspects
# if weighted_type == 'random_weighted':
#     weights = [0,0,0,0,0]
#     for i in range(5):
#         weights[i] = random.random()
#     weights = softmax(weights)
#     aspects = ['food', 'service', 'price', 'ambience', 'misc']
#     aspect_weights = {}
#     for i, aspect in enumerate(aspects):
#         aspect_weights[aspect] = weights[i]

# elif weighted_type == 'uniform_weighted':
#     aspect_weights = {'food': 0.2, 'service': 0.2, 'price': 0.2, 'ambience': 0.2, 'misc': 0.2}

# else:
#     pass


print("Fetching sentence embeddings...")
with open("embedding_train_"+data_amount+".pkl", "rb") as f:
    sentences_embedding_model = pickle.load(f)

train1_feature_vector_file = "train1_feature_vector.txt"
test1_feature_vector_file = "test1_feature_vector.txt"
train2_feature_vector_file = "train2_feature_vector.txt"
test2_feature_vector_file = "test2_feature_vector.txt"
train3_feature_vector_file = "train3_feature_vector.txt"
test3_feature_vector_file = "test3_feature_vector.txt"

# if weighted_type == "random_weighted":
#     train_feature_vector_file = train_feature_vector_file + "_random.txt"
#     test_feature_vector_file = test_feature_vector_file + "_random.txt"
# elif weighted_type == "uniform_weighted":
#     train_feature_vector_file = train_feature_vector_file + "_uniform.txt"
#     test_feature_vector_file = test_feature_vector_file + "_uniform.txt"
# elif weighted_type == "":
#     if model_type == "weighted":
#         train_feature_vector_file = train_feature_vector_file + "_aspect.txt"
#         test_feature_vector_file = test_feature_vector_file + "_aspect.txt"
#     else:
#         train_feature_vector_file = train_feature_vector_file + ".txt"
#         test_feature_vector_file = test_feature_vector_file + ".txt"

f_train = open(train_feature_vector_file, 'w')

for sentence, label in zip(X_train, Y_train):
    word_embeddings = []
    for word in sentence:
        if model_type == 'weighted' and word in aspect_term_list:
            word_embeddings.append(sentences_embedding_model[word] * aspect_weights[aspect_term_mapping[word]])
        else:
            word_embeddings.append(sentences_embedding_model[word])
    emb_val = np.sum(word_embeddings, axis=0)/len(sentence)
    f_train.write(str(label) + ' ')
    for val in emb_val:
        f_train.write(str(val) + ' ')
    f_train.write('\n')
f_train.close()

# Test data
if data_amount == 'less_data':
    dataframe = pandas.read_csv("test.csv")
else:
    dataframe = pandas.read_csv("dataset/merge_test.csv")
print(">>>>>>>>>>>>>> dataframe.shape = ", dataframe.shape)
dataset = dataframe.values
X_test = dataset[0:,0]
Y_test = dataset[0:,1].astype(int)

X_test = process_reviews(X_test)

print('set(Y_test): {}'.format(set(Y_test)))

with open("embedding_test_"+data_amount+".pkl", "rb") as f:
    sentences_embedding_model = pickle.load(f)

f_test = open(test_feature_vector_file, 'w')

for sentence, label in zip(X_test, Y_test):
    word_embeddings = []
    for word in sentence:
        if model_type == 'weighted' and word in aspect_term_list:
            word_embeddings.append(sentences_embedding_model[word] * aspect_weights[aspect_term_mapping[word]])
        else:
            word_embeddings.append(sentences_embedding_model[word])
    emb_val = np.sum(word_embeddings, axis=0)/len(sentence)
    f_test.write(str(label) + ' ')
    for val in emb_val:
        f_test.write(str(val) + ' ')
    f_test.write('\n')
f_test.close()