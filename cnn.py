import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import precision_recall_fscore_support


word2idx = pickle.load(open(f'pickle_files/word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'pickle_files/idx2word.pkl', 'rb'))

train_sentences = pickle.load(open(f'pickle_files/train_sentences.pkl', 'rb'))
val_sentences = pickle.load(open(f'pickle_files/val_sentences.pkl', 'rb'))
test_sentences = pickle.load(open(f'pickle_files/test_sentences.pkl', 'rb'))
Train_Y1 = pickle.load(open(f'pickle_files/Train_Y1.pkl', 'rb'))
Train_Y2 = pickle.load(open(f'pickle_files/Train_Y2.pkl', 'rb'))
Train_Y3 = pickle.load(open(f'pickle_files/Train_Y3.pkl', 'rb'))
val_Y1 = pickle.load(open(f'pickle_files/val_Y1.pkl', 'rb'))
val_Y2 = pickle.load(open(f'pickle_files/val_Y2.pkl', 'rb'))
val_Y3 = pickle.load(open(f'pickle_files/val_Y3.pkl', 'rb'))
Test_Y1 = pickle.load(open(f'pickle_files/Test_Y1.pkl', 'rb'))
Test_Y2 = pickle.load(open(f'pickle_files/Test_Y2.pkl', 'rb'))
Test_Y3 = pickle.load(open(f'pickle_files/Test_Y3.pkl', 'rb'))

TASK = sys.argv[1]

if TASK == 1:
    train_labels = Train_Y1
    val_labels = val_Y1
    test_labels = Test_Y1
    num_classes = 2                 # Classes: NOT(Non Hate-Offensive) and HOF(Hate and Offensive)
elif TASK == 2:
    train_labels = Train_Y2
    val_labels = val_Y2
    test_labels = Test_Y2
    num_classes = 3                 # Classes: HATE(Hate), OFFN(Offensive) and PRFN(Profane)
else:
    train_labels = Train_Y3
    val_labels = val_Y3
    test_labels = Test_Y3
    num_classes = 2                 # Classes: TIN(Targeted Insult) and UNT(Untargeted)

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

# Parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 150
display_step = 1

train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


# Network Parameters
hidden_size = 100               # 1st layer and 2nd layer number of features
input_size = len(word2idx) + 1  # Words in vocab

class CNN(nn.Module):
 def __init__(self, input_size, hidden_size, num_classes):
     super(CNN, self).__init__()
     self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
     self.relu = nn.ReLU()
     self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
     self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

 def forward(self, x):
     out = self.layer_1(x)
     out = self.relu(out)
     out = self.layer_2(out)
     out = self.relu(out)
     out = self.output_layer(out)
     return out

CNN_model = CNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=learning_rate)

# Train the Model
print("Training model...")
for epoch in range(num_epochs):
    # total_batch = int(len(newsgroups_train.data)/batch_size)
    # for i in range(total_batch):
    #     batch_x,batch_y = get_batch(newsgroups_train,i,batch_size)
    #     articles = Variable(torch.FloatTensor(batch_x))
    #     labels = Variable(torch.FloatTensor(batch_y))
    i = 0
    for inputs, labels in train_loader:
        # Forward + Backward + Optimize
        inputs = Variable(torch.FloatTensor(inputs))
        labels = Variable(torch.LongTensor(labels))
        optimizer.zero_grad() # zero the gradient buffer
        outputs = CNN_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 4 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_sentences)//batch_size, loss.data[0]))
        i = i + 1

# Test the Model
print("Testing model...")
correct = 0
total = 0
total_test_data = len(test_sentences)

total_labels = torch.LongTensor()
total_preds = torch.LongTensor()

for inputs, labels in test_loader:
    inputs = Variable(torch.FloatTensor(inputs))
    labels = Variable(torch.LongTensor(labels))
    outputs = CNN_model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    total_labels = torch.cat((total_labels, torch.LongTensor(labels)))
    total_preds = torch.cat((total_preds, torch.LongTensor(predicted)))

# batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
# articles = Variable(torch.FloatTensor(batch_x_test))
# labels = Variable(torch.LongTensor(batch_y_test))
# outputs = net(articles)
# _, predicted = torch.max(outputs.data, 1)
# total += labels.size(0)
# correct += (predicted == labels).sum()

print("Printing results::: ")
print("Last lot predicted:")
print(predicted)
labels = total_labels.data.numpy()
preds = total_preds.data.numpy()

print("weighted precision_recall_fscore_support:")
print(precision_recall_fscore_support(labels, preds, average='weighted'))
print("============================================")

print(precision_recall_fscore_support(labels, preds, average=None))
print("============================================")
    
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")