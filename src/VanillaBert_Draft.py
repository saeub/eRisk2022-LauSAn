"""
Exploratory file to test Bert model.
WORK IN PROGRESS! -> Hochgeladen, damit Sooyeon es sich anschauen kann.

todo:

- method to feed training data to the model (now: only the first post -> join all posts and divide them into batches of size 512?)
- early decision strategies (aktuell evaluiert es anhand des 1 posts)
- Integrate into models.py
- try out different truncation methods (head-only, tail-only, head+tail) für zu lange texte
- play around with model, see if we can get better results

Allg. Fragen:
- filtern wir die links raus oder lassen wir die drin?
- validation data needed?
"""

import os
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm

from data import parse_subject

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {True: 1, False: 0}

# Data preparation
def get_subj_labels_text(filename):
    """Takes a file name and returns label and texts for the file."""
    subject = parse_subject(filename)
    subj_label = subject.label
    subj_texts = []

    for post in subject.posts:
        subj_texts.append(post.text)

    #subj_texts_joined = ' '.join(subj_texts) # join all texts of one subject in training

    #return subj_label, subj_texts_joined
    return subj_label, subj_texts[0] #todo: hier habe ich nur den ersten Text genommen, weil ich mich noch nicht um die early decision strategie gekümmert habe


def get_train_data(train_set):
    """Takes a list of xml file names and returns lists for labels and texts for all subjects in data set."""
    train_data_paths = []

    # get filenames of training set
    with open(train_set, 'r') as infile:
        names = infile.readlines()
        for name in names:
            train_data_paths.append(os.path.join("../", name.rstrip()))

    # get labels and texts from all subjects in training set
    labels = []  # labels of all subjects
    texts = []  # texts of all subjects
    for file in train_data_paths:
        subj_label, subj_texts = get_subj_labels_text(file)[0], get_subj_labels_text(file)[1]
        labels.append(subj_label)
        texts.append(subj_texts)

    return labels, texts


# encode texts
#def encode_subject_texts(filename):
    #"""Takes a subject object as input and returns embeddings as a list of dictionaries.
    #The dictionaries each contain input_ids, token_type_ids and attention_mask."""

    #subject = parse_subject(filename)

    #bert_input = []  # bert input for one subject

    #for post in subject.posts:
        #text = post.text
        ## Encode the sentence
        ##encoded = tokenizer.encode_plus(
            #text=text,  # the sentence to be encoded
            #add_special_tokens=True,  # Add [CLS] and [SEP]
            #max_length=64,  # maximum length of a sentence
            #padding='max_length',  # Add [PAD]s
            #return_attention_mask=True,  # Generate the attention mask
            #return_tensors='pt',  # ask the function to return PyTorch tensors
        #)
        #bert_input.append(encoded)

    #return bert_input


class Dataset(torch.utils.data.Dataset):
    """Generates data for BertClassifier.
    The text variable contains the encoded text."""

    def __init__(self, data_path):
        # [labels[label] for label in df['category']]
        self.labels = [labels[label] for label in
                       (get_train_data(data_path)[0])]  # map labels to boolean -> list

        # list of dictionaries (input ids, token type ids, attention mask)
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in (get_train_data(data_path)[
            1])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # _ containts the embedding vectors of all the tokens in a sequence
        # pooled_output contains the embedding vector of [CLS] token -> use for classification
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)  # output is vector of size 2 (corresponds to categories)

        return final_layer


def train(model, train_data, learning_rate, epochs):
    #train, val = Dataset(train_data), Dataset(val_data)
    train = Dataset(train_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        #total_acc_val = 0
        #total_loss_val = 0

        #with torch.no_grad():

            #for val_input, val_label in val_dataloader:
                #val_label = val_label.to(device)
                #mask = val_input['attention_mask'].to(device)
                #input_id = val_input['input_ids'].squeeze(1).to(device)

                #output = model(input_id, mask)

                #batch_loss = criterion(output, val_label)
                #total_loss_val += batch_loss.item()

                #acc = (output.argmax(dim=1) == val_label).sum().item()
                #total_acc_val += acc

        #print(
            #f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            #    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            #    | Val Loss: {total_loss_val / len(val_data): .3f} \
             #   | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f}')


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def main():
    train_data = "../data/train_set_1.txt"
    test_data = "../data/test_set_1.txt"
    #print(get_subj_labels_text("../data/training_t2/TRAINING_DATA/2017_cases/neg/test_subject25.xml"))
    #print(get_train_data(train_data))
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    train(model, train_data, LR, EPOCHS)
    evaluate(model, test_data)


if __name__ == '__main__':
    main()



