#!/usr/bin/env python
"""
- Data preprocessing: Some steps to augment training data
- Model: Bert/Longformer + some layers
Model is heavily inspired by https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
"""
import os
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

from data import parse_subject


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # TODO: Why can't i put that in the main function???


def get_datasets(train_files, test_files) -> Tuple[List, List, List]:
    """Takes txt-files with the names of xml-files from our prior train-test split. Splits our train set into a
    train and val set since we need that for BERT and returns lists with xml-filenames for test, train and val data.
    :param train_files: txt-file with names of xml-files.
    :param test_files: txt-file with names of xml-files.
    """

    # train, val split from training data txt-file
    train_data_paths = []  # 80% of train set (txt-file)
    val_data_paths = []  # 20% of train set

    # get lists of filenames of training set
    with open(train_files, 'r') as infile:
        names = infile.readlines()
        # train, val split
        train, val = names[:int(len(names) * 0.8)], names[-int(len(names) * 0.2):]

        for name in train:
            train_data_paths.append(name.rstrip())
        for name in val:
            val_data_paths.append(name.rstrip())

    # list with filenames for test set
    with open(test_files, 'r') as infile:
        test_data_paths = [line.rstrip() for line in infile.readlines()]

    return train_data_paths, val_data_paths, test_data_paths


################################# prepare data for DataLoader Class ####################################

def merge_posts(posts: List[str], number: int, overlap: int, max_len: int) -> List[str]:
    """
    Takes a list of strings (list of all posts by one subject) and merges strings
    in the list according to the specifications from the parameters. The strings are
    merged in reverse order so that the oldest post is to the right and the newest
    post is to the left.
    :param posts: a list of strings (posts by one subject)
    :param number: the number of strings that should get merged into one string,
    must be > 0 (e.g. number = 2 will always merge two strings together)
    :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
    :param max_len: maximal input length for model (e.g. 512 or 4096)
    """

    merged_posts = []
    step = number - overlap
    for i in range(0, len(posts) - 1, step):
        # put the number of required sentences in a list
        count = 0  # repeat while loop as many times as the number of sentences we want to concatinate
        step2 = 0  # counter so it knows which sentence to pick next
        merged_sentence = []  # list for required sentences that need to be merged together

        while count < number:  # for as many times as the number of sentences we want to concatinate
            try:
                sentence = posts[i + step2]
                count += 1  # make one more iteration if the number of required sentence hasn't been reached yet
                step2 += 1  # take one sentence to the right next time

                merged_sentence.append(sentence)
            except IndexError:
                break

        # nur sätze nehmen, bei denen es aufgeht (=duplikate vermeiden) und die ins modell passen
        if len(merged_sentence) == number:
            merged_sentence.reverse()  # newest post on the left (will be truncated on the right)
            merged_sent_str = ' '.join(merged_sentence)
            if len(merged_sent_str.split()) <= max_len:
                merged_posts.append(merged_sent_str)

    return merged_posts


def data_augmentation(posts: List[str], numbers_concat: List[int], overlap: int, max_len: int) -> List[str]:
    """
    Function to augment the training and validation data.
    Takes a list of strings and returns concatinations of 2 posts, 3 posts, etc.
    The newest post is always at the beginning of the string, the oldest at the end.
    :param posts: a list of strings (posts by one subject)
    :param numbers_concat: list of integers that determines how many strings should be concatinated.
    :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
    :param max_len: maximal input length for model (e.g. 512 or 4096)
    """

    augmented_data = []

    # current post only (no history)
    for post in posts:
        augmented_data.append(post)

    # current post + n posts of history
    for n in numbers_concat:
        # TODO: try out if it works better with an overlap (e.g. overlap 10% of n --> more data)
        for s in merge_posts(posts, n, 0, 512):
            augmented_data.append(s)

    return augmented_data


def prepare_subject_data(filename, numb_conc: List[int], overlap: int, max_len: int):
    """Takes a filename for a subject and returns two lists:
    - list of labels of the same length as the list of augmented posts data
    - augmented data: list of merged posts
    :param filename: xml file for a subject
    :param numb_conc: list with numbers that determine how many posts of a subject should be concatinated.
    :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
    :param max_len: maximal input length for model (e.g. 512 or 4096)
    """
    # mapping label
    labels = {True: 1, False: 0}

    # parse subject
    subject = parse_subject(filename)
    subject_id = subject.id
    subject_texts = [p.text for p in subject.posts]

    if subject.label == True:
        subject_label = 1
    elif subject.label == False:
        subject_label = 0

    # augment text
    augmented_texts = data_augmentation(subject_texts, numb_conc, overlap, max_len)

    # get list with labels which is as long as augmented text list
    labels = [subject_label] * len(augmented_texts)

    return labels, augmented_texts


def prepare_dataset(dataset, numb_conc: List[int], overlap: int, max_len: int) -> Tuple[List, List]:
    """Takes a list of file names (all file names from train or val set) and returns
  a list of labels and a list of strings that can be fed into the Dataloader class.
  :param dataset: list of xml file names
  :param numb_conc: list with numbers that determine how many posts of a subject should be concatinated.
  :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
  :param max_len: maximal input length for model (e.g. 512 or 4096)
  """
    all_labels = []
    all_texts = []
    for subject in dataset:
        info = prepare_subject_data(subject, numb_conc, overlap, max_len)
        for i in info[0]:
            all_labels.append(i)
        for i in info[1]:
            all_texts.append(i)

    return all_labels, all_texts


def prepare_test_dataset(dataset, numb_conc, overlap, max_len):
    """Takes a list of file names (all file names from test set) and returns
    a list concatinated sentences to test the model with.
    :param dataset: list of xml file names
    :param overlap: 0 if no overlap, 1 if 1 string overlap etc.
    :param max_len: maximal input length for model (e.g. 512 or 4096)
    """
    # all_labels = []
    all_texts = []
    for subject in dataset:
        for i in prepare_subject_data(subject, numb_conc, overlap, max_len)[1]:
            all_texts.append(i)

    return all_texts


################################################ Dataset Class ###################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.labels = data[0]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in data[1]]

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


################################## Model ############################################

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


################################## Training Loop ############################################


def training_loop(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

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

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


################################## Evaluation ############################################

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
    # path to the file names of our initial train/test split
    # train_data = "../data/train_set_1.txt" # TODO
    # test_data = "../data/test_set_1.txt"  # TODO
    train_data = "data/train_set_1_small.txt"
    test_data = "data/test_set_1_small.txt"

    # get lists with file names for training, validation and test set (we need a val set for BERT)
    train, val, test = get_datasets(train_data, test_data)

    # augmentation of training data
    # for each subject, this will concationate 2 posts, 3 posts...50 posts together as long as max length is not met.
    numbers_to_concatinate = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]

    # prepare data to load into the DataLoader Class
    train_prepared = prepare_dataset(train, numbers_to_concatinate, 0, 512)
    val_prepared = prepare_dataset(val, numbers_to_concatinate, 0, 512)  # (list of labels, list of texts)
    test_prepared = prepare_dataset(test, [5], 0, 512)

    # Train Model
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6
    training_loop(model, train_prepared, val_prepared, LR, EPOCHS)

    # Evaluation
    evaluate(model, test_prepared)


if __name__ == '__main__':
    main()
