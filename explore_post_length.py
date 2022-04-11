"""
Some explorations for the longformer model.
We want to concatinate the posts of the training data and feed texts of different length to the longformer.
Which text length makes sense?
Exploring the data...
"""
import os
import numpy as np

from data import parse_subject

def get_subj_labels_text(filename):
    """Takes a file name and returns label and texts for the file."""
    subject = parse_subject(filename)
    subj_label = subject.label
    subj_texts = []

    for post in subject.posts:
        subj_texts.append(post.text)

    return subj_label, subj_texts


def get_data(train_set):
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


def get_metrics(lengths_list, percent_top, percent_bottom):
    """Input: sorted list of string lengths.
    Getting some metrics after cutting off some percentage of longest and shortest posts."""
    if percent_bottom == 0 and percent_top == 0:
        post_lengths = lengths_list
        length = len(post_lengths)
        max_len = max(post_lengths)
        min_len = min(post_lengths)
        mean_len = sum(post_lengths)/length
    else:
        post_lengths = lengths_list[int(len(lengths_list) * percent_bottom): int(len(lengths_list) * (1 - percent_top))]
        length = len(post_lengths)
        max_len = max(post_lengths)
        min_len = min(post_lengths)
        mean_len = sum(post_lengths) / length

    return length, max_len, min_len, mean_len


def percentage_number(numbers_list, number):
    """"Calculates how often a number occurs in a list in percent."""
    percentage = (100/len(numbers_list))*numbers_list.count(number)

    return percentage

def main():
    train_data = "../data/train_set_1.txt"
    test_data = "../data/test_set_1.txt"
    post_lengths = []


    for subject in get_data(test_data)[1]:
        for post in subject:
            post_lengths.append(len(post.split()))

    for subject in get_data(train_data)[1]:
        for post in subject:
            post_lengths.append(len(post.split()))

    post_lengths.sort()  # sort from 0 to highest number
    print("Some analysis on the post length in our test+train data:\n")
    print("How many strings are there in this list?")
    print(len(post_lengths))

    print("\n-----------------------------\nMetrics for the entire data:")
    metrics_0_0 = get_metrics(post_lengths, 0, 0)
    print(f"No. of posts:\t {metrics_0_0[0]} \nMax. Length:\t {metrics_0_0[1]} "
          f"\nMin. Length:\t {metrics_0_0[2]} \nMean. Length:\t {metrics_0_0[3]} ")

    print("\n-----------------------------\nAfter cutting off 2% off the data:")
    metrics_2_2 = get_metrics(post_lengths, 0.02, 0.02)
    print(f"No. of posts:\t {metrics_2_2[0]} \nMax. Length:\t {metrics_2_2[1]} "
          f"\nMin. Length:\t {metrics_2_2[2]} \nMean. Length:\t {metrics_2_2[3]} ")

    print("\n-----------------------------\nAfter cutting off 5% off the data:")
    metrics_5_5 = get_metrics(post_lengths, 0.05, 0.05)
    print(f"No. of posts:\t {metrics_5_5[0]} \nMax. Length:\t {metrics_5_5[1]} "
          f"\nMin. Length:\t {metrics_5_5[2]} \nMean. Length:\t {metrics_5_5[3]} ")

    print("\n-----------------------------\nAfter cutting off 10% off the data:")
    metrics_10_10 = get_metrics(post_lengths, 0.1, 0.1)
    print(f"No. of posts:\t {metrics_10_10[0]} \nMax. Length:\t {metrics_10_10[1]} "
          f"\nMin. Length:\t {metrics_10_10[2]} \nMean. Length:\t {metrics_10_10[3]} ")

    print("\n-----------------------------\nAfter cutting off 25% off the data:")
    metrics_25_25 = get_metrics(post_lengths, 0.25, 0.25)
    print(f"No. of posts:\t {metrics_25_25[0]} \nMax. Length:\t {metrics_25_25[1]} "
          f"\nMin. Length:\t {metrics_25_25[2]} \nMean. Length:\t {metrics_25_25[3]} ")

    print("\n-----------------------------\n-----------------------------\n")
    print("How many of the strings have the length 0? (in percent)")
    percent_zero = percentage_number(post_lengths, 0)
    print(percent_zero)
    print("How many of the strings have the length 8167? (in percent)")
    percent_big = percentage_number(post_lengths, 8167)
    print(percent_big)
    print("After cutting all the 0s and 5% at the top:")
    print("\n-----------------------------\nAfter cutting all the 0s and 5% at the top:")
    metrics_5_nozero = get_metrics(post_lengths, 0.05, percent_zero/100)
    print(f"No. of posts:\t {metrics_5_nozero[0]} \nMax. Length:\t {metrics_5_nozero[1]} "
          f"\nMin. Length:\t {metrics_5_nozero[2]} \nMean. Length:\t {metrics_5_nozero[3]} ")

    print("\n-----------------------------\nAfter cutting all the 0s and 15% at the top:")
    metrics_10_nozero = get_metrics(post_lengths, 0.15, percent_zero / 100)
    print(f"No. of posts:\t {metrics_10_nozero[0]} \nMax. Length:\t {metrics_10_nozero[1]} "
          f"\nMin. Length:\t {metrics_10_nozero[2]} \nMean. Length:\t {metrics_10_nozero[3]} ")

    print("\n-----------------------------\nAfter cutting all the 0s and 10% at the top and bottom:")
    metrics_10_nozero = get_metrics(post_lengths, 0.1, (percent_zero / 100)+0.1)
    print(f"No. of posts:\t {metrics_10_nozero[0]} \nMax. Length:\t {metrics_10_nozero[1]} "
          f"\nMin. Length:\t {metrics_10_nozero[2]} \nMean. Length:\t {metrics_10_nozero[3]}

if __name__ == '__main__':
    main()
