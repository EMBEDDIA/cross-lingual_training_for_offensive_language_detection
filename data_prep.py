from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import json
import os
import copy
import math


def frenk_stats(datapath):
    dataset = pd.read_csv(datapath, sep="\t")
    print(len(dataset))
    not_relevant = dataset.loc[dataset['label'] == 'Not relevant']
    print(len(not_relevant))


def transform_frenk_dataset(data_path, output_file):

    comments = []
    label = []

    with open(data_path, 'r') as f:
        json_object = json.loads(f.read())

    for doc in json_object:
        for comment in doc['comments']:
            comments.append(comment['text'])
            label.append(comment['type_mode'])

    dataset_dict = {'data': comments, 'label': label}
    dataset_df = pd.DataFrame(data=dataset_dict)
    dataset_df.to_csv(output_file, sep="\t", index=False)


def dataset_statistics(datasets, label_row, not_label):
    frames = []
    for dataset in datasets:
        df_dataset = pd.read_csv(dataset, sep="\t")
        frames.append(df_dataset)

    full_dataset = pd.concat(frames)

    not_count = 0
    off_count = 0
    for index, row in full_dataset.iterrows():
        if row[label_row] == not_label:
            not_count += 1
        else:
            off_count += 1

    print("Offensive: %d" % off_count)
    print("Not offensive: %d" % not_count)
    print("Length of the dataset: %d" % len(full_dataset))


def concatenate_datasets(datasets, output_file):
    frames = []
    for dataset in datasets:
        df_dataset = pd.read_csv(dataset, sep="\t")
        frames.append(df_dataset)

    full_dataset = pd.concat(frames)
    full_dataset.to_csv(output_file, sep="\t", index=False)


def from_txt_to_tsv(dataset_path, output_path):
    f_txt = open(dataset_path, "r", encoding='utf-8')
    #f_tsv = open(output_path, "a")

    labels = []
    data = []
    for line in f_txt:
        print(line)
        split_line = line.split()
        labels.append(split_line[-2])
        split_line = split_line[:-2]
        cleaned_line = " "
        cleaned_line = cleaned_line.join(split_line)
        data.append(cleaned_line)
    f_txt.close()
    print(len(data))
    print(len(labels))
    for i in range (10):
        print(data[i])
    print(labels)

    data_dict = {'data': data, 'labels': labels}
    df_cleaned_data = pd.DataFrame(data=data_dict)
    df_cleaned_data.to_csv(output_path, sep="\t", index=False)
    #lines = []
    #for instance, label in zip(data, labels):
    #    lines.append(instance + "\t" + label + "\n")

    #f_tsv.writelines(lines)
    #f_tsv.close()


def train_dev_test_split(dataset_path, output_dir, language, label_column, min_label, min_class_distr):
    df_data = pd.read_csv(dataset_path, sep="\t")
    df_data = df_data.sample(frac=1)
    #df_data_copy = copy.deepcopy(df_data)

    min_label_absolute = math.floor(7839 * min_class_distr)
    maj_label_absolute = math.ceil(7839 * (1 - min_class_distr))

    data_maj = df_data.loc[df_data[label_column] != min_label]
    data_min = df_data.loc[df_data[label_column] == min_label]
    print(data_maj)
    print(data_min)

    data_maj = data_maj.sample(frac=1)
    data_min = data_min.sample(frac=1)
    data_maj = data_maj.iloc[:maj_label_absolute]
    data_min = data_min.iloc[:min_label_absolute]
    print(str(len(data_maj)+len(data_min)))

    frames = [data_maj, data_min]
    resized_data = pd.concat(frames)
    print(resized_data)
    print(len(resized_data))
    resized_data = resized_data.sample(frac=1)

    data_train, data_test = train_test_split(resized_data, test_size=0.1, random_state=42)
    data_train, data_eval = train_test_split(data_train, test_size=0.1, random_state=42)

    test_split_name = language + "_internal_test.tsv"
    test_split_path = os.path.join(output_dir, test_split_name)
    data_test.to_csv(test_split_path, sep="\t", index=False)

    eval_split_name = language + "_val.tsv"
    eval_split_path = os.path.join(output_dir, eval_split_name)
    data_eval.to_csv(eval_split_path, sep="\t", index=False)

    train_split_name = language + "_train.tsv"
    train_split_path = os.path.join(output_dir, train_split_name)
    data_train.to_csv(train_split_path, sep="\t", index=False)


def train_dev_test_split_no_resample(dataset_path, output_dir, language, data_column, label_column):
    df_data = pd.read_csv(dataset_path, sep="\t")
    data = df_data[data_column].tolist()
    labels = df_data[label_column].tolist()

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    data_train, data_eval, labels_train, labels_eval = train_test_split(data_train, labels_train, test_size=0.1,
                                                                        random_state=42)

    test_split_name = language + "_internal_test.tsv"
    test_split_path = os.path.join(output_dir, test_split_name)
    test_dict = {'data': data_test, 'label': labels_test}
    test_dataset = pd.DataFrame(data=test_dict)
    test_dataset.to_csv(test_split_path, sep="\t", index=False)

    eval_split_name = language + "_val.tsv"
    eval_split_path = os.path.join(output_dir, eval_split_name)
    eval_dict = {'data': data_eval, 'label': labels_eval}
    eval_dataset = pd.DataFrame(data=eval_dict)
    eval_dataset.to_csv(eval_split_path, sep="\t", index=False)

    train_split_name = language + "_train.tsv"
    train_split_path = os.path.join(output_dir, train_split_name)
    train_dict = {'data': data_train, 'label': labels_train}
    train_dataset = pd.DataFrame(data=train_dict)
    train_dataset.to_csv(train_split_path, sep="\t", index=False)


def train_dev_split_no_resample(dataset_path, output_dir, language, data_column, label_column, dev_size=0.2):
    df_data = pd.read_csv(dataset_path, sep="\t")
    data = df_data[data_column].tolist()
    labels = df_data[label_column].tolist()

    data_train, data_eval, labels_train, labels_eval = train_test_split(data, labels, test_size=dev_size, random_state=42)

    eval_split_name = language + "_val.tsv"
    eval_split_path = os.path.join(output_dir, eval_split_name)
    eval_dict = {'data': data_eval, 'label': labels_eval}
    eval_dataset = pd.DataFrame(data=eval_dict)
    eval_dataset.to_csv(eval_split_path, sep="\t", index=False)

    train_split_name = language + "_train.tsv"
    train_split_path = os.path.join(output_dir, train_split_name)
    train_dict = {'data': data_train, 'label': labels_train}
    train_dataset = pd.DataFrame(data=train_dict)
    train_dataset.to_csv(train_split_path, sep="\t", index=False)


def resize_frenk_dataset(migrant_data, lgbt_data, output_dir, language, label_column, migrant_min_label,
                         migrant_min_class_distr, lgbt_min_label, lgbt_min_class_distr):
    df_migrant_data = pd.read_csv(migrant_data, sep="\t")
    df_migrant_data = df_migrant_data.sample(frac=1)

    df_lgbt_data = pd.read_csv(lgbt_data, sep="\t")
    df_lgbt_data = df_lgbt_data.loc[df_lgbt_data[label_column] != 'Not relevant']
    df_lgbt_data = df_lgbt_data.sample(frac=1)

    migrant_data_fraction = round(len(df_migrant_data) / (len(df_lgbt_data) + len(df_migrant_data)), 2)
    print(migrant_data_fraction)
    lgbt_data_fraction = 1 - migrant_data_fraction
    print(lgbt_data_fraction)
    # df_data_copy = copy.deepcopy(df_data)

    migrant_min_label_absolute = math.floor((7839 * migrant_data_fraction) * migrant_min_class_distr)
    migrant_maj_label_absolute = math.ceil((7839 * migrant_data_fraction) * (1 - migrant_min_class_distr))

    migrant_data_maj = df_migrant_data.loc[df_migrant_data[label_column] != migrant_min_label]
    migrant_data_min = df_migrant_data.loc[df_migrant_data[label_column] == migrant_min_label]
    print(migrant_data_maj)
    print(migrant_data_min)

    migrant_data_maj = migrant_data_maj.sample(frac=1)
    migrant_data_min = migrant_data_min.sample(frac=1)
    migrant_data_maj = migrant_data_maj.iloc[:migrant_maj_label_absolute]
    migrant_data_min = migrant_data_min.iloc[:migrant_min_label_absolute]
    print(str(len(migrant_data_maj) + len(migrant_data_min)))

    frames = [migrant_data_maj, migrant_data_min]
    migrant_resized_data = pd.concat(frames)
    print(migrant_resized_data)
    print(len(migrant_resized_data))
    migrant_resized_data = migrant_resized_data.sample(frac=1)

    lgbt_min_label_absolute = math.floor((7839 * lgbt_data_fraction) * lgbt_min_class_distr)
    lgbt_maj_label_absolute = math.ceil((7839 * lgbt_data_fraction) * (1 - lgbt_min_class_distr))

    lgbt_data_maj = df_lgbt_data.loc[df_lgbt_data[label_column] != lgbt_min_label]
    lgbt_data_min = df_lgbt_data.loc[df_lgbt_data[label_column] == lgbt_min_label]
    print(lgbt_data_maj)
    print(lgbt_data_min)

    lgbt_data_maj = lgbt_data_maj.sample(frac=1)
    lgbt_data_min = lgbt_data_min.sample(frac=1)
    lgbt_data_maj = lgbt_data_maj.iloc[:lgbt_maj_label_absolute]
    lgbt_data_min = lgbt_data_min.iloc[:lgbt_min_label_absolute]
    print(str(len(lgbt_data_maj) + len(lgbt_data_min)))

    frames = [lgbt_data_maj, lgbt_data_min]
    lgbt_resized_data = pd.concat(frames)
    print(lgbt_resized_data)
    print(len(lgbt_resized_data))
    lgbt_resized_data = lgbt_resized_data.sample(frac=1)

    frames = [migrant_resized_data, lgbt_resized_data]
    full_resized_data = pd.concat(frames)
    full_resized_data = full_resized_data.sample(frac=1)
    print(full_resized_data)
    print(len(full_resized_data))

    data_train, data_test = train_test_split(full_resized_data, test_size=0.1, random_state=42)
    data_train, data_eval = train_test_split(data_train, test_size=0.1, random_state=42)

    test_split_name = language + "_internal_test.tsv"
    test_split_path = os.path.join(output_dir, test_split_name)
    data_test.to_csv(test_split_path, sep="\t", index=False)

    eval_split_name = language + "_val.tsv"
    eval_split_path = os.path.join(output_dir, eval_split_name)
    data_eval.to_csv(eval_split_path, sep="\t", index=False)

    train_split_name = language + "_train.tsv"
    train_split_path = os.path.join(output_dir, train_split_name)
    data_train.to_csv(train_split_path, sep="\t", index=False)


def resize_two_datasets(data1, data2, output_dir, language, label_column, data1_min_label,
                         data1_min_class_distr, data2_min_label, data2_min_class_distr):
    df_migrant_data = pd.read_csv(data1, sep="\t")
    df_migrant_data = df_migrant_data.sample(frac=1)

    df_lgbt_data = pd.read_csv(data2, sep="\t")
    df_lgbt_data = df_lgbt_data.loc[df_lgbt_data[label_column] != 'Not relevant']
    df_lgbt_data = df_lgbt_data.sample(frac=1)

    migrant_data_fraction = round(len(df_migrant_data) / (len(df_lgbt_data) + len(df_migrant_data)), 2)
    print(migrant_data_fraction)
    lgbt_data_fraction = 1 - migrant_data_fraction
    print(lgbt_data_fraction)
    # df_data_copy = copy.deepcopy(df_data)

    migrant_min_label_absolute = math.floor((7839 * migrant_data_fraction) * data1_min_class_distr)
    migrant_maj_label_absolute = math.ceil((7839 * migrant_data_fraction) * (1 - data1_min_class_distr))

    migrant_data_maj = df_migrant_data.loc[df_migrant_data[label_column] != data1_min_label]
    migrant_data_min = df_migrant_data.loc[df_migrant_data[label_column] == data1_min_label]
    print(migrant_data_maj)
    print(migrant_data_min)

    migrant_data_maj = migrant_data_maj.sample(frac=1)
    migrant_data_min = migrant_data_min.sample(frac=1)
    migrant_data_maj = migrant_data_maj.iloc[:migrant_maj_label_absolute]
    migrant_data_min = migrant_data_min.iloc[:migrant_min_label_absolute]
    print(str(len(migrant_data_maj) + len(migrant_data_min)))

    frames = [migrant_data_maj, migrant_data_min]
    migrant_resized_data = pd.concat(frames)
    print(migrant_resized_data)
    print(len(migrant_resized_data))
    migrant_resized_data = migrant_resized_data.sample(frac=1)

    lgbt_min_label_absolute = math.floor((7839 * lgbt_data_fraction) * data2_min_class_distr)
    lgbt_maj_label_absolute = math.ceil((7839 * lgbt_data_fraction) * (1 - data2_min_class_distr))

    lgbt_data_maj = df_lgbt_data.loc[df_lgbt_data[label_column] != data2_min_label]
    lgbt_data_min = df_lgbt_data.loc[df_lgbt_data[label_column] == data2_min_label]
    print(lgbt_data_maj)
    print(lgbt_data_min)

    lgbt_data_maj = lgbt_data_maj.sample(frac=1)
    lgbt_data_min = lgbt_data_min.sample(frac=1)
    lgbt_data_maj = lgbt_data_maj.iloc[:lgbt_maj_label_absolute]
    lgbt_data_min = lgbt_data_min.iloc[:lgbt_min_label_absolute]
    print(str(len(lgbt_data_maj) + len(lgbt_data_min)))

    frames = [lgbt_data_maj, lgbt_data_min]
    lgbt_resized_data = pd.concat(frames)
    print(lgbt_resized_data)
    print(len(lgbt_resized_data))
    lgbt_resized_data = lgbt_resized_data.sample(frac=1)

    frames = [migrant_resized_data, lgbt_resized_data]
    full_resized_data = pd.concat(frames)
    full_resized_data = full_resized_data.sample(frac=1)
    print(full_resized_data)
    print(len(full_resized_data))

    data_train, data_test = train_test_split(full_resized_data, test_size=0.1, random_state=42)
    data_train, data_eval = train_test_split(data_train, test_size=0.1, random_state=42)

    test_split_name = language + "_internal_test.tsv"
    test_split_path = os.path.join(output_dir, test_split_name)
    data_test.to_csv(test_split_path, sep="\t", index=False)

    eval_split_name = language + "_val.tsv"
    eval_split_path = os.path.join(output_dir, eval_split_name)
    data_eval.to_csv(eval_split_path, sep="\t", index=False)

    train_split_name = language + "_train.tsv"
    train_split_path = os.path.join(output_dir, train_split_name)
    data_train.to_csv(train_split_path, sep="\t", index=False)


def binarize_frenk_dataset(datapath, label_column, output_dir):
    dataset = pd.read_csv(datapath, sep="\t")
    for index, row in dataset.iterrows():
        if row[label_column] == "Acceptable speech":
            row[label_column] = "NOT"
        else:
            row[label_column] = "OFF"

    dataset.to_csv(output_dir, sep="\t", index=False)


def read_german_data(datapath):
    dataset = pd.read_csv(datapath, sep="\t", skipinitialspace=True, quotechar='"')
    print(len(dataset))


def calculate_baselines(datapath, label_column, maj_label):
    df_data = pd.read_csv(datapath, sep="\t")
    actual = df_data[label_column].tolist()
    predicted = np.full_like(actual, maj_label)
    print(predicted)

    print("Accuracy: %0.4f" % accuracy_score(actual, predicted))
    print("Recall: %0.4f" % recall_score(actual, predicted, average="macro"))
    print("Precision: %0.4f" % precision_score(actual, predicted, average="macro"))
    print("F1: %0.4f" % f1_score(actual, predicted, average="macro"))

#labels = df_data[label_column].tolist()

#    not_labels = []
##    off_labels = []
 #   for index, row in labels.iterrow():
 #       if row[label_column] == not_label:
 #           not_labels.append(row[label_column])
 #       else:
 #           off_labels.append(row[label_column])

    #sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.1)
    #for train_index, test_index in sss.split(data, labels):
    #    print("TRAIN:", train_index, "TEST:", test_index)
    #    data_train, data_test = data[train_index], data[test_index]
    #    labels_train, labels_test = labels[train_index], labels[test_index]

    #test_split_name = language + "_internal_test.tsv"
    #test_split_path = os.path.join(output_dir, test_split_name)
    #test_dict = {'data': data_test, 'label': labels_test}
    #test_split = pd.DataFrame(test_dict)
    #test_split.to_csv(test_split_path, sep="\t", index=False)

    #sss2 = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.1)
    #for train_index, test_index in sss2.split(data_train, labels_train):
    #    print("TRAIN:", train_index, "TEST:", test_index)
    #    data_train, data_val = data_train[train_index], data_train[test_index]
    #    labels_train, labels_val = labels_train[train_index], labels_train[test_index]

    #val_split_name = language + "_val.tsv"
    #val_split_path = os.path.join(output_dir, val_split_name)
    #val_dict = {'data': data_val, 'label': labels_val}
    #val_split = pd.DataFrame(val_dict)
    #val_split.to_csv(val_split_path, sep="\t", index=False)

    #train_split_name = language + "_val.tsv"
    #train_split_path = os.path.join(output_dir, train_split_name)
    #train_dict = {'data': data_train, 'label': labels_train}
    #train_split = pd.DataFrame(train_dict)
    #train_split.to_csv(train_split_path, sep="\t", index=False)


if __name__ == "__main__":
    #train_dev_split_no_resample("/home/andrazp/cross_lingual_hate_speech/cross_lingual_hate_speech/data/english/raw_data/olid-training-v1.0.tsv",
    #                        "/home/andrazp/cross_lingual_hate_speech/cross_lingual_hate_speech/data/english",
    #                        "english",
    #                        "tweet",
    #                        "subtask_a")
    #transform_frenk_dataset("../data/slovenian/raw_data/lgbt_homofobija_final.json",
    #                        "../data/slovenian/raw_data/lgbt_homofobija_final.tsv")
    #concatenate_datasets(["../data/slovenian/raw_data/begunci_islamofobija_final.tsv",
    #                      "../data/slovenian/raw_data/lgbt_homofobija_final.tsv"],
    #                    "../data/slovenian/raw_data/concatenated_slovenian.tsv")
    #train_dev_test_split("../data/croatian/raw_data/concatenated_croatian.tsv",
    #                     "../data/croatian",
    #                     "cro",
    #                     "label",
    #                     1,
    #                    0.22)
    #resize_two_datasets("../data/german/raw_data/germeval2018.training_corrected.tsv",
    #                     "../data/german/raw_data/germeval2019.training.tsv",
    #                     "../data/german/",
    #                     "german",
    #                     "labels",
    #                     "OFFENSE",
    #                     0.34,
    #                     "OFFENSE",
    #                     "OFFENSE",
    #                    0.32)
    #binarize_frenk_dataset("../data/slovenian/slo_val.tsv",
    #                       "label",
    #                       "../data/slovenian/slo_val_binarized.tsv")
    #from_txt_to_tsv("../data/german/raw_data/germeval2019.training_subtask1_2_korrigiert.txt",
    #                "../data/german/raw_data/germeval2019.training.tsv")
    calculate_baselines("../data/english/english_test.tsv",
                        "label",
                        "NOT")