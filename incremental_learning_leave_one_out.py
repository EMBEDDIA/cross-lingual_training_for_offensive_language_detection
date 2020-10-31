import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np
import argparse
import random
import math
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import matplotlib
import matplotlib.pyplot as plt

def incremental_learning():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_language",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--no_finetuning",
                        action='store_true')
    parser.add_argument("--model_type",
                        required=True,
                        type=str)

    # parser.add_argument("--eval_split",
    #                    default=0.1,
    #                    type=float)
    # parser.add_argument("--test_split",
    #                    default=0.1,
    #                    type=float)
    parser.add_argument("--max_len",
                        default=256,
                        type=int)
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=4,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float)
    parser.add_argument("--random_seed",
                        default=42,
                        type=int)


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        print("Error: the specified directory is not empty.")
        print("Please specify an empty directory in order to avoid rewriting "
              "important data.")
        sys.exit()

    language = {"arabic": {"data": "data", "label": "label", "train_data_path": "../data/arabic/arabic_train.tsv",
                           "eval_data_path": "../data/arabic/arabic_val.tsv", "test_data_path": "../data/arabic/arabic_internal_test.tsv",
                           "offensive_label": "OFF"},
                "german": {"data": "data", "label": "labels", "train_data_path": "../data/german/german_train.tsv",
                           "eval_data_path": "../data/german/german_val.tsv", "test_data_path": "../data/german/german_internal_test.tsv",
                           "offensive_label": "OFFENSE"},
                "slovenian": {"data": "data", "label": "label", "train_data_path": "../data/slovenian/slo_train_binarized.tsv",
                           "eval_data_path": "../data/slovenian/slo_val_binarized.tsv", "test_data_path": "../data/slovenian/slo_internal_test_binarized.tsv",
                              "offensive_label": "OFF"},
                "croatian": {"data": "text_a", "label": "label", "train_data_path": "../data/croatian/cro_train.tsv",
                           "eval_data_path": "../data/croatian/cro_val.tsv", "test_data_path": "../data/croatian/cro_internal_test.tsv",
                             "offensive_label": 1}
                }

    if args.train_language not in language:
        print("Set the language as one of the following: arabic german croatian slovenian")
        sys.exit()

    print("Setting the random seed...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = os.path.join(args.output_dir, "log")

    results_f1 = []
    iteration = []

    for i in range(0, 11):
        print("Training model on the " + str(i * 10) + "% of data...")
        # print("Reading data...")
        non_target_languages_data = []
        for lang, lan_dict in language.items():
            if lang != args.train_language:
                df_data = pd.read_csv(language[lang]["train_data_path"], sep="\t")
                df_data = consolidate_dataset(df_data, language[lang]["data"], language[lang]["label"],
                                              language[lang]["offensive_label"])
                non_target_languages_data.append(df_data)

        if i != 0:
            df_target_data = pd.read_csv(language[args.train_language]["train_data_path"], sep="\t")
            df_target_data = consolidate_dataset(df_target_data, language[args.train_language]["data"],
                                             language[args.train_language]["label"],
                                             language[args.train_language]["offensive_label"])

            df_target_data = df_target_data.iloc[:math.floor((len(df_target_data) * i * 0.1))]
            # print(len(df_target_data))
            non_target_languages_data.append(df_target_data)

        df_data = pd.concat(non_target_languages_data)
        df_data = df_data.sample(frac=1, random_state=args.random_seed)
        # print(len(df_data))
        train_data = df_data['data'].tolist()
        train_labels = df_data['labels'].tolist()
        label_set = list(set(train_labels))
        # print(label_set)
        label_set = sorted(label_set)
        train_labels = encode_labels(train_labels, label_set)

        df_eval_data = pd.read_csv(language[args.train_language]["eval_data_path"], sep="\t")
        df_eval_data = consolidate_dataset(df_eval_data, language[args.train_language]["data"],
                                             language[args.train_language]["label"],
                                             language[args.train_language]["offensive_label"])
        eval_data = df_eval_data['data'].tolist()
        eval_labels = df_eval_data['labels'].tolist()
        eval_labels = encode_labels(eval_labels, label_set)

        df_test_data = pd.read_csv(language[args.train_language]["test_data_path"], sep="\t")
        df_test_data = consolidate_dataset(df_test_data, language[args.train_language]["data"],
                                             language[args.train_language]["label"],
                                             language[args.train_language]["offensive_label"])
        test_data = df_test_data['data'].tolist()
        test_labels = df_test_data['labels'].tolist()
        test_labels = encode_labels(test_labels, label_set)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.no_finetuning:
            print("no finetuning")
            if args.model_type == "shebert":
                tokenizer = BertTokenizer.from_pretrained("./models/crosloengual-bert-pytorch/vocab.txt")
                config = BertConfig.from_pretrained("./models/crosloengual-bert-pytorch/bert_config.json",
                                                    num_labels=len(label_set))
                model = BertForSequenceClassification.from_pretrained(
                    "./models/crosloengual-bert-pytorch/pytorch_model.bin", config=config)
            elif args.model_type == "mbert":
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
                model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                                      num_labels=len(label_set))
            else:
                print("Wrong argument value for model type")
                sys.exit()
        else:
            if args.model_type == "shebert":
                tokenizer = BertTokenizer.from_pretrained("./models/shebert_en_finetune/vocab.txt")
                config = BertConfig.from_pretrained("./models/shebert_en_finetune/config.json",
                                                    num_labels=len(label_set))
                model = BertForSequenceClassification.from_pretrained("./models/shebert_en_finetune/pytorch_model.bin",
                                                                      config=config)
            elif args.model_type == "mbert":
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
                config = BertConfig.from_pretrained("./models/mbert_en_finetune/config.json",
                                                    num_labels=len(label_set))
                model = BertForSequenceClassification.from_pretrained("./models/mbert_en_finetune/pytorch_model.bin",
                                                                      config=config)
            else:
                print("Wrong argument value for model type")
                sys.exit()

        output_subdir = os.path.join(args.output_dir, str(i * 10) + "%")
        print(output_subdir)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        #print(output_dir)
        #print(log_file)
        #print(log_path)

        #test_data = data[(floor(len(data) * i * 0.1)):(floor(len(data) * (i + 1) * 0.1))]
        #test_labels = labels[floor((len(labels) * i * 0.1)):floor((len(labels) * (i + 1) * 0.1))]

        #print("Train data: %d" % len(train_data))
        #print("Train labels: %d" % len(train_labels))
        #print("Test data: %d" % len(test_data))
        #print("Test labels: %d" % len(test_labels))
        #print("Test labels: %d" % len(test_labels))

        # print("Train data length: %d" % len(train_data))
        # print("Train label:")
        # print(train_labels[0])
        # print("Train data:")
        # print(train_data[0])
        train_dataloader = prepare_labeled_data(train_data, train_labels, tokenizer, args.max_len, args.batch_size)
        eval_dataloader = prepare_labeled_data(eval_data, eval_labels, tokenizer, args.max_len, args.batch_size)
        test_dataloader = prepare_labeled_data(test_data, test_labels, tokenizer, args.max_len, args.batch_size)
        _, eval_metrics = bert_train(model, device, train_dataloader, eval_dataloader, test_dataloader, eval_data, test_data, label_set, output_subdir, args.num_epochs,
                           args.warmup_proportion, args.weight_decay, args.learning_rate, args.adam_epsilon,
                           save_best=True)

        #plotting metrics for eval data
        output_eval_plot_file = os.path.join(output_subdir, "eval_f1_plot.png")
        x = np.arange(1, args.num_epochs+1, 1)
        eval_f1_scores = []
        for dict in eval_metrics:
            eval_f1_scores.append(dict['f1'])
        x_label = "Epoch"
        y_label = "macro F1"
        plot_f1(x, eval_f1_scores, x_label, y_label, output_eval_plot_file)

        print("Testing the trained model...")
        metrics, predictions = bert_evaluate(model, test_dataloader, device)
        with open(log_path, 'a') as f:
            f.write("Results for " + str(i * 10) + "% of training data:\n")
            f.write("Acc: " + str(metrics['accuracy']) + "\n")
            f.write("Recall: " + str(metrics['recall']) + "\n")
            f.write("Precision: " + str(metrics['precision']) + "\n")
            f.write("F1: " + str(metrics['f1']) + "\n")
            f.write("\n")

        results_f1.append(metrics['f1'])
        iteration.append(i * 0.1)

    output_plot_file = os.path.join(args.output_dir, "plot.png")
    x_label = '% of data'
    y_label = 'macro F1'
    plot_f1(iteration, results_f1, x_label, y_label, output_plot_file)
    print("Done.")


def plot_f1(iterations, results, x_label, y_label, output_file):
    fig, ax = plt.subplots()
    ax.plot(iterations, results)

    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid()

    fig.savefig(output_file)


def bert_train(model, device, train_dataloader, eval_dataloader, test_dataloader, eval_data, test_data,
               labels_set, output_dir, num_epochs, warmup_proportion, weight_decay,
               learning_rate, adam_epsilon, save_best=False):
    """Training loop for bert fine-tuning. Save best works with F1 only currently."""

    t_total = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    train_iterator = trange(int(num_epochs), desc="Epoch")
    model.to(device)
    tr_loss_track = []
    eval_metric_track = []
    output_filename = os.path.join(output_dir, 'pytorch_model.bin')
    f1 = float('-inf')
    predictions_file_name_counter = 1

    for _ in train_iterator:
        eval_predictions_filename = os.path.join(output_dir,
                                                 "eval_predictions_epoch_" + str(predictions_file_name_counter))
        test_predictions_filename = os.path.join(output_dir,
                                                 "test_predictions_epoch_" + str(predictions_file_name_counter))
        model.train()
        model.zero_grad()
        tr_loss = 0
        nr_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            tr_loss = 0
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=input_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nr_batches += 1
            model.zero_grad()

        print("Evaluating the model on the evaluation split...")
        metrics, eval_predictions = bert_evaluate(model, eval_dataloader, device)
        eval_metric_track.append(metrics)
        write_model_outputs(eval_predictions, eval_predictions_filename, eval_data, labels_set)
        if save_best:
            if f1 < metrics['f1']:
                model.save_pretrained(output_dir)
                torch.save(model.state_dict(), output_filename)
                print("The new value of f1 score of " + str(metrics['f1']) + " is higher then the old value of " +
                      str(f1) + ".")
                print("Saving the new model...")
                f1 = metrics['f1']
            else:
                print("The new value of f1 score of " + str(metrics['f1']) + " is not higher then the old value of " +
                      str(f1) + ".")
        test_predictions = bert_predict_during_train(model, test_dataloader, device)
        write_model_outputs(test_predictions, test_predictions_filename, test_data, labels_set)
        predictions_file_name_counter += 1

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)

    if not save_best:
        model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, eval_metric_track


def bert_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.eval()
    predictions = []
    true_labels = []
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        #loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        # print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label, logit in zip(label_ids, logits):
            true_labels.append(label)
            predictions.append(np.argmax(logit))

    #print(predictions)
    #print(true_labels)
    metrics = get_metrics(true_labels, predictions)
    return metrics, predictions


def bert_predict_during_train(model, dataloader, device):
    """Outputing predictions from a trained model."""
    model.eval()
    predictions = []
    data_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, _ = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        # print(type(logits))
        logits = logits.to('cpu').numpy()
        for l in logits:
            predictions.append(np.argmax(l))

    return predictions

    #print(predictions)
    #f = open("results.txt", 'a')
    #for p in predictions:
    #    f.write(labeled_set[int(p)] + "\n")


def prepare_labeled_data(data, labels, tokenizer, max_len, batch_size):
    for i, sentence in enumerate(data):
        if isinstance(sentence, float):
            data[i] = " "

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    # print("Example of tokenized sentence:")
    # print(tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]
    # print("Printing encoded sentences:")
    # print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_data(data, tokenizer, max_len, batch_size):
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    # print("Example of tokenized sentence:")
    # print(tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]
    # print("Printing encoded sentences:")
    # print(input_ids[0])
    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    sampler = RandomSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def encode_labels(labels, labels_set):
    """Maps each label to a unique index.
    :param labels: (list of strings) labels of every instance in the dataset
    :param labels_set: (list of strings) set of labels that appear in the dataset
    :return (list of int) encoded labels
    """
    encoded_labels = []
    for label in labels:
        encoded_labels.append(labels_set.index(label))
    return encoded_labels


def get_metrics(actual, predicted):
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'recall': recall_score(actual, predicted, average="macro"),
               'precision': precision_score(actual, predicted, average="macro"),
               'f1': f1_score(actual, predicted, average="macro")}

    return metrics


def write_model_outputs(predictions, output_file, original_data, labels_set):
    f = open(output_file, 'a')
    # print(predictions)
    for p, data in zip(predictions, original_data):
        f.write(str(labels_set[int(p)]) + "\t")
        f.write(data + "\n")
    f.close()


def consolidate_dataset(dataset, data_column, label_column, offensive_label):
    """Turns all the datasets into a unified format:
    data columns - renamed to 'data'
    label columns - renamed to 'label'
    all other columns dropped
    labels: NOT for not offensive posts and OFF for offensive posts


    dataset - dataset in dataframe form
    labels_column - name of the column containing labels
    offensive_label - offensive label
    """

    labels = dataset[label_column].tolist()
    cons_labels = []
    for label in labels:
        if label == offensive_label:
            cons_labels.append("OFF")
        else:
            cons_labels.append("NOT")

    data = dataset[data_column].tolist()
    new_dataset = {'data': data, 'labels': cons_labels}
    new_dataset = pd.DataFrame(data=new_dataset)

    return new_dataset


if __name__ == "__main__":
    incremental_learning()
