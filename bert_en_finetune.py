import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np
import argparse
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def semeval_task():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--test_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--eval_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--data_column",
                        required=True,
                        type=str)
    parser.add_argument("--label_column",
                        required=True,
                        type=str)

    parser.add_argument("--tokenizer_file",
                        type=str)
    parser.add_argument("--config_file",
                        type=str)
    parser.add_argument("--model_file",
                        type=str)

    #parser.add_argument("--eval_split",
    #                    default=0.1,
    #                    type=float)
    #parser.add_argument("--test_split",
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

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("Setting the random seed...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Reading data...")
    df_data = pd.read_csv(args.train_data_path, sep="\t")
    train_data = df_data[args.data_column].tolist()
    train_labels = df_data[args.label_column].tolist()
    #print(df_data['label'].tolist())
    label_set = sorted(list(set(df_data[args.label_column].values)))
    train_labels = encode_labels(train_labels, label_set)

    df_eval_data = pd.read_csv(args.eval_data_path, sep="\t")
    eval_data = df_eval_data[args.data_column].tolist()
    eval_labels = df_eval_data[args.label_column].tolist()
    eval_labels = encode_labels(eval_labels, label_set)
    #print(labels)
    #num_labels = len(set(labels))
    #acc = []
    #f1 = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.tokenizer_file is not None and args.config_file is not None and args.model_file is not None:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file)
        config = BertConfig.from_pretrained(args.config_file, num_labels=len(label_set))
        model = BertForSequenceClassification.from_pretrained(args.model_file, config=config)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_set))
    #train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels,
    #                                                                                test_size=args.eval_split,
    #                                                                                random_state=42)
    train_dataloader = prepare_labeled_data(train_data, train_labels, tokenizer, args.max_len, args.batch_size)
    eval_dataloader = prepare_labeled_data(eval_data, eval_labels, tokenizer, args.max_len, args.batch_size)

    _, _ = bert_train(model, device, train_dataloader, eval_dataloader, output_dir, args.num_epochs,
                      args.warmup_proportion, args.weight_decay,
               args.learning_rate, args.adam_epsilon, save_best=True)

    del train_dataloader
    del eval_dataloader
    del train_data
    del train_labels
    del eval_data
    del eval_labels

    df_test_data = pd.read_csv(args.test_data_path, sep="\t")
    test_data = df_test_data[args.data_column].tolist()
    test_labels = df_test_data[args.label_column].tolist()
    test_labels = encode_labels(test_labels, label_set)
    test_dataloader = prepare_labeled_data(test_data, test_labels, tokenizer, args.max_len, args.batch_size)
    metrics = bert_evaluate(model, test_dataloader, device)
    print("Acc: " + str(metrics['accuracy']) + "\n")
    print("F1: " + str(metrics['f1']) + "\n")

    #df_test_data = pd.read_csv(args.test_data_path, sep="\t")
    #test_data = df_test_data['data'].tolist()
    #test_dataloader = prepare_data(test_data, tokenizer, args.max_len, args.batch_size)
    #bert_predict(model, test_dataloader, device, label_set)
    #bert_predict2(test_dataloader, label_set, device)


def get_metrics(actual, predicted):
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'recall': recall_score(actual, predicted, average="macro"),
               'precision': precision_score(actual, predicted, average="macro"),
               'f1': f1_score(actual, predicted, average="macro")}

    return metrics


def bert_train(model, device, train_dataloader, eval_dataloader, output_dir, num_epochs, warmup_proportion, weight_decay,
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

    for _ in train_iterator:
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
        metrics = bert_evaluate(model, eval_dataloader, device)
        eval_metric_track.append(metrics)
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

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)

    if not save_best:
        model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, eval_metric_track


def bert_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.to(device)
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
        print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label, logit in zip(label_ids, logits):
            true_labels.append(label)
            predictions.append(np.argmax(logit))

    #print(predictions)
    #print(true_labels)
    metrics = get_metrics(true_labels, predictions)
    return metrics


def prepare_labeled_data(data, labels, tokenizer, max_len, batch_size):
    for i, sentence in enumerate(data):
        if isinstance(sentence, float):
            data[i] = " "

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print("Example of tokenized sentence:")
    print(tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
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
    for i, sentence in enumerate(data):
        if isinstance(sentence, float):
            data[i] = " "
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    print("Example of tokenized sentence:")
    print(tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]
    print("Printing encoded sentences:")
    print(input_ids[0])
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


#def bert_predict(model, dataloader, device, labeled_set):
#    """Outputing predictions from a trained model."""
#    model = BertForSequenceClassification.from_pretrained(output_dir)
#    model =
#    model.to(device)
#    model.eval()
#    predictions = []
#    data_iterator = tqdm(dataloader, desc="Iteration")
#    for step, batch in enumerate(data_iterator):
#        input_ids, input_mask = batch
#        input_ids = input_ids.to(device)
#        input_mask = input_mask.to(device)

#        with torch.no_grad():
#            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
#        logits = outputs[0]
#        print(type(logits))
#        logits = logits.to('cpu').numpy()

#        predictions.append(np.argmax(logits))

#    print(predictions)
#    f = open("results.txt", 'a')
#    for p in predictions:
#        f.write(labeled_set[int(p)] + "\n")


def bert_predict2(dataloader,labeled_set, device):
    """Outputing predictions from a trained model."""
    output_dir = "../models/bert_task2"
    model = BertForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()
    predictions = []
    data_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        print(type(logits))
        logits = logits.to('cpu').numpy()

        predictions.append(np.argmax(logits))

    print(predictions)
    f = open("results.txt", 'a')
    for p in predictions:
        f.write(labeled_set[int(p)] + "\n")



if __name__ == "__main__":
    semeval_task()

