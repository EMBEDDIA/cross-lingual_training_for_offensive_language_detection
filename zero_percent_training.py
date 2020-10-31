import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import argparse
import os
import pandas as pd
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences


def zero_percent():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data_path",
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
    parser.add_argument("--offensive_label",
                        required=True,
                        type=str)

    parser.add_argument("--tokenizer_file",
                        type=str)
    parser.add_argument("--config_file",
                        type=str,
                        required=True)
    parser.add_argument("--model_file",
                        type=str,
                        required=True)

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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    log_path = os.path.join(args.output_dir, "log")

    print("Reading data...")
    df_test_data = pd.read_csv(args.test_data_path, sep="\t")
    df_test_data = consolidate_dataset_modified(df_test_data, args.data_column, args.label_column, args.offensive_label)
    test_data = df_test_data["data"].tolist()
    test_labels = df_test_data["labels"].tolist()
    print(test_labels)

    label_set = sorted(list(set(df_test_data["labels"].values)))
    test_labels = encode_labels(test_labels, label_set)
    print(test_labels)

    print("loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.tokenizer_file is not None:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    config = BertConfig.from_pretrained(args.config_file, num_labels=len(label_set))
    model = BertForSequenceClassification.from_pretrained(args.model_file, config=config)

    print("Evaluating on the test set...")
    test_dataloader = prepare_labeled_data(test_data, test_labels, tokenizer, args.max_len, args.batch_size)
    metrics = bert_evaluate(model, test_dataloader, device)

    with open(log_path, 'a') as f:
        f.write("Acc: " + str(metrics['accuracy']) + "\n")
        f.write("F1: " + str(metrics['f1']) + "\n")

    print("Done.")


def consolidate_dataset_modified(dataset, data_column, label_column, offensive_label):
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
        if str(label) == offensive_label:
            cons_labels.append("OFF")
        else:
            cons_labels.append("NOT")

    data = dataset[data_column].tolist()
    new_dataset = {'data': data, 'labels': cons_labels}
    new_dataset = pd.DataFrame(data=new_dataset)

    return new_dataset


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


def get_metrics(actual, predicted):
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'recall': recall_score(actual, predicted, average="macro"),
               'precision': precision_score(actual, predicted, average="macro"),
               'f1': f1_score(actual, predicted, average="macro")}

    return metrics


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


if __name__ == "__main__":
    zero_percent()