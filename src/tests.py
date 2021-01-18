from incremental_learning_leave_one_out import consolidate_dataset
from extract_attentions import get_max_five_results, get_min_five_results
from incremental_learning import load_model
import pandas as pd
import pickle

def test_consolidate_dataset():
    df_data = pd.read_csv("../data/croatian/cro_train.tsv", sep="\t")
    df_data = consolidate_dataset(df_data, 'text_a', 'label', 1)
    print(df_data)
    print(df_data.loc[:10])


def test_get_max_five_results():
    numbers = [7, 64, 25, 28, 42, 58, 16, 83, 89, 96]
    results = get_max_five_results(numbers)
    print(results)
    numbers = [7, 64, 25, 28, 42, 58, 16, 83, 89, 89]
    results = get_max_five_results(numbers)
    print(results)


def test_get_min_five_results():
    numbers = [7, 64, 25, 28, 42, 58, 16, 83, 89, 96]
    results = get_min_five_results(numbers)
    print(results)
    numbers = [7, 64, 25, 28, 42, 58, 16, 83, 89, 28]
    results = get_min_five_results(numbers)
    print(results)


def test_attentions():
    easiest_attentions = pickle.load(open("/home/andrazp/cross_lingual_hate_speech/cross_lingual_hate_speech/models/slovenian/"
                             "easiest_attentions_2", 'rb'))
    medium_attentions = pickle.load(open("/home/andrazp/cross_lingual_hate_speech/cross_lingual_hate_speech/models/slovenian/"
                             "medium_attentions_2", 'rb'))
    hardest_attentions = pickle.load(open("/home/andrazp/cross_lingual_hate_speech/cross_lingual_hate_speech/models/slovenian/"
                             "hardest_attentions_2", 'rb'))
    print(len(easiest_attentions))
    print(len(medium_attentions))
    print(len(hardest_attentions))

    print("Analyzing easiest attentions")
    print("First batch:")
    print(len(easiest_attentions))
    print(easiest_attentions[0].size())
    print(type(easiest_attentions[0]))
    print("Last batch:")
    print(len(easiest_attentions[-1]))
    print(type(easiest_attentions[-1]))


def test_load_model():
    label_set = [0, 1]
    no_finetuning = True
    model_type = "mbert"
    load_model(no_finetuning, model_type, label_set)
    no_finetuning = False
    model_type = "mbert"
    load_model(no_finetuning, model_type, label_set)
    no_finetuning = True
    model_type = "shebert"
    load_model(no_finetuning, model_type, label_set)
    no_finetuning = False
    model_type = "shebert"
    load_model(no_finetuning, model_type, label_set)


if __name__ == "__main__":
    test_load_model()
