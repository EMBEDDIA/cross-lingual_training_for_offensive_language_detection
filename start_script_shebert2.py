import subprocess


def run():
    subprocess.call(["python", "incremental_learning.py",
                     "--train_data_path", "../data/slovenian/slo_train_binarized.tsv",
                    "--test_data_path", "../data/slovenian/slo_internal_test_binarized.tsv",
                     "--eval_data_path", "../data/slovenian/slo_val_binarized.tsv",
                     "--output_dir", "../models/shebert_slovenian1",
                     "--data_column", "data",
                     "--label_column", "label",
                     "--tokenizer_file", "..models/shebert_en_finetune/vocab.txt",
                     "--config_file", "..models/shebert_en_finetune/config.json",
                     "--model_file", "..models/shebert_en_finetune/pytorch_model.bin",
                     "--random_seed", "42"])
    subprocess.call(["python", "incremental_learning.py",
                     "--train_data_path", "../data/slovenian/slo_train_binarized.tsv",
                    "--test_data_path", "../data/slovenian/slo_internal_test_binarized.tsv",
                     "--eval_data_path", "../data/slovenian/slo_val_binarized.tsv",
                     "--output_dir", "../models/shebert_slovenian2",
                     "--data_column", "data",
                     "--label_column", "label",
                     "--tokenizer_file", "..models/shebert_en_finetune/vocab.txt",
                     "--config_file", "..models/shebert_en_finetune/config.json",
                     "--model_file", "..models/shebert_en_finetune/pytorch_model.bin",
                     "--random_seed", "84"])
    subprocess.call(["python", "incremental_learning.py",
                     "--train_data_path", "../data/slovenian/slo_train_binarized.tsv",
                    "--test_data_path", "../data/slovenian/slo_internal_test_binarized.tsv",
                     "--eval_data_path", "../data/slovenian/slo_val_binarized.tsv",
                     "--output_dir", "../models/shebert_slovenian3",
                     "--data_column", "data",
                     "--label_column", "label",
                     "--tokenizer_file", "..models/shebert_en_finetune/vocab.txt",
                     "--config_file", "..models/shebert_en_finetune/config.json",
                     "--model_file", "..models/shebert_en_finetune/pytorch_model.bin",
                     "--random_seed", "126"])


if __name__ == "__main__":
    run()
