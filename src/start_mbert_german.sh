python incremental_learning.py \
        --train_data_path ../data/german/german_train.tsv \
        --test_data_path ../data/german/german_internal_test.tsv \
        --eval_data_path ../data/german/german_val.tsv \
        --output_dir ./models/mbert_german1 \
        --config_file ../models/mbert_en_finetune/config.json \
        --model_file ../models/mbert_en_finetune/pytorch_model.bin \
        --data_column data \
        --label_column labels \
        --random_seed 42

python incremental_learning.py \
        --train_data_path ../data/german/german_train.tsv \
        --test_data_path ../data/german/german_internal_test.tsv \
        --eval_data_path ../data/german/german_val.tsv \
        --output_dir ./models/mbert_german2 \
        --config_file ../models/mbert_en_finetune/config.json \
        --model_file ../models/mbert_en_finetune/pytorch_model.bin \
        --data_column data \
        --label_column labels \
        --random_seed 84

python incremental_learning.py \
        --train_data_path ../data/german/german_train.tsv \
        --test_data_path ../data/german/german_internal_test.tsv \
        --eval_data_path ../data/german/german_val.tsv \
        --output_dir ./models/mbert_german3 \
        --config_file ../models/mbert_en_finetune/config.json \
        --model_file ../models/mbert_en_finetune/pytorch_model.bin \
        --data_column data \
        --label_column labels \
        --random_seed 126