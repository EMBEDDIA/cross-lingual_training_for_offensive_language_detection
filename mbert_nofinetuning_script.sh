python incremental_learning_no_finetuning.py \
        --train_data_path ../data/arabic/arabic_train.tsv \
        --test_data_path ../data/arabic/arabic_train.tsv \
        --eval_data_path ../data/arabic/arabic_train.tsv \
        --output_dir ./models/no_finetuning/mbert_arabic1 \
        --data_column data \
        --label_column label \
        --random_seed 42

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/slovenian/slo_train_binarized.tsv \
        --test_data_path ../data/slovenian/slo_internal_test_binarized.tsv \
        --eval_data_path ../data/slovenian/slo_val_binarized.tsv \
        --output_dir ../models/no_finetuning/mbert_slovenian1 \
        --data_column data \
        --label_column label \
        --random_seed 42

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/croatian/cro_train.tsv \
        --test_data_path ../data/croatian/cro_internal_tes.tsv \
        --eval_data_path ../data/croatian/cro_val.tsv \
        --output_dir ../models/no_finetuning/mbert_croatian1 \
        --data_column text_a \
        --label_column label \
        --random_seed 42

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/arabic/arabic_train.tsv \
        --test_data_path ../data/arabic/arabic_train.tsv \
        --eval_data_path ../data/arabic/arabic_train.tsv \
        --output_dir ./models/no_finetuning/mbert_arabic2 \
        --data_column data \
        --label_column label \
        --random_seed 84

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/slovenian/slo_train_binarized.tsv \
        --test_data_path ../data/slovenian/slo_internal_test_binarized.tsv \
        --eval_data_path ../data/slovenian/slo_val_binarized.tsv \
        --output_dir ../models/no_finetuning/mbert_slovenian2 \
        --data_column data \
        --label_column label \
        --random_seed 84

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/croatian/cro_train.tsv \
        --test_data_path ../data/croatian/cro_internal_tes.tsv \
        --eval_data_path ../data/croatian/cro_val.tsv \
        --output_dir ../models/no_finetuning/mbert_croatian2 \
        --data_column text_a \
        --label_column label \
        --random_seed 84

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/arabic/arabic_train.tsv \
        --test_data_path ../data/arabic/arabic_train.tsv \
        --eval_data_path ../data/arabic/arabic_train.tsv \
        --output_dir ./models/no_finetuning/mbert_arabic3 \
        --data_column data \
        --label_column label \
        --random_seed 126

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/slovenian/slo_train_binarized.tsv \
        --test_data_path ../data/slovenian/slo_internal_test_binarized.tsv \
        --eval_data_path ../data/slovenian/slo_val_binarized.tsv \
        --output_dir ../models/no_finetuning/mbert_slovenian3 \
        --data_column data \
        --label_column label \
        --random_seed 126

python incremental_learning_no_finetuning.py \
        --train_data_path ../data/croatian/cro_train.tsv \
        --test_data_path ../data/croatian/cro_internal_tes.tsv \
        --eval_data_path ../data/croatian/cro_val.tsv \
        --output_dir ../models/no_finetuning/mbert_croatian3 \
        --data_column text_a \
        --label_column label \
        --random_seed 126