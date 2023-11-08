export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 16 \
        --d_ff 16 \
        --dropout 0.2 \
        --batch_size 16
done


