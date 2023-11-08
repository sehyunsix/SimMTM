export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model SimMTM \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 64 \
    --n_heads 8 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --temperature 0.02 \
    --train_epochs 50


