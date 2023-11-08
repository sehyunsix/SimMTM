export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2 \
    --model SimMTM \
    --data ETTm2 \
    --features M \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 64 \
    --n_heads 8 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --dropout 0.2 \
    --temperature 0.02 \
    --use_multi_gpu \
    --devices 0,1 \
    --train_epochs 50
