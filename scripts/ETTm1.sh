seeds=(51 2024)
d_states=(16 32)
pred_lens=(96 192 336 720)
seq_len=512
for seed in "${seeds[@]}"
do
    for d_state in "${d_states[@]}"
    do
        for pred_len in "${pred_lens[@]}"
        do
          python -u Forecasting/Affirm_Forecasting.py \
          --root_path datasets/ETT-small \
          --pred_len $pred_len \
          --data ETTm1\
          --data_path ETTm1.csv \
          --seq_len $seq_len \
          --emb_dim 64 \
          --d_state $d_state \
          --d_conv_1 2 \
          --d_conv_2 4 \
          --depth 2 \
          --batch_size 32 \
          --dropout 0.5 \
          --patch_size 8 \
          --train_epochs 20 \
          --seed $seed \
          --enc_in 7 \
          --Mamba True \
          --AFFB True
        done
    done
done