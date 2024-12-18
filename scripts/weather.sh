for len in 96 192 336 720
do
  python -u Forecasting/Affirm_Forecasting.py \
  --root_path datasets/weather \
  --pred_len $len \
  --data custom \
  --data_path weather.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 64 \
  --dropout 0.5 \
  --patch_size 64 \
  --train_epochs 50 \
  --pretrain_epochs 10 > mamba_${data_path}_train_epochs_${train_epochs}'_'len_$len.log
done