seq_len=336
for pred_len in 96 192 336 720
do
  python -u Forecasting/Affirm_Forecasting.py \
  --root_path datasets/ETT-small \
  --pred_len $pred_len \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len $seq_len \
  --emb_dim 32 \
  --depth 1 \
  --batch_size 32 \
  --dropout 0.5 \
  --patch_size 16 \
  --train_epochs 20 \
  --AFFB True
done