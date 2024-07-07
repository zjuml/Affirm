for len in 96 192 336 720
do
  python -u Forecasting/Affirm_Forecasting.py \
  --root_path datasets/ETT-small \
  --pred_len $len \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 1 \
  --batch_size 512 \
  --dropout 0.5 \
  --patch_size 32 \
  --train_epochs 20 \
  --pretrain_epochs 10 \
  --ASB False
done