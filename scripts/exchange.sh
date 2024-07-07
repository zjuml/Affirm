for len in 96 192 336 720
do
  python -u Affirm_Forecasting.py \
  --root_path /mnt/sdb/hhj/TSLANet/all_datasets/exchange_rate \
  --pred_len $len \
  --data custom \
  --data_path exchange_rate.csv \
  --seq_len 64 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 64 \
  --dropout 0.5 \
  --patch_size 64 \
  --train_epochs 20 \
  --pretrain_epochs 10 > mamba_${data_path}_train_epochs_${train_epochs}'_'len_$len.log
done