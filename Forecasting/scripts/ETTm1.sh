for len in 96 192 336 720
do
  python -u TSLANet_Forecasting.py \
  --root_path /mnt/sdb/hhj/TSLANet/all_datasets/ETT-small/ \
  --pred_len $len \
  --data ETTm1 \
  --data_path ETTm1.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 2 \
  --batch_size 512 \
  --dropout 0.5 \
  --patch_size 8 \
  --train_epochs 50 \
  --pretrain_epochs 10 > mamba_${data_path}_train_epochs_${train_epochs}'_'len_$len.log
done