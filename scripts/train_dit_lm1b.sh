NUM_GPUS=8
torchrun --master-port 1111 --nproc_per_node=$NUM_GPUS --nnodes=1 train/train_dit.py \
    --train_config config/train_config/config_dit_lm1b.yaml \
    --num_gpus $NUM_GPUS \