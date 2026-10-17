NUM_GPUS=8
torchrun --master-port 2222 --nproc_per_node=$NUM_GPUS --nnodes=1 train/train_decoder.py \
    --train_config config/train_config/config_decoder_lm1b.yaml \
    --num_gpus $NUM_GPUS \