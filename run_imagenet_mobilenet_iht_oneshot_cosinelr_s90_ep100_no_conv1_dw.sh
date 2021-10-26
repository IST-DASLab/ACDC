declare -a gpu=0,1,2,3
declare -a manual_seed=(1)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=/home/Datasets/ILSVRC/Data/CLS-LOC/ \
	--arch=mobilenet \
	--config_path=./configs/neurips/iht_imagenet_mobilenet_insta_cosinelr_s90_ep100_no_conv1_dw.yaml \
	--workers=20 \
	--epochs=100 \
	--fp16 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 10 \
	--batch_size=256 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--from_checkpoint_path "./experiments_iht/iht_imagenet_mobilenet_cosinelr_fp16_s90_ep100_no_conv1_dw/seed1/20210506140643/regular_checkpoint9.ckpt" \
	--exp_name=iht_imagenet_mobilenet_cosinelr_fp16_s90_ep100_no_conv1_dw \
        --wandb_project "iht_imagenet_mobilenet" 

done
