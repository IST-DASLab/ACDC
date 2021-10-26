declare -a gpu=4,5,6,7
declare -a manual_seed=(3)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=/home/Datasets/ILSVRC \
	--arch=mobilenet \
	--config_path=./configs/neurips/iht_imagenet_mobilenet_insta_cosinelr_ep100_ftdense.yaml \
	--workers=20 \
	--epochs=100 \
	--fp16 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 10 \
	--batch_size=256 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=iht_imagenet_mobilenet_oneshot_cosinelr_fp16_s90_ep100_ftdense \
	--from_checkpoint_path "CHECKPOINT" \
        --wandb_project "imagenet_mobilenet" 

done
