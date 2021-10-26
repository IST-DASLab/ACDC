declare -a gpu=0,1,2,3
declare -a manual_seed=(2)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=$IMAGENET_PATH \
	--arch=resnet50 \
	--config_path=./configs/neurips/iht_imagenet_resnet50_insta_cosinelr_s95_ep100.yaml \
	--workers=20 \
	--fp16 \
	--epochs=100 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 10 \
	--batch_size=256 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=iht_imagenet_resnet50_oneshot_cosinelr_fp16_s95_ep100_gradmask \
        --wandb_project "iht_imagenet_flops" 

done
