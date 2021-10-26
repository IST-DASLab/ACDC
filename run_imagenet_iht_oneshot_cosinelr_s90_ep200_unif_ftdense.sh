declare -a gpu=4,5,6,7
declare -a manual_seed=(1)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=$IMAGENET_PATH \
	--arch=resnet50 \
	--config_path=./configs/neurips/iht_imagenet_resnet50_insta_cosinelr_s90_ep200_ftdense.yaml \
	--workers=20 \
	--fp16 \
	--epochs=200 \
	--reset_momentum_after_recycling \
	--checkpoint_freq 10 \
	--batch_size=256 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=iht_imagenet_resnet50_oneshot_cosinelr_fp16_s90_ep200_unif_gradmask_ftdense \
	--from_checkpoint_path "./experiments_iht/iht_imagenet_resnet50_oneshot_cosinelr_fp16_s90_ep200_unif_gradmask/seed1/20210710193121/best_dense_checkpoint.ckpt" \
        --wandb_project "iht_imagenet_flops" 

done
