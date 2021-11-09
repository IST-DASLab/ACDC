declare -a gpu=0
declare -a manual_seed=(1)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=imagenet \
	--dset_path=$IMAGENET_PATH \
	--arch=mobilenet \
	--config_path=configs/neurips/imagenet_mobilenet_cosinelr_ep100_dense.yaml \
	--workers=20 \
	--epochs=100 \
	--batch_size=256 \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=acdc_imagenet_mobilenet_eval \
	--only_model \
	--eval_only \
	--from_checkpoint_path=CHECKPOINT_PATH \
        --wandb_project "acdc_imagenet" 

done
