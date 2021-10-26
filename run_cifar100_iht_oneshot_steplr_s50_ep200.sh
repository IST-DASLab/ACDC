declare -a gpu=0,1
declare -a manual_seed=(10)


for ((j=0;j<${#manual_seed[@]};++j));
do
python main.py \
	--dset=cifar100 \
	--dset_path=~/Datasets/cifar100 \
	--arch=wideresnet \
	--config_path=./configs/neurips/iht_cifar100_wideresnet_steplr_freq20_s50.yaml \
	--workers=4 \
	--epochs=200 \
	--warmup_epochs=5 \
	--batch_size=128 \
	--gpus=${gpu} \
	--reset_momentum_after_recycling \
	--checkpoint_freq 50 \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=cifar100_wideresnet \
        --wandb_project "cifar100_wideresnet" 

done
