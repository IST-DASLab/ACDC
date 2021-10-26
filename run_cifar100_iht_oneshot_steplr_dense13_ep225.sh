declare -a gpu=4,5,6,7
declare -a manual_seed=(10)
declare -a sparsity=(50 75 90 95)


for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#sparsity[@]};++i));
do
python main.py \
	--dset=cifar100 \
	--dset_path=/home/Datasets/ \
	--arch=wideresnet \
	--config_path=./configs/neurips/iht_cifar100_wideresnet_steplr_dense13_s${sparsity[i]}.yaml \
	--workers=4 \
	--epochs=225 \
	--warmup_epochs=5 \
	--batch_size=128 \
	--gpus=${gpu} \
	--reset_momentum_after_recycling \
	--checkpoint_freq 50 \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name=cifar100_wideresnet_dense13_s${sparsity[i]} \
        --wandb_project "cifar100_wideresnet" 

done
done
