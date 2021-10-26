declare -a gpu=1
declare -a manual_seed=(21)
declare -a num_rand=1000
declare -a sparsities=(50 75 90 95)

for ((j=0;j<${#manual_seed[@]};++j));
do
for ((i=0;i<${#sparsities[@]};++i)); 
do
python main.py \
	--dset=cifar10 \
	--dset_path=/home/Datasets/cifar10 \
	--arch=resnet20 \
	--config_path=./configs/neurips/iht_cifar10_resnet20_unstructured_insta_prune_freq20_${sparsities[i]}_constant.yaml \
	--workers=4 \
	--epochs=200 \
	--num_random_labels=${num_rand} \
	--batch_size=128 \
	--reset_momentum_after_recycling \
	--gpus=${gpu} \
        --manual_seed=${manual_seed[j]} \
	--experiment_root_path "./experiments_iht" \
	--exp_name="cifar10_random_${num_rand}_iht_oneshot_freq20_${sparsities[i]}_no_da" \
	--wandb_group="cifar10_resnet20_${num_rand}" \
	--wandb_name "iht_oneshot_freq20_sp${sparsities[i]}_no_da" \
        --wandb_project "acdc_cifar10_memorization" 

done
done
