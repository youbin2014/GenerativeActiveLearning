## SVHN
#python demo.py --dataset_name 'SVHN'



## Cifar100
#python demo.py --dataset_name 'CIFAR100'



# TinyImageNet
#python demo.py --dataset_name 'TinyImageNet'

## Ablation Study
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 0 --dataset_name "TinyImageNet" --GAL_data_folder "RS_epsilon_tinyimagenet_linear_0.5_1xgen" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5

# CIFAR100
python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 1 --dataset_name "CIFAR100" --GAL_data_folder "RS_epsilon_tinyimagenet_linear_0.5_1xgen" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5