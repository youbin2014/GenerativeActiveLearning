## SVHN
#python demo.py --dataset_name 'SVHN'



## Cifar100
#python demo.py --dataset_name 'CIFAR100'

#nohup bash -u experiments.sh > Abl_text_template.out 2>&1 &

# TinyImageNet
#python demo.py --dataset_name 'TinyImageNet'

<<<<<<< HEAD
## Ablation Study
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 0 --dataset_name "TinyImageNet" --GAL_data_folder "RS_epsilon_tinyimagenet_linear_0.5_1xgen" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5

# CIFAR100
python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 1 --dataset_name "CIFAR100" --GAL_data_folder "RS_epsilon_tinyimagenet_linear_0.5_1xgen" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5
=======
# Ablation Study
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_1" --template "a photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_2" --template "{}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_3" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_4" --template "a simple photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_5" --template "a recent color photograph of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "4000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "abl_temp_opt_6" --template "a recent color photograph of a {}, ultra detailed, 4k resolution" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 0.5 --alpha_factor 5

#main exp
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active 1 --quota "9000" --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 1 --alpha_factor 5

#abl for text opt
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_1" --template "a realistic photo of a {}" --epsilon 200 --emb_update_step 30 --emb_num_per_prompt 16 --samp_num_per_prompt 50 --samp_num_factor 0.01 --alpha_factor 20
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_2" --template "a realistic photo of a {}" --epsilon 50 --emb_update_step 30 --emb_num_per_prompt 16 --samp_num_per_prompt 50 --samp_num_factor 0.01 --alpha_factor 20
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 100 --batch 100 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_3" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 30 --emb_num_per_prompt 16 --samp_num_per_prompt 50 --samp_num_factor 1 --alpha_factor 20
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_4" --template "a realistic photo of a {}" --epsilon 10 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 1 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_5" --template "a realistic photo of a {}" --epsilon 25 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 0.1 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_6" --template "a realistic photo of a {}" --epsilon 10 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 0.1 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_7" --template "a realistic photo of a {}" --epsilon 25 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 0.1 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "0" --initseed 10000 --batch 10000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_text_opt_8" --template "a realistic photo of a {}" --epsilon 10 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 50 --samp_num_factor 0.1 --alpha_factor 5

python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "8000" --initseed 2000 --batch 2000 --gpu 1 --dataset_name "CIFAR10" --GAL_data_folder "Abl_epsilon_0" --template "{}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 100 --samp_num_factor 0.25 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "8000" --initseed 2000 --batch 2000 --gpu 0 --dataset_name "CIFAR10" --GAL_data_folder "Abl_epsilon_05" --template "{}" --epsilon 0.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 100 --samp_num_factor 0.25 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "8000" --initseed 2000 --batch 2000 --gpu 0 --dataset_name "CIFAR10" --GAL_data_folder "Abl_epsilon_25" --template "{}" --epsilon 2.5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 100 --samp_num_factor 0.25 --alpha_factor 5
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "min_loss" --GAL_active 1 --quota "8000" --initseed 2000 --batch 2000 --gpu 0 --dataset_name "CIFAR10" --GAL_data_folder "Abl_epsilon_50" --template "{}" --epsilon 5 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 100 --samp_num_factor 0.25 --alpha_factor 5
>>>>>>> 924624dadaba159831552bcd0d3137988e9431f2
