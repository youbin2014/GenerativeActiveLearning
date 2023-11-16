
## CIFAR10 resnet
#python demo.py --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 2 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5

## CIFAR10 vgg16
#python demo.py --net_name "vgg" --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 2 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5

## # CIFAR10 densenet
#python demo.py --net_name "densenet" --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 0 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5
#
# # CIFAR10 mobilenetv2
python demo.py --net_name "mobilenetv2" --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 0 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5

## # CIFAR10 dla
#python demo.py --net_name "dla" --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 2 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5
#
#
## # CIFAR10 dpn
#python demo.py --net_name "dpn" --ALstrategy "MarginSampling" --GALstrategy "MarginSampling" --GAL_active False --quota "9000" --gpu 2 --dataset_name "CIFAR10" --GAL_data_folder "ours" --template "a realistic photo of a {}" --epsilon 0 --emb_update_step 10 --emb_num_per_prompt 6 --samp_num_per_prompt 32 --samp_num_factor 1 --alpha_factor 5
