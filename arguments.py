import warnings
import argparse
import sys
import os
import re

import yaml
from ast import literal_eval
import copy

	
def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	#basic arguments
	parser.add_argument('--ALstrategy', '-a', default='MarginSampling', type=str, help='name of active learning strategies')
	parser.add_argument('--GALstrategy', '-ga', default='MarginSampling', type=str,help='name of active learning strategies')
	parser.add_argument('--GAL_active', '-active', default=2, type=int, help='0 for AL, 1 for GAL, 2 for GAL+AL')
	parser.add_argument('--initseed', '-s', default = 1000, type = int, help = 'Initial pool of labeled data')
	parser.add_argument('--quota', '-q', default=4000, type=int, help='quota of active learning')
	parser.add_argument('--batch', '-b', default=1000, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--dataset_name', '-d', default='CIFAR10', type=str, help='dataset name')
	parser.add_argument('--GAL_data_folder', '-f', default='baseline', type=str, help='folder path for generated data')
	parser.add_argument('--template', '-temp', default='a photo of a {}', type=str, help='template for T2I, labels will be put in {}')
	parser.add_argument('--epsilon', '-e', default=0.5, type=float, help='range of linear epsilon schedule for text opt')
	parser.add_argument('--emb_update_step', '-es', default=10, type=int, help='step for text opt')
	parser.add_argument('--emb_num_per_prompt', '-en', default=6, type=int, help='image sampling num for text opt')
	parser.add_argument('--samp_num_per_prompt', '-bs', default=32, type=int, help='image sampling batch for each text')
	parser.add_argument('--samp_num_factor', '-x', default=0.5, type=float, help='total image sampling num/batch size in AL')
	parser.add_argument('--alpha_factor', '-alpha', default=5, type=float, help='epsilon/alpha')
	parser.add_argument('--gpu', '-g', default = '1', type = str, help = 'which gpu')

	parser.add_argument('--iteration', '-t', default=1, type=int, help='time of repeat the experiment')
	parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')
	parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', default=True, help='If training is to be done on a GPU')
	#parser.add_argument('--model', '-m', default='ResNet18', type=str, help='model name')
	parser.add_argument('--seed', default=4666, type=int, help='random seed')
	
	# lpl
	parser.add_argument('--lpl_epoches', type=int, default=20, help='lpl epoch num after detach')
	# ceal
	parser.add_argument('--delta', type=float, default=5 * 1e-5, help='value of delta in ceal sampling')
	#hyper parameters
	parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')

	#specific parameters
	parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')

	parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
	parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
	parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
	parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')

	
	args = parser.parse_args()
	return args



