import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from GAL import update_embedding_reverse,dataset_sampling,update_train_loader,pseudo_loss,margin,least_confidence,entropy,max_entropy,min_margin,max_pseudo_loss,min_least_confidence
import os
class Strategy:
    def __init__(self, dataset, net, args_input, args_task,diffuser=None):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task
        self.diffuser=diffuser
        self.device='cuda:{}'.format(args_input.gpu)

    def query(self, n):
        pass
    
    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)
    
    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, data = None, model_name = None):
        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                self.net.train(labeled_data)
            else:
                self.net.train(data)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    def GAL_train(self, data = None, model_name = None,cycle=None,iter=None, GAL_active=None):
        # Get the current working directory
        current_directory = os.getcwd()
        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                dataset_name=self.args_task['name'].lower()
                if dataset_name == 'cifar100':
                    with open('./cifar-100-labels.json', 'r') as file:
                        labels = json.load(file)
                elif dataset_name == 'tinyimagenet':
                    with open('./tiny_imagenet_labels.json', 'r') as file:
                        labels = json.load(file)
                else:
                    labels = self.args_task['labels']

                # epsilons = self.args_task['epsilon'] #for epsilon ablation study
                # alphas = self.args_task['alpha']

                total_cycle=(self.args_input.quota+self.args_input.initseed)/self.args_input.batch
                epsilons = self.args_input.epsilon / total_cycle * (cycle + 1) #linear epsilon
                # epsilons = self.args_input.epsilon
                print("epsilon=",epsilons)
                alphas = epsilons / self.args_input.alpha_factor
                model=self.net.get_model()
                emb_num_per_prompt=self.args_input.emb_num_per_prompt
                update_step=self.args_input.emb_update_step
                samp_num_per_prompt=self.args_input.samp_num_per_prompt
                # samp_num_per_class=self.args_task['samp_num_per_class']
                samp_num_per_class=int(len(labeled_idxs)/len(labels)*self.args_input.samp_num_factor) ## for sample numbers
                data_folder = './generated_data/{}_{}_iter{}'.format(dataset_name,self.args_input.GAL_data_folder,iter)

                GAL_strategy=self.args_input.GALstrategy.lower()
                if 'margin' in GAL_strategy:
                    GAL_function=min_margin
                elif 'confidence' in GAL_strategy:
                    GAL_function=min_least_confidence
                elif 'entropy' in GAL_strategy:
                    GAL_function=max_entropy
                elif 'min_loss' in GAL_strategy:
                    GAL_function=pseudo_loss
                elif 'random' in GAL_strategy:
                    GAL_function='random'
                else:
                    print('unrecognized GAL strategy')

                template=self.args_input.template

                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                embedding_list_updated = update_embedding_reverse(emb_num_per_prompt,update_step,dataset_name, alphas, epsilons, labels, model,
                                                                  self.diffuser, self.device, GAL_function,template)
                dataset_sampling(self.diffuser, samp_num_per_class,samp_num_per_prompt, embedding_list_updated, labels, cycle,
                                 data_folder, dataset_name)
                labeled_data = update_train_loader(data_folder, labeled_data, cycle, dataset_name,GAL_active)
                self.net.train(labeled_data)
            else:
                self.net.train(data)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled,X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
    
    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

