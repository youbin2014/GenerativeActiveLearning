import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from GAL import update_embedding_reverse,max_entropy,dataset_sampling,update_train_loader
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

    def GAL_train(self, data = None, model_name = None,cycle=None,iter=None):
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

                # epsilons = self.args_task['epsilon']
                # alphas = self.args_task['alpha']
                epsilons = self.args_task['epsilon'] / 11 * (cycle + 1)
                alphas = epsilons / 5
                model=self.net.get_model()
                emb_num_per_prompt=self.args_task['emb_num_per_prompt']
                update_step=self.args_task['emb_update_step']
                samp_num_per_prompt=self.args_task['samp_num_per_prompt']
                # samp_num_per_class=self.args_task['samp_num_per_class']
                samp_num_per_class=int(len(labeled_idxs)/len(labels)) ## for sample numbers
                data_folder = './generated_data/{}/{}_{}'.format(dataset_name,self.args_task['data_folder'],iter)
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                embedding_list_updated = update_embedding_reverse(emb_num_per_prompt,update_step,dataset_name, alphas, epsilons, labels, model,
                                                                  self.diffuser, self.device, max_entropy)
                dataset_sampling(self.diffuser, samp_num_per_class,samp_num_per_prompt, embedding_list_updated, labels, cycle,
                                 data_folder, dataset_name)
                labeled_data = update_train_loader(data_folder, labeled_data, cycle, dataset_name)
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

