import torch
import torch.nn.functional as F
import os
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        if len(dataset1) == 0:
            self.X = None
        else:
            self.X = dataset1.X  # Just for compatibility, will not store any actual data

    def __getitem__(self, index):
        # Check if the index belongs to the first dataset
        if index < len(self.dataset1):
            x, y, idx = self.dataset1[index]
            return x, y, idx
        # Adjust the index and fetch from the second dataset
        adjusted_index = index - len(self.dataset1)
        x, y = self.dataset2[adjusted_index]
        return x, torch.tensor(y), index

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

def update_train_loader(data_folder,train_subset,cycle,dataset_name,GAL_active):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize images to CIFAR10 resolution
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ])

    elif dataset_name == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975],
                                 [0.2302, 0.2265, 0.2262])
       ])

    labeled_dataset = ImageFolder(root=os.path.join(data_folder, "cycle{}".format(cycle)),
                          transform=transform)
    dataset=CombinedDataset(train_subset, labeled_dataset)
    if GAL_active==1:
        dataset = CombinedDataset([], labeled_dataset)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
    #                                            num_workers=4, pin_memory=True)
    return dataset
def embedding_prepare(dataset,labels, diffuser,num_images_per_prompt,device,template):
    embeddings = []
    label_index=[]
    if dataset == "cifar10":
        # labels = ["Airplane"]
        for idx,prompt in enumerate(labels):
            if prompt=="Automobile":
                prompt="Car"
            prompt=template.format(prompt)
            # prompt='a photo of an animal'
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)
            label_index.append(idx)


    elif dataset == "svhn":
        for idx,prompt in enumerate(labels):
            prompt=template.format(prompt)
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)
            label_index.append(idx)

    elif dataset == "cifar100":
        # labels = ["Airplane"]
        for idx,prompt in enumerate(labels):
            if prompt=="Automobile":
                prompt="Car"
            prompt=template.format(prompt)
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)
            label_index.append(idx)


    elif dataset == "tinyimagenet":
        for idx,id in enumerate(labels):
            prompt = labels[id].split(", ")
            transformed_prompt = [prompt[0]]
            for word in prompt[1:]:
                transformed_prompt.append("or a " + word)
            prompt = ' '.join(transformed_prompt)
            prompt=template.format(prompt)

            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)
            label_index.append(idx)

    # return torch.cat(embeddings,dim=0)
    return embeddings,label_index
def update_embedding_reverse(imgnum_per_prompt,update_step,dataset_name,alpha,epsilon,labels,model,diffuser,device,AL_function,template):

    embeddings_list,label_index = embedding_prepare(dataset_name,labels, diffuser,imgnum_per_prompt, device,template)
    embeddings_list_updated=[]
    print("updating embeddings")
    for idx,embeddings in enumerate(embeddings_list):
        label=label_index[idx]
        embeddings=embeddings.view(-1,imgnum_per_prompt,embeddings.shape[1],embeddings.shape[2]).mean(dim=1)
        embeddings_original = embeddings.clone()
        # alpha = config["emb_alpha"]
        # epsilon = config["emb_l2_epsilon"]
        if epsilon>0:
            for i in range(update_step):
                if AL_function=='random':
                    embeddings_grad = torch.randn_like(embeddings).to(device=embeddings.device) # for random text opt
                else:
                    embeddings_grad = diffuser.compute_gradient(dataset_name=dataset_name, model=model, prompt_embeds=embeddings,label=label,num_inference_steps=50, AL_function=AL_function,num_images_per_prompt=imgnum_per_prompt)
                    embeddings_grad = embeddings_grad.view(-1,imgnum_per_prompt,embeddings_grad.shape[1],embeddings_grad.shape[2]).mean(dim=1)

                batchsize=embeddings_grad.shape[0]
                grad_norms = torch.norm(embeddings_grad.view(batchsize, -1), p=2, dim=1) +1e-8  # nopep8
                embeddings_grad = embeddings_grad / grad_norms.view(batchsize, 1, 1)
                embeddings = embeddings.detach() + alpha * embeddings_grad

                delta = embeddings - embeddings_original
                delta_norms = torch.norm(delta.view(batchsize, -1), p=2, dim=1)
                factor = epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1)
                embeddings = (embeddings_original + delta).detach()

        embeddings_list_updated.append(embeddings)

    return embeddings_list_updated

def margin(dataset_name, image, model,label=None):
    '''maximize the margin'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    probs_sorted, idxs = probs.sort(descending=True)
    uncertainties = torch.sum(probs_sorted[:, 0] - probs_sorted[:, 1])
    return uncertainties

def entropy(dataset_name, image, model,label=None):
    '''negative entropy'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    log_probs = torch.log(probs)
    uncertainties = (probs * log_probs).sum(1)
    return uncertainties.sum()

def least_confidence(dataset_name, image, model,label=None):
    '''maximize top 1 confidence'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    uncertainties = probs.max(1)[0]
    return uncertainties.sum()

def pseudo_loss(dataset_name, image, model,label=None):
    '''maximize top 1 confidence'''
    pseudo_label=torch.tensor([label]).to(device=image.device)
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(probs, pseudo_label.repeat(probs.shape[0]))
    print(loss)
    return -loss

def min_margin(dataset_name, image, model,label=None):
    '''maximize the margin'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    probs_sorted, idxs = probs.sort(descending=True)
    uncertainties = torch.sum(probs_sorted[:, 0] - probs_sorted[:, 1])
    return -uncertainties
def min_least_confidence(dataset_name, image, model,label=None):
    '''maximize top 1 confidence'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    uncertainties = probs.max(1)[0]
    return -uncertainties.sum()

def max_pseudo_loss(dataset_name, image, model,label=None):
    '''maximize top 1 confidence'''
    pseudo_label=torch.tensor([label]).to(device=image.device)
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(probs, pseudo_label.repeat(probs.shape[0]))
    # print(loss)
    return loss

def max_entropy(dataset_name, image, model,label=None):
    '''negative entropy'''
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    elif dataset_name == 'tinyimagenet':
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs=torch.nn.functional.softmax(probs)
    log_probs = torch.log(probs)
    uncertainties = (probs * log_probs).sum(1)
    return -uncertainties.sum()


# def current_model_sample_score(model,image,device):
#     model.eval()
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),  # This size might vary depending on the model
#         transforms.ToTensor()
#     ])
#     image_tensor = transform(image).unsqueeze(0).to(device)  # Add a batch dimension
#     score=max_entropy(image_tensor, model)
#     return score

def dataset_sampling(diffuser,sample_per_class,sample_per_prompt,embedding_list_updated,labels,cycle,data_folder,dataset_name):

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    epoch_folder=os.path.join(data_folder,"cycle{}".format(cycle))
    if not os.path.exists(epoch_folder):
        os.mkdir(epoch_folder)

    for idx, embedding in enumerate(embedding_list_updated):
        if dataset_name == 'tinyimagenet':
            label_list = list(labels.keys())
            # label_list.sort()
            label = label_list[idx]
        else:
            label = labels[idx]
        class_folder = os.path.join(epoch_folder, label)
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)
        # Count existing images in the folder
        existing_image_files = [f for f in os.listdir(class_folder) if f.endswith(".png")]
        generated_samples = len(existing_image_files)

        while generated_samples < sample_per_class:
            print("data sampling, class {}, samples {}/{}".format(label,generated_samples,sample_per_class))
            # Adjust the number of images to generate if we're close to the sample_per_class
            images_needed = sample_per_class - generated_samples
            current_num_images_per_prompt = min(images_needed, sample_per_prompt)

            images = diffuser(prompt_embeds=embedding, num_images_per_prompt=current_num_images_per_prompt).images
            for image_idx, image in enumerate(images):

                image.save(os.path.join(class_folder, f"{cycle}_{label}_{generated_samples}.png"))
                generated_samples+=1
