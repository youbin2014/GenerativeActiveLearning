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

def update_train_loader(data_folder,train_subset,cycle):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to CIFAR10 resolution
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    labeled_dataset = ImageFolder(root=os.path.join(data_folder, "cycle{}".format(cycle)),
                          transform=transform)
    dataset=CombinedDataset(train_subset, labeled_dataset)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
    #                                            num_workers=4, pin_memory=True)
    return dataset
def embedding_prepare(dataset,labels, diffuser,num_images_per_prompt,device):
    embeddings = []
    if dataset == "cifar10":
        # labels = ["Airplane"]
        for prompt in labels:
            if prompt=="Automobile":
                prompt="Car"
            prompt='a photo of a ' + prompt
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)

    elif dataset == "svhn":
        for prompt in labels:
            prompt='a house number style image of a single digit' + prompt
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)

    elif dataset == "cifar100":
        # labels = ["Airplane"]
        for prompt in labels:
            if prompt=="Automobile":
                prompt="Car"
            prompt='a photo of a ' + prompt
            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)


    elif dataset == "tinyimagenet":
        for prompt in labels:
            prompt = prompt.split(", ")
            transformed_prompt = ["A photo of a " + prompt[0]]
            for word in prompt[1:]:
                transformed_prompt.append("or a " + word)
            prompt = ' '.join(transformed_prompt)

            prompt_embeds,negative_prompt_embeds = diffuser.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
            )
            embeddings.append(prompt_embeds)

    # return torch.cat(embeddings,dim=0)
    return embeddings
def update_embedding_reverse(imgnum_per_prompt,update_step,dataset_name,alpha,epsilon,labels,model,diffuser,device,AL_function):

    embeddings_list = embedding_prepare(dataset_name,labels, diffuser,imgnum_per_prompt, device)
    embeddings_list_updated=[]
    print("updating embeddings")
    for embeddings in embeddings_list:
        embeddings=embeddings.view(-1,imgnum_per_prompt,embeddings.shape[1],embeddings.shape[2]).mean(dim=1)
        embeddings_original = embeddings.clone()
        # alpha = config["emb_alpha"]
        # epsilon = config["emb_l2_epsilon"]
        for i in range(update_step):
            embeddings_grad = diffuser.compute_gradient(model=model, prompt_embeds=embeddings,num_inference_steps=50, AL_function=AL_function,num_images_per_prompt=imgnum_per_prompt)
            embeddings_grad = embeddings_grad.view(-1,imgnum_per_prompt,embeddings_grad.shape[1],embeddings_grad.shape[2]).mean(dim=1)
            batchsize=embeddings_grad.shape[0]
            grad_norms = torch.norm(embeddings_grad.view(batchsize, -1), p=2, dim=1) +1e-8  # nopep8
            embeddings_grad = embeddings_grad / grad_norms.view(batchsize, 1, 1)
            embeddings = embeddings.detach() - alpha * embeddings_grad

            delta = embeddings - embeddings_original
            delta_norms = torch.norm(delta.view(batchsize, -1), p=2, dim=1)
            factor = epsilon / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1)

            embeddings = (embeddings_original + delta).detach()
        embeddings_list_updated.append(embeddings)

    return embeddings_list_updated

def max_entropy(image, model):
    image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False).float()
    probs,_ = model(image)
    probs_sorted, idxs = probs.sort(descending=True)
    uncertainties = torch.sum(probs_sorted[:, 0] - probs_sorted[:, 1])
    return uncertainties


def current_model_sample_score(model,image,device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # This size might vary depending on the model
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add a batch dimension
    score=max_entropy(image_tensor, model)
    return score

def dataset_sampling(diffuser,sample_per_class,sample_per_prompt,embedding_list_updated,labels,cycle,data_folder):

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    epoch_folder=os.path.join(data_folder,"cycle{}".format(cycle))
    if not os.path.exists(epoch_folder):
        os.mkdir(epoch_folder)

    for idx, embedding in enumerate(embedding_list_updated):
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
