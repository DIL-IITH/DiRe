import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.pairwise import pairwise_cosine_similarity,pairwise_euclidean_distance
import sys 


def clip_tiny(image_tensor):
    """
    adjust the input based on mean and variance, tiny-imagenet
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)

    return image_tensor


def denormalize_tiny(image_tensor):
    """
    convert floats back to input, tiny-imagenet
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def save_images_multiple(args, images, targets,images_per_class=1):


    for id in range(images.shape[0]):

        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()


        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)

        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, args.ipc_start+(id%images_per_class))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())

class DiversityLoss(torch.nn.Module):
    def __init__(self,args_p,batch_size):
        super(DiversityLoss, self).__init__()
        self.multiplier = args_p.multiplier  # Weight tensor for scaling losses
        self.start_class_id = args_p.start_class_id
        self.batch_size = batch_size
        self.images_per_class=args_p.IPC
        self.device = args_p.device 
        self.r_cos = args_p.r_cos
        self.r_euc = args_p.r_euc
        print("Diversity loss regularizer settings")
        print("multiplier: {}, start_class_id:{}, batch_size:{}, ipc:{},device:{},r_cos:{},r_euc:{}".format(self.multiplier,self.start_class_id,self.batch_size,self.images_per_class,self.device,self.r_cos,self.r_euc))

    def forward(self, regularizer_output,original_embedding):

        for class_id_syn in range(self.start_class_id,self.start_class_id+self.batch_size):
            class_wise_val = regularizer_output[(class_id_syn-self.start_class_id)*self.images_per_class:(class_id_syn-self.start_class_id+1)*self.images_per_class,:]
            class_wise_original=original_embedding[class_id_syn]
            a=pairwise_cosine_similarity(class_wise_val,zero_diagonal=True)
            b=pairwise_euclidean_distance(class_wise_val,class_wise_original,zero_diagonal=True)
            c=1-pairwise_cosine_similarity(class_wise_val,class_wise_original,zero_diagonal=True)

            if class_id_syn == self.start_class_id:
                loss_val = self.r_cos*torch.mean(torch.abs(a))+self.r_cos*torch.mean(torch.abs(c))+ self.r_euc*torch.mean(b)
            else:
                loss_val += self.r_cos*torch.mean(torch.abs(a))+self.r_cos*torch.mean(torch.abs(c))+ self.r_euc*torch.mean(b)


        return self.multiplier*loss_val
