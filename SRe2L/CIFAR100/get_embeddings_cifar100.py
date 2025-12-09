import torchvision 
import torchvision.transforms as transforms
import argparse 
import os 
from tqdm import tqdm 
import numpy as np
import torch 
from torch import nn 

ap=argparse.ArgumentParser()
ap.add_argument("--device",default='cuda:0')
ap.add_argument("--model_path",required=True,help="path to the model checkpoint")
args=ap.parse_args()

if not os.path.exists('embeddings'):
    os.mkdir("embeddings")

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

mean = np.array([0.5071, 0.4867, 0.4408])
std = np.array([0.2675, 0.2565, 0.2761])

normalize = transforms.Normalize(mean=mean,std=std)
transform_train=transforms.Compose([
    transforms.ToTensor(),
    normalize
])

train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=500,
    shuffle=False
)

model_teacher = torchvision.models.get_model("resnet18", num_classes=100)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()
model_teacher.avgpool.register_forward_hook(get_activation('avgpool'))
model_teacher=model_teacher.to(args.device)
checkpoint = torch.load(args.model_path,map_location=args.device)
model_teacher.load_state_dict(checkpoint)

class_id=0
pbar=tqdm(range(len(trainloader)))
for images,_ in trainloader:
    images = images.to(args.device)
    outputs = model_teacher(images)
    torch.save(activation["avgpool"].reshape(-1,512),'embeddings/class_{}.pt'.format(class_id))
    pbar.update(1)
    class_id +=1 


