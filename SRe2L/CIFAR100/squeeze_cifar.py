# Modified from https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L/*small_dataset

import argparse
import os
import time
from tqdm import tqdm 
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--output-dir", default="./save", type=str)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--check-ckpt", default=None, type=str)
parser.add_argument("--device",default="cuda:0")
args = parser.parse_args()

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print(f"==> test ckp: {args.check_ckpt}, acc: {best_acc}, epoch: {start_epoch}")
    exit()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


device = args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print("==> Building model..")

model = torchvision.models.get_model("resnet18", num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()


net = model.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Train
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss,correct*100.0/total
    # print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct*100.0/total 



start_time = time.time()
best_acc = 0.0
pbar = tqdm(range(args.epochs))
for epoch in range(start_epoch, start_epoch + args.epochs):
    train_loss,train_acc=train(epoch)
    # fast test
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        test_acc=test(epoch)
        if test_acc>best_acc:
            best_acc = test_acc 
            path = os.path.join(args.output_dir, "./ckpt.pth")
            torch.save(net.state_dict(), path)

    scheduler.step()
    pbar.update(1)
    pbar.set_postfix_str("Train loss: {:.3f}, Train acc: {:.3f}, best acc: {:.3f}".format(train_loss,train_acc,best_acc))
end_time = time.time()
print(f"total time: {end_time - start_time} s")
