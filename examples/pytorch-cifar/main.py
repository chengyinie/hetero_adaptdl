# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import adaptdl
import adaptdl.torch as adl

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=True,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--mixed-precision', dest='mixed_precision', default=False,
                    action='store_true', help='use automatic mixed precision')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

if adaptdl.env.replica_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = adl.HeteroDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
else:
    dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = adl.HeteroDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

if args.autoscale_bsz:
    trainloader.autoscale_batch_size(4000, local_bsz_bounds=(20, 4000), gradient_accumulation=False)

validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
validloader = adl.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = eval(args.model)()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)

if args.mixed_precision:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None
net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler, scaler)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    stats = adl.Accumulator()
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        if args.mixed_precision:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item()

    trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
    net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/")
    if args.mixed_precision:
        writer.add_scalar("MixedPrecision/scale", scaler.get_scale(), epoch)
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        print("Train:", stats)


def valid(epoch):
    net.eval()
    stats = adl.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
        print("Valid:", stats)


# tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")
#                                if adaptdl.env.replica_rank() == 0 else "/tmp",
#                                adaptdl.env.job_id())
tensorboard_dir = './tensorboard'
with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adl.remaining_epochs_until(args.epochs):
        train(epoch)
        valid(epoch)
        lr_scheduler.step()

