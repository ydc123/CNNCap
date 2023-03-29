import torch
import torch.nn as nn
import json
from torch import optim
from dataset import GridCapData
from resnet import resnet34
import argparse
from utils import AverageMeter
import time
from tqdm import tqdm
import random
import numpy as np
import os
import shutil
from collections import OrderedDict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epoch', default=100, type=int,
                    help='number of epochs')
parser.add_argument('--seed', default=11037, type=int,
                    help='random seed')
parser.add_argument('--batch_size', '--bs', default=64, type=int,
                    help='batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                help='momentum')
parser.add_argument('--logfile', default='log/log.txt', type=str, help='log file path')
parser.add_argument('--data_path', default='data/55nm_B_2_3_6.json', type=str, help='path of data')
parser.add_argument('--savename', type=str, help='checkpoint name', default='demo.pth')
parser.add_argument('--filtered', action='store_true', default=False)
parser.add_argument('--log', action='store_true', default=False)
parser.add_argument('--loss', default='msre', choices=['mse', 'msre'])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--goal', type=str, default='total', choices=['total', 'env', 'all'])
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)',
                dest='weight_decay')
args = parser.parse_args()
print(args)
logfile = open(args.logfile, 'w')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def save_state(model, epoch, loss, args, optimizer, isbest):
    dirpath = 'saved_models/'
    os.makedirs(dirpath, exist_ok=True)
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'isbest': isbest,
            'loss': loss,
            }
    filename = args.savename
    torch.save(state,dirpath+filename)
    if isbest:
        shutil.copyfile(dirpath+filename, dirpath+'best.'+filename)
def train(train_loader, optimizer, model, epoch, args):
    losses = AverageMeter()
    maxerr = 0
    errs = []

    model.train()
    for i, (xs, ys, masks, ys_total) in enumerate(tqdm(train_loader)):

        xs = xs.cuda()
        ys = ys.cuda()
        masks = masks.bool().cuda()
        if args.log:
            normalized_ys = torch.log(ys)
        else:
            normalized_ys = ys

        predict = model(xs)
        if args.loss == 'mse':
            loss = torch.mean((predict - normalized_ys).masked_select(masks) ** 2)
        else:
            loss = torch.mean((1 - predict / normalized_ys).masked_select(masks) ** 2)

        losses.update(loss.item(), ys.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.log:
            y_pred = torch.exp(predict)
        else:
            y_pred = predict
        err = torch.abs((y_pred - ys) / ys).masked_select(masks)
        err = err.data.cpu().numpy()
        maxerr = max(maxerr, err.max())
        errs += err.tolist()
    avgerr = np.mean(errs)
    print(epoch, losses.avg, maxerr, avgerr)
    logfile.write('Training {} {} {} {}\n'.format(epoch, losses.avg, maxerr, avgerr))
    logfile.flush()
    return losses.avg
def test(test_loader, model, epoch, args):
    losses = AverageMeter()
    maxerr = 0
    errs = []

    model.eval()
    with torch.no_grad():
        for i, (xs, ys, masks, ys_total) in enumerate(tqdm(test_loader)):
            xs = xs.cuda()
            ys = ys.cuda()
            masks = masks.bool().cuda()
            if args.log:
                normalized_ys = torch.log(ys)
            else:
                normalized_ys = ys

            predict = model(xs)
            if args.loss == 'mse':
                loss = torch.mean((predict - normalized_ys).masked_select(masks) ** 2)
            else:
                loss = torch.mean((1 - predict / normalized_ys).masked_select(masks) ** 2)

            losses.update(loss.item(), ys.size(0))
            if args.log:
                y_pred = torch.exp(predict)
            else:
                y_pred = predict
            err = torch.abs((y_pred - ys) / ys).masked_select(masks)
            err = err.cpu().numpy()
            maxerr = max(maxerr, err.max())
            errs += err.tolist()
    avgerr = np.mean(errs)
    print(epoch, losses.avg, maxerr, avgerr)
    logfile.write('Testing {} {} {} {}\n'.format(epoch, losses.avg, maxerr, avgerr))
    logfile.flush()
    return losses.avg, maxerr, errs

with open(args.data_path, 'r') as f:
    all_data = json.load(f)
nSample = len(all_data)
indices = list(range(nSample))
random.shuffle(indices)
train_size = round(nSample * 0.9)
train_indices = indices[:train_size]
val_indices = indices[train_size:]
train_dataset = GridCapData(all_data, train_indices, args.goal, args.filtered)
val_dataset = GridCapData(all_data, val_indices, args.goal, args.filtered)
model = resnet34(in_channel=val_dataset.in_channel)
model = model.cuda()
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=8)
optimizer = optim.Adam(model.parameters(), 
    lr=args.lr, weight_decay=args.weight_decay)
logfile.write('{}\n'.format(args))
logfile.flush()
best_maxerr = 1e100
if args.resume is not None:
    info = torch.load(args.resume)
    checkpoint = info['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(info['optimizer'])
if args.pretrained is not None:
    info = torch.load(args.pretrained)
    checkpoint = info['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
for epoch in tqdm(range(args.epoch)):
    print(args)
    train_loss = train(trainloader, optimizer, model, epoch, args)
    val_loss, maxerr, errs = test(valloader, model, epoch, args)
    if best_maxerr > maxerr:
        best_maxerr = maxerr
        isbest = True
    else:
        isbest = False
    save_state(model, epoch, maxerr, args, optimizer, isbest)
