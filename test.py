import torch
import torch.nn as nn
from dataset import GridCapData
from resnet import resnet34
import argparse
from utils import AverageMeter
import time
from tqdm import tqdm
import random
import numpy as np
import os
import json
import shutil
from collections import OrderedDict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=11037, type=int,
                    help='random seed')
parser.add_argument('--batch_size', '--bs', default=256, type=int,
                    help='batch size')
parser.add_argument('--data_path', default='data/55nm_total_2.json', type=str, help='path of data')
parser.add_argument('--log', action='store_true', default=False)
parser.add_argument('--logfile', default='log/eval_log.txt', type=str, help='log file path')
parser.add_argument('--goal', type=str, default='total', choices=['total', 'env', 'all'])
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()
print(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def test(test_loader, model, args):
    losses = AverageMeter()
    maxerr = 0
    msgs = []

    model.eval()
    logfile = open(args.logfile, 'w')
    with torch.no_grad():
        for i, (xs, ys, masks, ys_total) in enumerate(tqdm(test_loader)):
            xs = xs.cuda()
            ys = ys.cuda()
            masks = masks.bool().cuda()

            predict = model(xs)
            if args.log:
                y_pred = torch.exp(predict)
            else:
                y_pred = predict
            errs = (y_pred - ys) / ys
            for j in range(errs.shape[0]):
                err = errs[j].masked_select(masks[j])
                y = ys[j].masked_select(masks[j])
                for k in range(err.shape[0]):
                    msgs.append((err[k].item(), y[k].item(), ys_total[j]))
            errs = errs.masked_select(masks)
            maxerr = max(maxerr, err.max().item())
    for msg in msgs:
        logfile.write('{},{},{}\n'.format(msg[0], msg[1], msg[2]))
    logfile.close()
    return losses.avg, maxerr

with open(args.data_path, 'r') as f:
    all_data = json.load(f)
nSample = len(all_data)
indices = list(range(nSample))
random.shuffle(indices)
train_size = round(nSample * 0.9)
train_indices = indices[:train_size]
val_indices = indices[train_size:]
test_dataset = GridCapData(all_data, val_indices, args.goal, filtered=True)
model = resnet34(in_channel=test_dataset.in_channel)
model = model.cuda()
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=8)
if args.model is not None:
    print(args.model)
    info = torch.load(args.model)
    checkpoint = info['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
loss, maxerr = test(testloader, model, args)
