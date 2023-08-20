import torch
import torch.nn as nn
import math
from dataset import CouplingDataset, MainDataset
from torchvision.models.resnet import resnet34, resnet50, resnet18
import argparse
from utils import AverageMeter
from tqdm import tqdm
import random
import numpy as np
import os
from collections import OrderedDict
import resnet_custom
# torch.cuda.set_per_process_memory_fraction(0.2, 0)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=11037, type=int,
                    help='random seed')
parser.add_argument('--batch_size', '--bs', default=64, type=int,
                    help='batch size')
parser.add_argument('--logfile', default='log/log.txt', type=str, help='log file path')
parser.add_argument('--data_path', default='../dataset', type=str, help='path of data')
parser.add_argument('--savename', type=str, help='checkpoint name', default='demo.pth')
parser.add_argument('--evaluate', '-e', action='store_true', default=True)
parser.add_argument('--filtered', action='store_true', default=False)
parser.add_argument('--filter_threshold', default=0.05, type=float)
parser.add_argument('--log', action='store_true', default=False)
parser.add_argument('--loss', default='msre', choices=['mse', 'msre'])
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=["resnet34", "resnet50", "resnet50_no_avgpool", "resnet18"], default="resnet34")
parser.add_argument('--goal', type=str, default='total', choices=['total', 'env'])
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)',
                dest='weight_decay')
args = parser.parse_args()
print(args)
if os.path.exists(args.logfile) and not args.evaluate:
    print(f"error: {args.logfile} exists, abort.")
    exit()
logfile = open(args.logfile, 'a')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def get_model(typ : str):
    if typ == "resnet34":
        return resnet34(num_classes=1)
    elif typ == "resnet50":
        return resnet50(num_classes=1)
    elif typ == "resnet18":
        return resnet18(num_classes=1)
    elif typ == "resnet50_no_avgpool":
        return resnet_custom.resnet50_no_avgpool(num_classes=1)
    else: 
        raise NotImplementedError

def test(test_loader, model, epoch, args):
    losses = AverageMeter()
    maxerr = 0
    errs = []
    predicts = []
    model.eval()
    with torch.no_grad():
        for i, (xs, ys) in enumerate(tqdm(test_loader)):
            xs = xs.cuda()
            ys = ys.cuda()
            if args.log:
                normalized_ys = torch.log(ys)
            else:
                normalized_ys = ys

            predict = model(xs)
            if args.loss == 'mse':
                loss = torch.mean((predict - normalized_ys) ** 2)
            else:
                loss = torch.mean((1 - predict / normalized_ys) ** 2)

            losses.update(loss.item(), ys.size(0))
            if args.log:
                y_pred = torch.exp(predict)
            else:
                y_pred = predict
            err = torch.abs((y_pred - ys) / ys)
            err = err.cpu().numpy()
            maxerr = max(maxerr, err.max())
            errs += err.tolist()
            predicts += y_pred.cpu().numpy().tolist()
    avgerr = np.mean(errs)
    print(epoch, losses.avg, maxerr, avgerr)
    five_ratio = np.sum(np.array(errs)>report_ratio)/len(errs)
    logfile.write('Testing {} {} {} {} {}\n'.format(epoch, losses.avg, maxerr, avgerr, five_ratio))
    logfile.flush()
    return losses.avg, maxerr, avgerr, errs, predicts


target_layers = "POLY1_MET1_MET2"
goal = "total" if args.goal == "total" else "env"
report_ratio = 0.05 if args.goal == "total" else 0.1
with open(os.path.join(args.data_path, f"label/{target_layers}_{goal}_val.txt"), "r") as f:
    val_content = f.read().strip().splitlines(keepends=False)
if args.goal == "total":
    DataSetClass = MainDataset
elif args.goal == "env":
    DataSetClass = CouplingDataset
val_dataset = DataSetClass(target_layers, val_content, args.data_path, None, filter_threshold = args.filter_threshold)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=8)

model = get_model(args.model_type)
model = model.cuda()


logfile.write('{}\n'.format(args))
logfile.flush()


if args.pretrained is not None:
    info = torch.load(args.pretrained)
    checkpoint = info['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
if args.evaluate:
    loss, maxerr, avgerr, errs, preds = test(valloader, model, 0, args)
    f_errs = logfile
    earr = np.array(errs)
    max_err = math.ceil(np.max(earr)*100)
    hist_dat, hist_sp = np.histogram(earr, max_err, (0, max_err/100))
    for i in range(max_err):
        f_errs.write(f"{hist_sp[i]}\t{hist_dat[i]}\n")
    standard = report_ratio
    five = np.sum(earr>standard)/earr.shape[0]
    f_errs.write(f"error over {standard*100}% :{(1-five)*100}%\n")
    results = []
    for i,err in enumerate(errs):
        f_errs.write(f"{i}, {err[0]}, {preds[i][0]}, {val_dataset.cases[i]}\n")
        results.append((preds[i][0], val_dataset.cases[i].val))
    np.save(f"{args.logfile}.npy", np.array(results))
