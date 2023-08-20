import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm
Case = namedtuple("Case", ["x_offset", "y_offset", "x", "y", "mid", "cid", "lid", "mval", "val"])
class CouplingDataset(Dataset):
    def __init__(self, layers, labels, np_dir, indicies, padding = 12, filter_threshold = None):
        self.window_size = 200
        lys = layers.split("_")
        self.layer_name_map = {}
        dens_map = []
        idx_map = []
        for idx, ly in enumerate(lys):
            self.layer_name_map[ly]=idx
            arrs = np.load(os.path.join(np_dir, f"{ly}.npz"))
            dens_map.append(arrs["img"])
            idx_map.append(arrs["idx"])
        padding_vec = [(0, 0), (self.window_size, self.window_size), (self.window_size, self.window_size)]
        self.dens_map = np.pad(np.array(dens_map), padding_vec, "constant")
        self.idx_map = np.pad(np.array(idx_map), padding_vec, "constant")
        self.dens_map.setflags(write=0)
        self.idx_map.setflags(write=0)
        self.cases = []
        self.padding = padding
        # with open(os.path.join(labels, f"{layers}_cp.txt"), "r") as f:
        #     f_content = f.read().strip().splitlines(keepends=False)
        if indicies is None:
            indicies = list(range(len(labels)))
        for i in indicies:
            line = labels[i]
            ls = line.split()
            main_val = float(ls[7])*(1e16)
            env_val = float(ls[8])*(-1e16)
            xo, yo = int(ls[0]), int(ls[1])
            mid, cid, lid = int(ls[4]), int(ls[5]), self.layer_name_map[ls[6]]
            if filter_threshold is not None and env_val/main_val > filter_threshold:
                # if mid != cid or lid != 1:
                    if xo%80==0 and yo%80==0:
                        self.cases.append(Case(
                        xo,
                        yo,
                        int(ls[2]), 
                        int(ls[3]), 
                        mid,
                        cid,
                        lid, 
                        main_val,
                        env_val
                    ))
        print(f"{len(self)} cases loaded.")

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, index):
        # todo: replace magic number
        case = self.cases[index]
        x_o = case.x*160+180+case.x_offset
        y_o = case.y*160+180+case.y_offset
        idx_vec = self.idx_map[:, x_o:x_o+200, y_o:y_o+200]
        dens_vec = np.copy(self.dens_map[:, x_o:x_o+200, y_o:y_o+200])
        main_layer = np.copy(dens_vec[1])
        main_layer[idx_vec[1]==case.mid]+=1
        dens_vec[1, 20:180, 20:180] = main_layer[20:180, 20:180]
        dens_vec[case.lid][idx_vec[case.lid]==case.cid]*=-1
        dens_vec = np.pad(dens_vec, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), "constant")
        return torch.tensor(dens_vec, dtype=torch.float), torch.tensor([case.val], dtype=torch.float)

MainCase = namedtuple("MainCase", ["x_offset", "y_offset", "x", "y", "mid", "val"])
class MainDataset(Dataset):
    def __init__(self, layers, labels, np_dir, indicies, padding = 12, filter_threshold = None):
        self.window_size = 200
        lys = layers.split("_")
        self.layer_name_map = {}
        dens_map = []
        idx_map = []
        for idx, ly in enumerate(lys):
            self.layer_name_map[ly]=idx
            arrs = np.load(os.path.join(np_dir, f"{ly}.npz"))
            dens_map.append(arrs["img"])
            idx_map.append(arrs["idx"])
        padding_vec = [(0, 0), (self.window_size, self.window_size), (self.window_size, self.window_size)]
        self.dens_map = np.pad(np.array(dens_map), padding_vec, "constant")
        self.idx_map = np.pad(np.array(idx_map), padding_vec, "constant")
        self.dens_map.setflags(write=0)
        self.idx_map.setflags(write=0)
        self.cases = []
        self.padding = padding
        # with open(os.path.join(labels, f"{layers}_cp.txt"), "r") as f:
        #     f_content = f.read().strip().splitlines(keepends=False)
        if indicies is None:
            indicies = list(range(len(labels)))
        for i in indicies:
            line = labels[i]
            ls = line.split()
            self.cases.append(MainCase(
                int(ls[0]), 
                int(ls[1]), 
                int(ls[2]), 
                int(ls[3]), 
                int(ls[4]),
                float(ls[5])*(1e16)
            ))
        print(f"{len(self)} cases loaded.")

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, index):
        case = self.cases[index]
        x_o = case.x*160+180+case.x_offset
        y_o = case.y*160+180+case.y_offset
        idx_vec = self.idx_map[:, x_o:x_o+200, y_o:y_o+200]
        dens_vec = np.copy(self.dens_map[:, x_o:x_o+200, y_o:y_o+200])
        main_layer = np.copy(dens_vec[1])
        main_layer[idx_vec[1]==case.mid]+=1
        dens_vec[1, 20:180, 20:180] = main_layer[20:180, 20:180]
        dens_vec = np.pad(dens_vec, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), "constant")
        return torch.tensor(dens_vec, dtype=torch.float), torch.tensor([case.val], dtype=torch.float)



if __name__ == "__main__":
    target_layers = "POLY1_MET1_MET2"
    with open(os.path.join("../dataset/data", f"label/{target_layers}_env_train.txt"), "r") as f:
        train_content = f.read().strip().splitlines(keepends=False)
    cpds = CouplingDataset(target_layers, train_content, "../dataset/data", None, filter_threshold=0.025)
    print(len(cpds))