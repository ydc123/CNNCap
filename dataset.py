import numpy as np
import os
from torch.utils.data import Dataset

class GridCapData(Dataset):
    def __init__(self, data, indices, goal, filtered=True):
        assert goal in ['total', 'env']
        self.data = []
        for x in indices:
            item = data[x]
            self.in_channel = max([x[0] for x in item['x_compress_total']]) + 1
            if goal == 'total':
                self.data.append({
                    'x_compress': item['x_compress_total'],
                    'y': item['y_total'],
                    'total_cap': item['y_total'],
                    'dim': item['dim_total']
                })
            else:
                for item_env in item['env_data']:
                    if not filtered or item_env['y'] >= item['y_total'] * 0.01:
                        self.data.append({
                            'x_compress': item_env['x_compress'],
                            'y': item_env['y'],
                            'total_cap': item['y_total'],
                            'dim': item_env['dim']
                        })
        self.dim = self.data[0]['dim']
        self.scale = 8.854187817e-18 * 0.5
        

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x_data = np.zeros((self.in_channel, self.dim), dtype=np.float32)
        x_compress = self.data[index]['x_compress']
        y = self.data[index]['y'] / self.scale
        y_total = self.data[index]['total_cap'] / self.scale
        for layer_id, l, r, ratio in x_compress:
            l, r = round(l), round(r)
            x_data[layer_id, l:r+1] = ratio
        return x_data, np.array([y]), np.array([1]), np.array(y_total)