import os
import torch
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from eval_utils.oracle_metrics import OracleMetrics


def arguments():
    parser = argparse.ArgumentParser(description='FVA arguments.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id')
    parser.add_argument('--oracle-path', default='models/oracle.pth', type=str,
                        help='Oracle path')
    parser.add_argument('--exp-name', required=True, type=str,
                        help='Experiment Name')
    parser.add_argument('--output-path', required=True, type=str,
                        help='Results Path')

    return parser.parse_args()


# create dataset to read the counterfactual results images
class CFDataset():
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    def __init__(self, path, exp_name):

        self.images = []
        self.path = path
        self.exp_name = exp_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        for CL, CF in itertools.product(['CC', 'IC'], ['CCF']):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(path, 'Results', self.exp_name, CL, CF, 'CF'))]

    def __len__(self):
        return len(self.images)

    def switch(self, partition):
        if partition == 'C':
            LCF = ['CCF']
        elif partition == 'I':
            LCF = ['ICF']
        else:
            LCF = ['CCF', 'ICF']

        self.images = []

        for CL, CF in itertools.product(['CC', 'IC'], LCF):
            self.images += [(CL, CF, I) for I in os.listdir(osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF'))]

    def __getitem__(self, idx):
        CL, CF, I = self.images[idx]
        # get paths
        cl_path = osp.join(self.path, 'Original', 'Correct' if CL == 'CC' else 'Incorrect', I)
        cf_path = osp.join(self.path, 'Results', self.exp_name, CL, CF, 'CF', I)

        cl = self.load_img(cl_path)
        cf = self.load_img(cf_path)

        return cl, cf

    def load_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transform(img)


@torch.no_grad()
def compute_MNAC(oracle,
                 path,
                 exp_name):

    dataset = CFDataset(path, exp_name)

    cosine_similarity = torch.nn.CosineSimilarity()

    MNACS = []
    dists = []
    loader = data.DataLoader(dataset, batch_size=15,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    for cl, cf in tqdm(loader):
        cl = cl.to(device, dtype=torch.float)
        cf = cf.to(device, dtype=torch.float)
        _, cl_feat = oracle.oracle(cl)
        _, cf_feat = oracle.oracle(cf)
        d_cl = torch.sigmoid(cl_feat)
        d_cf = torch.sigmoid(cf_feat)
        MNACS.append(((d_cl > 0.5) != (d_cf > 0.5)).sum(dim=1).cpu().numpy())
        dists.append([d_cl.cpu().numpy(), d_cf.cpu().numpy()])

    return np.concatenate(MNACS), np.concatenate([d[0] for d in dists]), np.concatenate([d[1] for d in dists])


if __name__ == '__main__':

    args = arguments()

    # load oracle trained on vggface2 and fine-tuned on CelebA
    ORACLEPATH = args.oracle_path
    device = torch.device('cuda:' + args.gpu)
    oracle = OracleMetrics(weights_path=ORACLEPATH,
                           device=device)
    oracle.eval()
    A = 0
    
    results = compute_MNAC(oracle,
                           args.output_path,
                           args.exp_name)

    print('MNAC:', np.mean(results[0]))
