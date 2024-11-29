import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_item2key(self, filepath=None):
        item2key = {}
        if filepath is None:
            filepath = osp.join(self.config['data_path'], f'{self.dataset_name}.item2key')
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                item_id, key = line.strip().split('\t')
                item2key[item_id] = int(key)
        return item2key

    def load_plm_embedding(self):
        item2key = self.load_item2key()
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[item2key[token]]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class PretrainUniSRecDataset(UniSRecDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_suffix_aug = config['plm_suffix_aug']
        if self.plm_suffix_aug is not None:
            plm_embedding_weight_aug = self.load_plm_embedding(plm_suffix_aug=self.plm_suffix_aug)
            self.plm_embedding_aug = self.weight2emb(plm_embedding_weight_aug)

    def load_plm_embedding(self, plm_suffix_aug=None):
        with open(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets'), 'r') as file:
            dataset_names = file.read().strip().split(',')
        self.logger.info(f'Pre-training datasets: {dataset_names}')

        d2feat = []
        for dataset_name in dataset_names:
            if plm_suffix_aug is None:
                feat_path = osp.join(self.config['data_path'], f'../{dataset_name}/{dataset_name}.{self.plm_suffix}')
            else:
                feat_path = osp.join(self.config['data_path'], f'../{dataset_name}/{dataset_name}.{plm_suffix_aug}')
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
            d2feat.append(loaded_feat)

        d2item2key = []
        for dataset_name in dataset_names:
            item2key_path = osp.join(self.config['data_path'], f'../{dataset_name}/{dataset_name}.item2key')
            d2item2key.append(self.load_item2key(item2key_path))

        iid2domain = np.zeros((self.item_num, 1))
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            did, iid = token.split('-')
            loaded_feat = d2feat[int(did)]
            mapped_feat[i] = loaded_feat[d2item2key[int(did)][iid]]
            iid2domain[i] = int(did)
        self.iid2domain = torch.LongTensor(iid2domain)

        return mapped_feat
