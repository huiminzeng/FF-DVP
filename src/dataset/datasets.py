from PIL import Image

import torch
from torch.utils.data import Dataset

class myCeleba(Dataset):
    def __init__(self, data_path=None, sensitive_rows=None, target_rows=None, process=None):
        self.data_path = data_path
        self.sensitive_rows = sensitive_rows
        self.target_rows = target_rows
        self.process = process

    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.process(image_set)
        
        sensitive_feature = torch.tensor(self.sensitive_rows[idx]).long()
        targets = torch.tensor(self.target_rows[idx]).long()

        return image_tensor, sensitive_feature, targets