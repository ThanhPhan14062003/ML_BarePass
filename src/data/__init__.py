from torch.utils.data import Dataset, DataLoader
import torch

class BrainCancerDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.y[idx]