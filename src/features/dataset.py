from torch.utils.data import Dataset


class AvitoDataset(Dataset):
    def __init__(self, features: list, targets: list):
        self.features = features
        self.targets = targets

    def __getitem__(self, item) -> dict:
        return {
            'feature': self.features[item],
            'target': self.targets[item]
        }

    def __len__(self) -> int:
        return len(self.targets)
