from torch.utils.data import Dataset


class AvitoDataset(Dataset):
    def __init__(self, feature: list, target: list, vocab_to_ind: dict) -> None:
        self.vocab_to_ind = vocab_to_ind
        feature_transform = (lambda x:
                             self.vocab_to_ind[x] if x in vocab_to_ind
                             else vocab_to_ind['<UNK>']
                             )
        self.feature = [list(map(feature_transform, tokens)) for tokens in feature]
        self.target = target

    def __getitem__(self, item) -> dict:
        return {
            'feature': self.feature[item],
            'target': self.target[item],
        }

    def __len__(self) -> int:
        return len(self.target)
