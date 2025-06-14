from torch.utils.data import Dataset


class DataClass(Dataset):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.items = kwargs

    def __len__(self):
        any_item = list(self.items.values())[0]
        return len(any_item)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.items.items()}
