"""\nData loading and preprocessing utilities.\n"""

from torch.utils.data import Dataset


class DataClass(Dataset):
    """Class `DataClass`.
    Args:
        Dataset:
    Returns:
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.items = kwargs

    def __len__(self):
        """\n        Function `__len__`.\n    \n        Returns: Description.\n"""
        any_item = list(self.items.values())[0]
        return len(any_item)

    def __getitem__(self, idx):
        """\n        Function `__getitem__`.\n    \n        Args:\n        idx: Description.\n    \n        Returns: Description.\n"""
        return {k: v[idx] for k, v in self.items.items()}
