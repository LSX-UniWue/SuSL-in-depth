from typing import Sequence, Dict

from torch import Tensor, LongTensor
from torch.utils.data import Subset, Dataset


class DatasetFacade(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.__subset = Subset(dataset=dataset, indices=indices)

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample, _ = self.__subset[index]
        return {"x_u": sample, "x_u_target": sample.detach().clone()}

    def __len__(self) -> int:
        return len(self.__subset)


class LabeledDatasetFacade(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], class_mapper: LongTensor | int) -> None:
        super().__init__()
        self.__subset = Subset(dataset=dataset, indices=indices)
        if isinstance(class_mapper, int):
            from torch import arange

            self.__class_mapper = arange(class_mapper).long()
        else:
            self.__class_mapper = class_mapper

    def __getitem__(self, index) -> Dict[str, Tensor]:
        sample, label = self.__subset[index]
        return {"x_l": sample, "x_l_target": sample.detach().clone(), "y_l": self.__class_mapper[label]}

    def __len__(self) -> int:
        return len(self.__subset)
