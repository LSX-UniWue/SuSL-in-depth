from typing import Dict

from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler


class MixedDataset(Dataset):
    def __init__(self, dataset_labelled: Dataset, dataset_unlabelled: Dataset) -> None:
        super().__init__()
        if len(dataset_labelled) < len(dataset_unlabelled):
            self.__min_dataset = dataset_labelled
            self.__max_dataset = dataset_unlabelled
        else:
            self.__min_dataset = dataset_unlabelled
            self.__max_dataset = dataset_labelled
        self.__min_sampler = None
        self.__reset_sampler()

    def __reset_sampler(self) -> None:
        self.__min_sampler = RandomSampler(
            data_source=self.__min_dataset, replacement=True, num_samples=len(self)
        ).__iter__()

    def __len__(self) -> int:
        return len(self.__max_dataset)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        try:
            sub_sample_id = next(self.__min_sampler)
        except StopIteration:
            self.__reset_sampler()
            sub_sample_id = next(self.__min_sampler)
        sub_sample = self.__min_dataset[sub_sample_id]
        sample = self.__max_dataset[index]
        return sub_sample | sample


class SemiUnsupervisedDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_unlabeled: Dataset,
        train_dataset_labeled: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 32,
    ):
        super().__init__()
        self.__batch_size = batch_size
        self.__train_dataset_unlabeled = train_dataset_unlabeled
        self.__train_dataset_labeled = train_dataset_labeled
        self.__validation_dataset = validation_dataset
        self.__test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MixedDataset(
                dataset_labelled=self.__train_dataset_labeled, dataset_unlabelled=self.__train_dataset_unlabeled
            ),
            batch_size=self.__batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.__validation_dataset, batch_size=self.__batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.__test_dataset, batch_size=self.__batch_size)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
