from typing import Any, Dict, Optional, List, Type

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from hydra.utils import instantiate

from .ab_data_processor import ABDataProcessor


class ABDataModule(LightningDataModule):

    def __init__(
        self,
        train_dataset: Type[Dataset],
        val_dataset: Type[Dataset],
        test_dataset: Optional[Type[Dataset]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.ab_data_processor = ABDataProcessor(size=256)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if stage == "fit":
            self.train_dataset = self.hparams.train_dataset()
            self.val_dataset = self.hparams.val_dataset()

            if self.hparams.test_dataset is not None:
                self.test_dataset = self.hparams.test_dataset()

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [self.ab_data_processor.preprocess_sample(sample) for sample in batch]

        A_image = [sample["A_image"] for sample in batch if "A_image" in sample]
        B_image = [sample["B_image"] for sample in batch if "B_image" in sample]

        A_size = [sample["A_size"] for sample in batch if "A_size" in sample]
        B_size = [sample["B_size"] for sample in batch if "B_size" in sample]

        output = dict()

        if len(A_image) > 0:
            output["A_image"] = torch.stack(A_image)
            output["A_size"] = A_size

        if len(B_image) > 0:
            output["B_image"] = torch.stack(B_image)
            output["B_size"] = B_size

        return output

    def create_dataloader(self, dataset: Dataset, shuffle: bool) -> Dataset:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self.create_dataloader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self.create_dataloader(self.val_dataset, True)

    def test_dataloader(self) -> DataLoader[Any]:
        return self.create_dataloader(self.val_dataset, True)
