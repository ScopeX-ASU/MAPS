"""
Description:
this code is define the dataset class used for the ML4FDFD model
I changed it from NeurOLight MMI dataset

need to accomodate different types of devices
"""

import os
import numpy as np
import torch
import glob
import yaml
import h5py
from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms import InterpolationMode

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["FDFD", "FDFDDataset"]


class FDFD(VisionDataset):
    url = None
    filename_suffix = "fields_epsilon_mode.pt"
    train_filename = "training"
    test_filename = "test"
    folder = "fdfd"

    def __init__(
        self,
        device_type: str,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        processed_dir: str = "processed",
        download: bool = False,
    ) -> None:
        self.device_type = device_type
        self.processed_dir = processed_dir
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.train_filename = self.train_filename
        self.test_filename = self.test_filename

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError(
        #         "Dataset not found or corrupted." + " You can use download=True to download it"
        #     )

        self.process_raw_data()
        self.data = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        # no matter the preprocessed file exists or not, we will always process the data, won't take too much time
        processed_training_file = os.path.join(
            processed_dir, f"{self.train_filename}.yml"
        )
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.yml")
        if (
            os.path.exists(processed_training_file)
            and os.path.exists(processed_test_file)
        ):
            print("Data already processed")
            return

        filenames = self._load_dataset()
        (
            filenames_train,
            filenames_test,
        ) = self._split_dataset(
            filenames
        )  # split device files to make sure no overlapping device_id between train and test
        data_train, data_test = self._preprocess_dataset(
            filenames_train, filenames_test
        )
        self._save_dataset(
            data_train,
            data_test,
            processed_dir,
            self.train_filename,
            self.test_filename,
        )

    def _load_dataset(self) -> List:
        ## do not load actual data here, too slow. Just load the filenames
        all_samples = [
                os.path.basename(i)
                for i in glob.glob(os.path.join(self.root, self.device_type, f"test2_{self.device_type}_*.h5"))
            ]
        return all_samples

    def _split_dataset(self, filenames) -> Tuple[List, ...]:
        from sklearn.model_selection import train_test_split
        print("this is the train ratio: ", self.train_ratio, flush=True)
        print("this is the length of the filenames: ", len(filenames), flush=True)
        (
            filenames_train,
            filenames_test,
        ) = train_test_split(
            filenames,
            train_size=int(self.train_ratio * len(filenames)),
            random_state=42,
        )
        print(
            f"training: {len(filenames_train)} device examples, "
            f"test: {len(filenames_test)} device examples"
        )
        return (
            filenames_train,
            filenames_test,
        )

    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        data_train: List,
        data_test: List,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.yml")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.yml")

        with open(processed_training_file, "w") as f:
            yaml.dump(data_train, f)

        with open(processed_test_file, "w") as f:
            yaml.dump(data_test, f)

        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = (
            f"{self.train_filename}.yml" if train else f"{self.test_filename}.yml"
        )
        path_to_file = os.path.join(self.root, self.processed_dir, filename)
        with open(path_to_file, "r") as f:
            data = yaml.safe_load(f)
        return data

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        raise NotImplementedError
        return all([os.path.exists(os.path.join(self.root, "raw", filename)) for filename in self.filenames])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        device_file = self.data[item]
        with h5py.File(os.path.join(self.root, self.device_type, device_file), "r") as f:
            keys = list(f.keys())
            eps_map = torch.from_numpy(f["eps_map"][()]).float()
            gradient = torch.from_numpy(f["gradient"][()]).float()
            field_solutions = {}
            s_params = {}
            adj_srcs = {}
            src_profile = {}
            fields_adj = {}
            field_normalizer = {}
            design_region_mask = {}
            for key in keys:
                if key.startswith("field_solutions"):
                    field_solutions[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("s_params"):
                    s_params[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("adj_src"):
                    adj_srcs[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("source_profile"):
                    src_profile[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("fields_adj"):
                    fields_adj[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("field_adj_normalizer"):
                    field_normalizer[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("design_region_mask"):
                    design_region_mask[key] = int(f[key][()])
        return eps_map, adj_srcs, gradient, field_solutions, s_params, src_profile, fields_adj, field_normalizer, design_region_mask

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class FDFDDataset:
    def __init__(
        self,
        device_type: str,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        processed_dir: str = "processed",
    ):
        self.device_type = device_type
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(f"Only support test_ratio from (0, 1), but got {test_ratio}")
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.processed_dir = processed_dir

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = FDFD(
                self.device_type,
                self.root,
                train=True,
                download=False,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                processed_dir=self.processed_dir,
            )

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1] > 0.99999:
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[:train_len+valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = FDFD(
                self.device_type,
                self.root,
                train=False,
                download=False,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                processed_dir=self.processed_dir,
            )

            self.data = test

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_fdfd():
    import pdb

    # pdb.set_trace()
    fdfd = FDFD(device_type="metacoupler", root="../../data", download=False, processed_dir="metacoupler")
    print(len(fdfd.data))
    fdfd = FDFD(device_type="metacoupler", root="../../data", train=False, download=False, processed_dir="metacoupler")
    print(len(fdfd.data))
    fdfd = FDFDDataset(
        device_type="metacoupler",
        root="../../data",
        split="train",
        test_ratio=0.1,
        train_valid_split_ratio=[0.9, 0.1],
        processed_dir="metacoupler",
    )
    print(len(fdfd))


if __name__ == "__main__":
    test_fdfd()