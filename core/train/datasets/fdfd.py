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
import ryaml
import h5py
from torch import Tensor
from torch.nn import functional as F
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms import InterpolationMode
from core.utils import resize_to_targt_size, print_stat
from thirdparty.ceviche.ceviche.constants import *
from core.utils import (
    Si_eps,
    SiO2_eps,
)

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
        data_dir: str = "raw",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        processed_dir: str = "processed",
        download: bool = False,
    ) -> None:
        self.device_type = device_type
        self.data_dir = data_dir
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
            processed_dir, self.data_dir, f"{self.train_filename}.yml"
        )
        processed_test_file = os.path.join(processed_dir, self.data_dir, f"{self.test_filename}.yml")
        if (
            os.path.exists(processed_training_file)
            and os.path.exists(processed_test_file)
        ):
            print("Data already processed")
            return

        device_id = self._load_dataset()
        (
            device_id_train,
            device_id_test,
        ) = self._split_dataset(
            device_id
        )  # split device files to make sure no overlapping device_id between train and test
        data_train, data_test = self._preprocess_dataset(
            device_id_train, device_id_test
        )
        self._save_dataset(
            data_train,
            data_test,
            processed_dir,
            self.data_dir,
            self.train_filename,
            self.test_filename,
        )

    def _load_dataset(self) -> List:
        ## do not load actual data here, too slow. Just load the filenames
        all_samples = [
                os.path.basename(i)
                for i in glob.glob(os.path.join(self.root, self.device_type, self.data_dir, f"{self.device_type}_*.h5"))
            ]
        total_device_id = []
        for filename in all_samples:
            device_id = filename.split("_")[1].split("-")[1]
            if device_id not in total_device_id:
                total_device_id.append(device_id)
        return total_device_id

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
        all_samples = [
                os.path.basename(i)
                for i in glob.glob(os.path.join(self.root, self.device_type, self.data_dir, f"{self.device_type}_*.h5"))
            ]
        filename_train = []
        filename_test = []
        for filename in all_samples:
            device_id = filename.split("_")[1].split("-")[1]
            opt_step = eval(filename.split("_")[-1].split(".")[0])
            if device_id in data_train: # only take the last step
                filename_train.append(filename)
            elif device_id in data_test: # only take the last step
                filename_test.append(filename)
        return filename_train, filename_test

    @staticmethod
    def _save_dataset(
        data_train: List,
        data_test: List,
        processed_dir: str,
        data_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, data_dir, f"{train_filename}.yml")
        processed_test_file = os.path.join(processed_dir, data_dir, f"{test_filename}.yml")

        with open(processed_training_file, "w") as f:
            yaml.dump(data_train, f)

        with open(processed_test_file, "w") as f:
            yaml.dump(data_test, f)

        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = (
            f"{self.train_filename}.yml" if train else f"{self.test_filename}.yml"
        )
        path_to_file = os.path.join(self.root, self.processed_dir, self.data_dir, filename)
        print(f"Loading data from {path_to_file}")
        with open(path_to_file, "r") as f:
            data = ryaml.load(f)
        return data

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        raise NotImplementedError
        return all([os.path.exists(os.path.join(self.root, self.data_dir, filename)) for filename in self.filenames])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        device_file = self.data[item]
        if "perturb" in device_file:
            input_slice = device_file.split("_perturbed_")[1].split("-")[1]
            wavelength = float(device_file.split("_perturbed_")[1].split("-")[2])
            mode = int(device_file.split("_perturbed_")[1].split("-")[3])
            temp = int(device_file.split("_perturbed_")[1].split("-")[-1][:-3])
        else:
            input_slice = device_file.split("_opt_step_")[1].split("-")[1]
            wavelength = float(device_file.split("_opt_step_")[1].split("-")[2])
            mode = int(device_file.split("_opt_step_")[1].split("-")[3])
            temp = int(device_file.split("_opt_step_")[1].split("-")[-1][:-3])
        path = os.path.join(self.root, self.device_type, self.data_dir, device_file)
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            eps_map = torch.from_numpy(f["eps_map"][()]).float() # sqrt the eps_map to get the refractive index TODO: I deleted the sqrt here, need to recheck the aux losses
            gradient = torch.from_numpy(f["gradient"][()]).float()
            fwd_field = torch.from_numpy(f["field_solutions"][()])
            adj_field = torch.from_numpy(f["fields_adj"][()])
            adj_src = torch.from_numpy(f["adj_src"][()])
            src_profile = torch.from_numpy(f["source_profile"][()])
            field_adj_normalizer = torch.from_numpy(f["field_adj_normalizer"][()]).float()
            A = {
                "entries_a": torch.from_numpy(f["A-entries_a"][()]),
                "indices_a": torch.from_numpy(f["A-indices_a"][()]),
            }
            s_params = {}
            design_region_mask = {}
            monitor_slice = {}
            ht_m = {}
            et_m = {}
            for key in keys:
                if key.startswith("s_params"):
                    value = f[key][()]
                    if isinstance(value, np.ndarray):
                        s_params[key] = torch.from_numpy(value).float()
                    else:  # Handle scalar values
                        s_params[key] = torch.tensor(value, dtype=torch.float32)
                elif key.startswith("design_region_mask"):
                    design_region_mask[key] = int(f[key][()])
                elif key.startswith("port_slice"):
                    data = f[key][()]
                    if isinstance(data, np.int64):
                        monitor_slice[key] = torch.tensor([data])
                    else:
                        monitor_slice[key] = torch.tensor(data)
                elif key.startswith("ht_m"):
                    ht_m[key] = torch.from_numpy(f[key][()])
                elif key.startswith("et_m"):
                    et_m[key] = torch.from_numpy(f[key][()])


        return input_slice, wavelength, mode, temp, eps_map, adj_src, gradient, fwd_field, s_params, src_profile, adj_field, field_adj_normalizer, design_region_mask, ht_m, et_m, monitor_slice, A, path

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class FDFDDataset:
    def __init__(
        self,
        device_type: str,
        root: str,
        data_dir: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        processed_dir: str = "processed",
    ):
        self.device_type = device_type
        self.root = root
        self.data_dir = data_dir
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
                data_dir=self.data_dir,
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
                data_dir=self.data_dir,
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