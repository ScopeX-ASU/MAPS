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
            processed_dir, f"{self.train_filename}.yml"
        )
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.yml")
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
        print(f"Loading data from {path_to_file}")
        with open(path_to_file, "r") as f:
            data = yaml.safe_load(f)
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
        path = os.path.join(self.root, self.device_type, self.data_dir, device_file)
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            orgion_size = torch.from_numpy(f["eps_map"][()]).float().size()
            # eps_map = resize_to_targt_size(torch.from_numpy(f["eps_map"][()]).float(), (200, 300))
            # gradient = resize_to_targt_size(torch.from_numpy(f["gradient"][()]).float(), (200, 300))
            eps_map = torch.from_numpy(f["eps_map"][()]).float() # sqrt the eps_map to get the refractive index TODO: I deleted the sqrt here, need to recheck the aux losses
            gradient = torch.from_numpy(f["gradient"][()]).float()
            field_solutions = {}
            s_params = {}
            adj_srcs = {}
            src_profile = {}
            fields_adj = {}
            field_normalizer = {}
            design_region_mask = {}
            ht_m = {}
            et_m = {}
            monitor_slice = {}
            As = {}
            for key in keys:
                if key.startswith("field_solutions"):
                    field = torch.from_numpy(f[key][()])
                    field_solutions[key] = field
                    # field = torch.view_as_real(field).permute(0, 3, 1, 2)
                    # field = resize_to_targt_size(field, (200, 300)).permute(0, 2, 3, 1)
                    # field_solutions[key] = torch.view_as_complex(field.contiguous())
                elif key.startswith("s_params"):
                    value = f[key][()]
                    if isinstance(value, np.ndarray):
                        s_params[key] = torch.from_numpy(value).float()
                    else:  # Handle scalar values
                        s_params[key] = torch.tensor(value, dtype=torch.float32)
                elif key.startswith("adj_src"):
                    adjoint_src = torch.from_numpy(f[key][()])
                    adj_srcs[key] = adjoint_src
                    # adjoint_src = torch.view_as_real(adjoint_src).permute(2, 0, 1)
                    # adjoint_src = resize_to_targt_size(adjoint_src, (200, 300)).permute(1, 2, 0)
                    # adj_srcs[key] = torch.view_as_complex(adjoint_src.contiguous())
                elif key.startswith("source_profile"):
                    source_profile = torch.from_numpy(f[key][()])
                    src_profile[key] = source_profile
                    # if key == "source_profile-wl-1.55-port-in_port_1-mode-1":
                    #     mode = source_profile[int(0.4 * source_profile.shape[0] / 2)]
                    #     mode = mode.unsqueeze(0).repeat(source_profile.shape[0], 1)
                    #     source_index = int(0.4 * source_profile.shape[0] / 2)
                    #     resolution = 2e-8
                    #     epsilon = Si_eps(1.55)
                    #     lambda_0 = 1.55e-6
                    #     k = (2 * torch.pi / lambda_0) * torch.sqrt(torch.tensor(epsilon))
                    #     x_coords = torch.arange(260).float()
                    #     distances = torch.abs(x_coords - source_index) * resolution
                    #     phase_shifts = (k * distances).unsqueeze(1)
                    #     mode = mode * torch.exp(1j * phase_shifts)
                    #     # mode = torch.view_as_real(mode).permute(2, 0, 1)
                    #     # mode = resize_to_targt_size(mode, (200, 300)).permute(1, 2, 0)
                    #     incident_key = key.replace("source_profile", "incident_field")
                    #     incident_field[incident_key] = mode
                    #     # incident_field[incident_key] = torch.view_as_complex(mode.contiguous())
                    # # source_profile = torch.view_as_real(source_profile).permute(2, 0, 1)
                    # # source_profile = resize_to_targt_size(source_profile, (200, 300)).permute(1, 2, 0)
                    # # src_profile[key] = torch.view_as_complex(source_profile.contiguous())
                elif key.startswith("fields_adj"):
                    field = torch.from_numpy(f[key][()])
                    fields_adj[key] = field
                    # field = torch.view_as_real(field).permute(0, 3, 1, 2)
                    # field = resize_to_targt_size(field, (200, 300)).permute(0, 2, 3, 1)
                    # fields_adj[key] = torch.view_as_complex(field.contiguous())
                elif key.startswith("field_adj_normalizer"):
                    field_normalizer[key] = torch.from_numpy(f[key][()]).float()
                elif key.startswith("design_region_mask"):
                    design_region_mask[key] = int(f[key][()])
                elif key.startswith("ht_m"):
                    ht_m[key] = torch.from_numpy(f[key][()])
                    ht_m[key+"-origin_size"] = torch.tensor(ht_m[key].shape)
                    ht_m[key] = torch.view_as_real(ht_m[key]).permute(1, 0).unsqueeze(0)
                    ht_m[key] = F.interpolate(ht_m[key], size=5000, mode='linear', align_corners=True)
                    ht_m[key] = ht_m[key].squeeze(0).permute(1, 0).contiguous()
                    ht_m[key] = torch.view_as_complex(ht_m[key])
                    # print("this is the dtype of the ht_m: ", ht_m[key].dtype, flush=True)
                elif key.startswith("et_m"):
                    et_m[key] = torch.from_numpy(f[key][()])
                    et_m[key+"-origin_size"] = torch.tensor(et_m[key].shape)
                    et_m[key] = torch.view_as_real(et_m[key]).permute(1, 0).unsqueeze(0)
                    et_m[key] = F.interpolate(et_m[key], size=5000, mode='linear', align_corners=True)
                    et_m[key] = et_m[key].squeeze(0).permute(1, 0).contiguous()
                    et_m[key] = torch.view_as_complex(et_m[key])
                    # print("this is the dtype of the et_m: ", et_m[key].dtype, flush=True)
                elif key.startswith("A-"):
                    As[key] = torch.from_numpy(f[key][()])
                elif key.startswith("port_slice"):
                    data = f[key][()]
                    # print("this is the key: ", key, type(data), data, flush=True)
                    # if isinstance(data, np.ndarray) and len(data.shape) == 1:
                    #     monitor_slice[key] = torch.tensor([data[0], data[-1] + 1])
                    if isinstance(data, np.int64):
                        monitor_slice[key] = torch.tensor([data])
                    else:
                        monitor_slice[key] = torch.tensor(data)
        return eps_map, adj_srcs, gradient, field_solutions, s_params, src_profile, fields_adj, field_normalizer, design_region_mask, ht_m, et_m, monitor_slice, As, path

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