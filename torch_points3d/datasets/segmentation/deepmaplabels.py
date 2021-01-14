import os
import os.path as osp
import torch
import tqdm

from torch_points3d.datasets.segmentation import labels_pb2
from torch_geometric.data import InMemoryDataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.core.data_transform import SaveOriginalPosId


class DeepmapLabels(InMemoryDataset):
    label_ids = {
        "Vegetation": 0,
        "Building walls": 1,
        "Vertical poles": 2,
        "Road": 3,
        "Movable objects": 4,
        "Unlabeled": 255,
    }

    weight_classes = torch.tensor([1., 1., 1., 1., 1., 0.])

    def __init__(self, root, transform=None, pre_transform=None):
        super(DeepmapLabels, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        path = osp.join(
            os.getcwd(),
            'torch_points3d/datasets/segmentation/deepmaplabels.txt')
        with open(path) as f:
            file_names = f.read().splitlines()
        assert len(file_names) == 99
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass  # POC100 is already in this repo.

    def process(self):
        # Read data into huge `Data` list.
        data_raw_list = []
        data_list = []

        # Map from label to [0, 1, ... num_classes).
        label_idx = {
            sorted(self.label_ids.values())[i]: i
            for i in range(len(self.label_ids))
        }

        for raw_path in self.raw_paths:
            labels = labels_pb2.Labels()
            with open(raw_path, "rb") as f:
                labels.ParseFromString(f.read())

            pos = []
            y = []
            for point in labels.points:
                pos.append([point.x, point.y, point.z])
                assert point.label in label_idx
                y.append(point.label)
            pos = torch.tensor(pos)
            y = torch.tensor(y).type(torch.long)
            data = Data(pos=pos, y=y)
            data = SaveOriginalPosId()(data)
            data_raw_list.append(
                data.clone() if self.pre_transform is not None else data)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DeepmapLabelsDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = DeepmapLabels(
            self._data_path,
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )
        self.test_dataset = DeepmapLabels(
            self._data_path,
            pre_transform=self.pre_transform,
            transform=self.test_transform,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self,
                                   wandb_log=wandb_log,
                                   use_tensorboard=tensorboard_log)
