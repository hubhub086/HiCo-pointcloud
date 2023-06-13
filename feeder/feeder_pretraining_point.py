import time
import torch

import numpy as np

np.set_printoptions(threshold=np.inf)
import random
import pointnet2.provider as provider
from pointnet2.pointnet2_utils import farthest_point_sample
try:
    from feeder.pointcloud_augmentations import *
except:
    from pointcloud_augmentations import *


class Feeder_pointcloud(torch.utils.data.Dataset):
    """
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size  # default = 32
        self.input_representation = input_representation  # default = joint
        self.crop_resize = True
        self.l_ratio = l_ratio  # default = [0.1, 1]

        self.load_data(mmap)

        self.N, self.C, self.T, self.V = self.data.shape

        print(self.data.shape, len(self.number_of_frames))
        print("l_ratio", self.l_ratio)

    def load_data(self, mmap):
        # data: N C T V M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        if self.num_frame_path is not None:
            self.number_of_frames = np.load(self.num_frame_path)
        else:
            self.number_of_frames = np.ones(self.data.shape[0], dtype=np.int32) * 50

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get raw input

        # input: C, T, V
        data_numpy = np.array(self.data[index])

        number_of_frames = self.number_of_frames[index]
        # apply spatio-temporal augmentations to generate  view 1

        # temporal crop-resize
        # input_size=n时相当于随机选取连续n帧，实际帧数少于n时使用双线性插值补齐
        data_numpy_v1_crop = temporal_cropresize(data_numpy, number_of_frames, self.l_ratio,
                                                               self.input_size)
        data_numpy_v1_crop = data_numpy_v1_crop.transpose(1, 2, 0)  # CTV to TVC
        # print(f"data_numpy v1 crop = {data_numpy_v1_crop.shape}")
        # data_numpy_v1_crop = provider.normalize_data(data_numpy_v1_crop)

        # randomly select one of the spatial augmentations
        flip_prob = random.random()
        if flip_prob < 0.5:
            data_numpy_q1 = provider.shuffle_points(data_numpy_v1_crop)
            data_numpy_q2 = provider.random_scale_point_cloud(data_numpy_v1_crop)
        else:
            data_numpy_q1 = provider.shuffle_points(data_numpy_v1_crop)
            data_numpy_q2 = provider.rotate_perturbation_point_cloud(data_numpy_v1_crop)
        # apply spatio-temporal augmentations to generate  view 2

        # temporal crop-resize
        data_numpy_v2_crop = temporal_cropresize(data_numpy, number_of_frames, self.l_ratio,
                                                               self.input_size)
        data_numpy_v2_crop = data_numpy_v2_crop.transpose(1, 2, 0)  # CTV to TVC

        # data_numpy_v2_crop = provider.normalize_data(data_numpy_v2_crop)

        # randomly select  one of the spatial augmentations
        flip_prob = random.random()
        if flip_prob < 0.5:
            data_numpy_k1 = provider.jitter_point_cloud(data_numpy_v1_crop)
            data_numpy_k2 = provider.shift_point_cloud(data_numpy_v2_crop)
        else:
            data_numpy_k1 = provider.jitter_point_cloud(data_numpy_v1_crop)
            data_numpy_k2 = provider.random_point_dropout(data_numpy_v2_crop, max_dropout_ratio=0.4)

        # joint
        # query
        # TVC --> TCV
        # time-majored
        qc_joint = data_numpy_q1.transpose(0, 2, 1)
        qc_joint = qc_joint.astype('float32')
        # TVC --> TCV
        # space-majored
        qp_joint = data_numpy_q2.transpose(0, 2, 1)
        qp_joint = qp_joint.astype('float32')

        # key
        kc_joint = data_numpy_k1.transpose(0, 2, 1)
        kc_joint = kc_joint.astype('float32')
        kp_joint = data_numpy_k2.transpose(0, 2, 1)
        kp_joint = kp_joint.astype('float32')
        # print(f'qc_joint {qc_joint.shape}')

        return qc_joint, qp_joint, kc_joint, kp_joint


if __name__ == "__main__":
    dataset = Feeder_pointcloud(
        data_path='../HiCo-data/pr_dataset_pointcloud/train_data_point.npy',
        num_frame_path='../HiCo-data/pr_dataset_pointcloud/train_num_frame.npy',
        l_ratio=[0.1, 1],
        input_size=32,
        input_representation='joint',
        mmap=True
    )
    qc, qp, kc, kp = dataset[0]
    print(qc.shape)
    print(qp.shape)
    print(kc.shape)
    print(kp.shape)
