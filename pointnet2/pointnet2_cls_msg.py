import torch.nn as nn
import torch.nn.functional as F
from pointnet2.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class GetPointEmbeddingModel(nn.Module):
    """ 基于Pointnet++的点云编码网络，将不同颗粒度的特征通过udm编码到同一的点云表征形式
    args:
        ...
    return:
        xc1, xc2, xc3 -- 分别代表由浅到深的点云表征向量
    """
    def __init__(self, normal_channel=True):
        super(GetPointEmbeddingModel, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 512], True)

        # self.udm1 = nn.Sequential(
        #         nn.Conv1d(in_channels=320, out_channels=512, kernel_size=4),
        #         nn.ReLU(),
        #         nn.MaxPool1d(kernel_size=128-4+1)
        #     )
        # self.udm2 = nn.Sequential(
        #     nn.Conv1d(in_channels=640, out_channels=512, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=64 - 5 + 1)
        # )

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        xc1 = self.udm1(l1_points)
        xc1 = xc1.permute(2, 0, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        xc2 = self.udm2(l2_points)
        xc2 = xc2.permute(2, 0, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        xc3 = l3_points.permute(2, 0, 1)

        return xc1, xc2, xc3


if __name__ == "__main__":
    import torch
    model = GetPointEmbeddingModel(normal_channel=False)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    xyz = torch.rand(32, 3, 4096)
    model.forward(xyz)



