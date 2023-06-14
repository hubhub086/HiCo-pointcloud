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
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 512], True)
        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512], group_all=True)

        self.udm1 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=512, kernel_size=4),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=256-4+1)
            )
        self.udm2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=128-5+1)
        )

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
        # print(xc1.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_points.shape)
        xc2 = self.udm2(l2_points)
        xc2 = xc2.permute(2, 0, 1)
        # print(xc2.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        xc3 = l3_points.permute(2, 0, 1)
        # print(xc3.shape)

        return xc1, xc2, xc3


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    model = GetPointEmbeddingModel(normal_channel=False)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    xyz = torch.rand(32, 3, 512)
    model.forward(xyz)
    # summary(model, (3, 512), batch_size=512, device='cpu')



