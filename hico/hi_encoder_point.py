import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from pointnet2.pointnet2_cls_msg import GetPointEmbeddingModel


class HiEncoder(nn.Module):
    """Two branch hierarchical encoder with multi-granularity"""

    def __init__(self, t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size,
                 num_head,
                 num_layer,
                 granularity,
                 encoder,
                 ) -> None:
        super().__init__()
        self.d_model = hidden_size
        self.granularity = granularity
        self.encoder = encoder

        # temporal and spatial branch embedding layers
        self.t_embedding = GetPointEmbeddingModel(normal_channel=False)
        self.s_embedding = GetPointEmbeddingModel(normal_channel=False)

        # seq2seq encoders
        if encoder == "GRU":
            self.t_encoder = nn.GRU(input_size=self.d_model, hidden_size=self.d_model // 2,
                                    num_layers=num_layer, batch_first=True, bidirectional=True)
            self.s_encoder = nn.GRU(input_size=self.d_model, hidden_size=self.d_model // 2,
                                    num_layers=num_layer, batch_first=True, bidirectional=True)
        elif encoder == "LSTM":
            self.t_encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model // 2,
                                     num_layers=num_layer, batch_first=True, bidirectional=True)
            self.s_encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model // 2,
                                     num_layers=num_layer, batch_first=True, bidirectional=True)
        elif encoder == "Transformer":
            encoder_layer = TransformerEncoderLayer(self.d_model, num_head, self.d_model, batch_first=True)
            self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
            self.s_encoder = TransformerEncoder(encoder_layer, num_layer)
        else:
            raise ValueError("Unknown encoder!")

    def forward(self, xc, xp):
        # Given the time-majored domain input sequence xc
        # and the space-majored domain input sequence xp

        if self.encoder == "GRU" or self.encoder == "LSTM":
            self.t_encoder.flatten_parameters()
            self.s_encoder.flatten_parameters()

        ## embedding --> N*T*D
        # print(f"xc.shape = {xc.shape}")
        N, T, C, V = xc.shape
        # xc1, xc2, xc3 = self.t_embedding(xc[0, :, :, :].squeeze())  # temporal domain
        # xp1, xp2, xp3 = self.s_embedding(xp[0, :, :, :].squeeze())  # spatial domain
        # for n in range(1, N):
        #     xc1_n, xc2_n, xc3_n = self.t_embedding(xc[n, :, :, :])  # temporal domain
        #     xp1_n, xp2_n, xp3_n = self.s_embedding(xp[n, :, :, :])  # spatial domain
        #     xc1 = torch.cat([xc1, xc1_n], dim=0)
        #     xc2 = torch.cat([xc2, xc2_n], dim=0)
        #     xc3 = torch.cat([xc3, xp3_n], dim=0)
        #     xp2 = torch.cat([xp2, xp2_n], dim=0)
        #     xp1 = torch.cat([xp1, xp1_n], dim=0)
        #     xp3 = torch.cat([xp3, xp3_n], dim=0)
        xc1, xc2, xc3 = self.t_embedding(xc.reshape(N*T, C, V))  # temporal domain
        xp1, xp2, xp3 = self.s_embedding(xp.reshape(N*T, C, V))  # spatial domain
        xc1, xc2, xc3 = xc1.reshape(N, T, -1), xc2.reshape(N, T, -1), xc3.reshape(N, T, -1)
        xp1, xp2, xp3 = xp1.reshape(N, T, -1), xp2.reshape(N, T, -1), xp3.reshape(N, T, -1)


        ## max pooling --> N*1*D
        # xc1, xc2, xc3 = torch.max(xc1, dim=1)[0], torch.max(xc2, dim=1)[0], torch.max(xc3, dim=1)[0]
        # xp1, xp2, xp3 = torch.max(xp1, dim=1)[0], torch.max(xp2, dim=1)[0], torch.max(xp3, dim=1)[0]

        # Encoder
        vc = self.t_encoder(xc1)
        vp = self.s_encoder(xp1)
        vc = vc.amax(dim=1).unsqueeze(1)
        vp = vp.amax(dim=1).unsqueeze(1)
        for i in range(1, self.granularity):
            vc_i = self.t_encoder(eval(f"xc{i + 1}"))
            vp_i = self.s_encoder(eval(f"xp{i + 1}"))

            vc_i = vc_i.amax(dim=1).unsqueeze(1)
            vp_i = vp_i.amax(dim=1).unsqueeze(1)
            # print(vc_i.shape)
            vc = torch.cat([vc, vc_i], dim=1)
            vp = torch.cat([vp, vp_i], dim=1)
        return vc, vp


class PretrainingEncoder_point(nn.Module):
    """hierarchical encoder network + projectors"""

    def __init__(self,t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(PretrainingEncoder_point, self).__init__()

        self.d_model = hidden_size

        self.hi_encoder = HiEncoder(
            t_input_size, s_input_size,
            kernel_size, stride, padding, factor,
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # clip level feature projector
        self.clip_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # part level feature projector
        self.part_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # temporal domain level feature projector
        self.td_proj = nn.Sequential(
            nn.Linear(3*self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # spatial domain level feature projector
        self.sd_proj = nn.Sequential(
            nn.Linear(3*self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # instance level feature projector
        self.instance_proj = nn.Sequential(
            nn.Linear(2 * 3 * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, xc, xp):
        # we use concatenation as our feature fusion method

        # obtain clip and part level representations
        vc, vp = self.hi_encoder(xc, xp)

        # concatenate different granularity features as temproal and spatial domain representations
        vt = vc.reshape(vc.shape[0], -1)
        vs = vp.reshape(vp.shape[0], -1)

        # same for instance level representation
        vi = torch.cat([vt, vs], dim=1)

        # projection
        zc = self.clip_proj(vc)
        zp = self.part_proj(vp)

        zt = self.td_proj(vt)
        zs = self.sd_proj(vs)

        zi = self.instance_proj(vi)

        return zc, zp, zt, zs, zi

class DownstreamEncoder(nn.Module):
    """hierarchical encoder network + classifier"""

    def __init__(self, t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.hi_encoder = HiEncoder(
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # linear classifier
        self.fc = nn.Linear(2 * granularity * self.d_model, num_class)

    def forward(self, xc, xp, knn_eval=False):

        vc, vp = self.hi_encoder(xc, xp)

        vt = vc.reshape(vc.shape[0], -1)
        vs = vp.reshape(vp.shape[0], -1)

        vi = torch.cat([vt, vs], dim=1)

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return vi
        else:
            return self.fc(vi)


if __name__ == "__main__":
    xc = torch.rand(2, 32, 3, 4096)
    xp = torch.rand(2, 32, 3, 4096)
    # model = PretrainingEncoder(
    #     hidden_size=512,
    #     num_head=4,
    #     num_layer=1,
    #     granularity=3,
    #     encoder='Transformer',
    #     num_class=128,
    # )
    # qc, qp, qt, qs, qi = model(xc, xp)
    model = HiEncoder(
        t_input_size=None, s_input_size=None,
        kernel_size=None, stride=None, padding=None, factor=None,
        hidden_size=512,
        num_head=4,
        num_layer=1,
        granularity=3,
        encoder='Transformer'
    )
    model(xc, xp)
