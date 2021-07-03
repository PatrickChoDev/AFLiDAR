import torch
import torch.nn as nn


class PointExtractor(nn.Module):
    def __init__(self, nb_in_ch, nb_out_ch, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(nb_in_ch, nb_out_ch, 1)
        self.bn1 = nn.BatchNorm1d(nb_out_ch, 1e-3, 0.01)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate, True)

    def forward(self, x):
        out = self.conv1(x.transpose(-1, -2))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out).transpose(-1, -2)
        return out


class PointFeatNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.MODULES = nn.Sequential()
        nb_in_ch = 4 if cfg.WITH_REFLEXTIVE else 3
        cfg.FEATURES[-1] -= nb_in_ch
        for i in range(len(cfg.FEATURES)):
            if i == 0:
                self.MODULES.add_module(
                    cfg.LAYER_NAME[i],
                    PointExtractor(
                        nb_in_ch,
                        cfg.FEATURES[i],
                        cfg.DROPOUT
                        if isinstance(cfg.DROPOUT, (int, float))
                        else cfg.DROPOUT[i],
                    ),
                )
                continue
            self.MODULES.add_module(
                cfg.LAYER_NAME[i],
                PointExtractor(
                    cfg.FEATURES[i - 1],
                    cfg.FEATURES[i],
                    cfg.DROPOUT
                    if isinstance(cfg.DROPOUT, (int, float))
                    else cfg.DROPOUT[i],
                ),
            )

    def forward(self, x):
        return torch.cat([x,self.MODULES(x)],-1)


class PointMAP2BEV(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.voxel_size = cfg.VOXEL_SIZE
        


if __name__ == "__main__":
    from config.config_parser import load_config

    model = PointFeatNet(
        load_config(
            "/home/patrick/Workspaces/AFLiDAR/data/config/baseline.yaml"
        ).MODEL.POINTFEAT
    )
    print(model)
    inp = torch.rand(4, 16384, 3)
    out = model(inp)
    print(out)
    print(out.shape)
