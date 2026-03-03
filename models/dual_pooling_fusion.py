import torch
import torch.nn as nn

from models.encoder import MelEncoder

class DualMelFusion(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Normal mel → MAX
        self.encoder_max = MelEncoder(pool_type="max")

        # Reverse mel → MIN
        self.encoder_min = MelEncoder(pool_type="min")

        self.feature_dim = 16384 * 2

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, mel, mel_comp):

        feat_max = self.encoder_max(mel)
        feat_min = self.encoder_min(mel_comp)

        fused = torch.cat([feat_max, feat_min], dim=1)

        return self.classifier(fused)