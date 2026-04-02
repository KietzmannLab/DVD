import torch
import torch.nn as nn
import torchvision.models as torchvision_models

import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Four-layer strided ConvNet → 64-d global feature → Linear classifier.
    Default img_size is now 112 so the first build works out-of-the-box
    for ecoset_square256* crops.
    """

    def __init__(
        self,
        n_input_channels: int = 3,
        num_classes: int = 1000,
        img_size: int = 112,          # <<-- changed from 224 to 112
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 7, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=2),               nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),               nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),               nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # infer flattened dimension once
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, img_size, img_size)
            feat_dim = self.features(dummy).shape[1]      # 1 600 when img_size=112
            assert feat_dim == 1600, (
                f"Unexpected feature dim {feat_dim} for img_size={img_size}. "
                "Check conv parameters!"
            )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


# Factory so `--arch custom_cnn` still works
def custom_cnn(pretrained: bool = False, **kwargs):
    if pretrained:
        raise NotImplementedError("No pretrained weights for custom_cnn.")
    return CustomCNN(**kwargs)