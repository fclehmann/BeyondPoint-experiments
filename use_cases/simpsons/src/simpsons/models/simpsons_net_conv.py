import torch.nn as nn
import math


class SimpsonsNetConv(nn.Sequential):

    def __init__(self, input_shape: tuple[int, int, int], num_classes: int):
        H, W, C = input_shape

        red_size = lambda x: math.floor((x - 2) / 2)

        dim_size = red_size(red_size(red_size(H))) * red_size(red_size(red_size(W))) * 256

        super().__init__(
            # (H, W, C)
            nn.Conv2d(
                in_channels=C,
                out_channels=32,
                kernel_size=(3, 3),
                padding="same"
            ),
            # (H, W, 32)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
            ),
            # (H-2, W-2, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # ((H-2)/2, (W-2)/2, 32)
            nn.Dropout(p=0.2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding="same"
            ),
            # ((H-2)/2, (W-2)/2, 64)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
            ),
            # ((H-2)/2-2, (W-2)/2-2, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # (((H-2)/2-2)/2, ((W-2)/2-2)/2, 64)
            nn.Dropout(p=0.2),

            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=(3, 3),
                padding="same"
            ),
            # (((H-2)/2-2)/2, ((W-2)/2-2)/2, 256)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
            ),
            # (((H-2)/2-2)/2-2, ((W-2)/2-2)/2-2, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # ((((H-2)/2-2)/2-2)/2, (((W-2)/2-2)/2-2)/2, 256)
            nn.Dropout(p=0.2),

            nn.Flatten(),
            nn.Linear(
                in_features=dim_size,
                out_features=1024
            ),
            # (1024,)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
