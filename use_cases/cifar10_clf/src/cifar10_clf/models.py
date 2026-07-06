import logging
from typing import Optional
import torchvision
import torch

_logger = logging.getLogger(__name__)


class VGG11BN(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float,
        weights_init_seed: Optional[int] = None,
        initial_seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = torchvision.models.vgg11_bn(
            weights=None, num_classes=num_classes, dropout=dropout, init_weights=False
        )

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, num_classes),
        )

        # Set torch seed to config.weights_init_seed
        if weights_init_seed is not None:
            _logger.info(
                f"Setting torch seed to config.weights_init_seed={weights_init_seed}..."
            )
            torch.manual_seed(weights_init_seed)
            torch.cuda.manual_seed(weights_init_seed)
            torch.cuda.manual_seed_all(weights_init_seed)

        # Init weights
        _logger.info(f"Init weights...")

        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        # Set torch seed back to config.seed
        if weights_init_seed is not None and initial_seed is not None:
            _logger.info(f"Setting torch seed back to config.seed={initial_seed}...")
            torch.manual_seed(initial_seed)
            torch.cuda.manual_seed(initial_seed)
            torch.cuda.manual_seed_all(initial_seed)

    def forward(self, x: torch.Tensor):
        return self.model(x)
