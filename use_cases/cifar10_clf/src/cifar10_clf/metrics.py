import pandas as pd
from torch import Tensor
from torchmetrics.classification import MulticlassConfusionMatrix


class HConfusionMatrix(MulticlassConfusionMatrix):
    def __init__(self, labels: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = labels

    def compute(self) -> pd.DataFrame:
        cm_tensor = super().compute()
        cm_df = pd.DataFrame(cm_tensor.cpu().numpy(), columns=self._labels, index=self._labels)
        cm_df.index.name = "class_label"
        return cm_df
