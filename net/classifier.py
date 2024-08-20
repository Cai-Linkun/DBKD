import torch.nn as nn
from collections import OrderedDict


class Classifier(nn.Module):
    def __init__(self, num_features, num_classes, droprate=0.5):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("__classifier_fc", nn.Linear(num_features, num_features)),
                    ("__classifier_dr", nn.Dropout(droprate)),
                    (
                        "classifier_weight",  # for visual
                        nn.Linear(num_features, num_classes),
                    ),
                ]
            )
        )

    def forward(self, x):
        x = self.model(x)
        return x
