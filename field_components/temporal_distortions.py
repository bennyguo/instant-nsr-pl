# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Space distortions which occur as a function of time."""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType
import torch.nn.functional as F

from field_components.encodings import Encoding, NeRFEncoding
from field_components.mlp import MLP


class TemporalDistortion(nn.Module):
    """Apply spatial distortions as a function of time"""

    def forward(self, positions: TensorType["bs":..., 3], times: Optional[TensorType[1]]) -> TensorType["bs":..., 3]:
        """
        Args:
            positions: Samples to translate as a function of time
            times: times for each sample

        Returns:
            Translated positions.
        """


class TemporalDistortionKind(Enum):
    """Possible temporal distortion names"""

    DNERF = "dnerf"

    def to_temporal_distortion(self, config: Dict[str, Any]) -> TemporalDistortion:
        """Converts this kind to a temporal distortion"""
        if self == TemporalDistortionKind.DNERF:
            return DNeRFDistortion(**config)
        raise NotImplementedError(f"Unknown temporal distortion kind {self}")


class DNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        temporal_encoding: Encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
        spatial = False,
        num_classes = None,
    ) -> None:
        super().__init__()
        # self.position_encoding = position_encoding
        self.spatial = spatial
        self.num_classes = num_classes
        if self.spatial and self.num_classes is not None:
            assert self.num_classes > 0
            mlp_deforms = []
            """
            self.mlp_cam = MLP(
                in_dim=self.num_classes,
                out_dim=self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=128,
                with_bias=True,
            )
            """
            self.in_dim_mlp = 3 # self.position_encoding.get_out_dim() # self.num_classes
            for i in range(self.num_classes):
                mlp_deforms.append(
                        MLP(
                    in_dim=self.in_dim_mlp,
                    out_dim=3,
                    num_layers=mlp_num_layers,
                    layer_width=mlp_layer_width,
                    activation=nn.Tanh(),
                    out_activation=nn.Tanh(),
                ))
            self.mlp_deforms = nn.ModuleList(mlp_deforms)
        else:
            self.temporal_encoding = temporal_encoding
            self.in_dim_mlp = self.position_encoding.get_out_dim() + self.temporal_encoding.get_out_dim()
            self.mlp_deform = MLP(
                in_dim=self.in_dim_mlp,
                out_dim=3,
                num_layers=mlp_num_layers,
                layer_width=mlp_layer_width,
                skip_connections=skip_connections,
            )

    def forward(self, positions, times=None):
        if times is None:
            return None
        # pos_feature = self.position_encoding(positions)
        if self.spatial and self.num_classes is not None:
            # time_feature = F.one_hot(times, num_classes=self.num_classes)
            offsets = torch.zeros_like(positions)
            for i in range(self.num_classes):
                # if offsets[times == i].nelement() > 0:
                # offsets[times == i] = self.mlp_deforms[i](pos_feature[times == i]).type_as(positions)
                offsets[times == i] = self.mlp_deforms[i](positions[times == i]).type_as(positions)
            """ debug
            feat_p = self.position_encoding(torch.tensor([[0.1, 0.5, 0.9]]).cuda())
            offset1 = self.mlp_deforms[1](feat_p)
            offset2 = self.mlp_deforms[2](feat_p)
            from termcolor import colored
            print('\n', colored(offset1.data, 'green'), '\n', colored(offset2.data, 'blue'))
            """
            return offsets 
        else:
            time_feature = self.temporal_encoding(times)
            return self.mlp_deform(torch.cat([pos_feature, time_feature], dim=-1))
