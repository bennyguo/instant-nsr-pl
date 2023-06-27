import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from field_components.embedding import Embedding

@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        if self.config.use_appearance_embedding:
            self.n_input_dims += self.config.appearance_embedding_dim
            self.embedding_appearance = Embedding(self.config.num_images, self.config.appearance_embedding_dim)
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network


    
    def forward(self, features, dirs, camera_indices, ray_indices, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        # with appearance embeddings
        if self.config.use_appearance_embedding:
            if self.training:
                appe_embd = self.embedding_appearance(camera_indices)
                appe_embd = appe_embd[ray_indices]
            elif camera_indices.nelement() > 0:
                appe_embd = self.embedding_appearance(camera_indices).repeat(features.size()[0], 1)
            else:
                appe_embd = self.embedding_appearance.mean(dim=0).repeat(features.size()[0], 1)
                if not self.config.use_average_appearance_embedding:
                    appe_embd *= 0
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, appe_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        else:
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        if self.config.use_appearance_embedding:
            self.n_input_dims += self.config.appearance_embedding_dim
            self.embedding_appearance = Embedding(self.config.num_images, self.config.appearance_embedding_dim)
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, camera_indices, ray_indices, *args):
        # with appearance embeddings
        if self.config.use_appearance_embedding:
            if self.training:
                appe_embd = self.embedding_appearance(camera_indices)
                appe_embd = appe_embd[ray_indices]
            elif camera_indices.nelement() > 0:
                appe_embd = self.embedding_appearance(camera_indices).repeat(features.size()[0], 1)
            else:
                appe_embd = self.embedding_appearance.mean(dim=0).repeat(features.size()[0], 1)
                if not self.config.use_average_appearance_embedding:
                    appe_embd *= 0
            network_inp = torch.cat([features.view(-1, features.shape[-1]), appe_embd], dim=-1)
        else:
            network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
