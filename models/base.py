import torch
import torch.nn as nn

from pytorch_lightning.utilities.rank_zero import _get_rank

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = _get_rank()
        self.setup()
        if self.config.get('weights', None):
            self.load_state_dict(torch.load(self.config.weights))
    
    def setup(self):
        raise NotImplementedError
    
    def update_step(self, epoch, global_step):
        pass
    
    def train(self, mode=True):
        return super().train(mode=mode)
    
    def eval(self):
        return super().eval()
    
    def regularizations(self, out):
        return {}
