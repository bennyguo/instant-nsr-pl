import torch
import torch.nn as nn

from utils.misc import get_rank

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
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

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        return value
    
    @torch.no_grad()
    def export(self, export_config):
        return {}
