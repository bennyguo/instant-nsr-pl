import torch
from torchtyping import TensorType

from field_components.base_field_component import FieldComponent


class Embedding(FieldComponent):
    """Index into embeddings.
    # TODO: add different types of initializations

    Args:
        in_dim: Number of embeddings
        out_dim: Dimension of the embedding vectors
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        self.embedding = torch.nn.Embedding(self.in_dim, self.out_dim)
        # self.embedding.weight.data.fill_(0.)

    def mean(self, dim=0):
        """Return the mean of the embedding weights along a dim."""
        return self.embedding.weight.mean(dim)

    def forward(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Call forward

        Args:
            in_tensor: input tensor to process
        """
        return self.embedding(in_tensor)
