import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.
    """
    return MLP(
        input_dim,
        [2048, 1024, 1024],
        output_dim,
        ["ReLU", "ReLU", "ReLU"],
        torch.nn.init.xavier_uniform_,
    )
