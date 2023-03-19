import torch
from torch import nn
import numpy as np


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, 1, batch_first=True,)
        self.fc = nn.Linear(hidden_dim, output_size, bias=False)


    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# n_epochs = 5
#
# for epoch in range(1, n_epochs + 1):
#
#     output, hidden = model(input_seq)
#     loss = criterion(output, target_seq.view(-1).long())
#     loss.backward()  # Does backpropagation and calculates gradients
#     optimizer.step()  # Updates the weights accordingly
#
#     if epoch % 10 == 0:
#         print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
#         print("Loss: {:.4f}".format(loss.item()))


class EspAlgorithm:
    def __init__(self, h: int, n: int, input_size: int, output_size: int):
        """

        h: Number of hidden layer neurons.
        n: Number of individuals for subpopulations.
        """
        self._number_of_hidden_neurons = h
        self._number_of_subpop = n
        self._input_size = input_size
        self._output_size = output_size
        self.init_model = Model(self._input_size,
                                self._output_size,
                                hidden_dim=h)

    def _run_trials(self):
        pass

    def _init_individual(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Inits an individual according to input tensor shape

        input_tensor: Tensor for making alike shaped individual.

        Returns a tensor with random numbers shaped like an input tensor.
        """
        new_tensor = torch.empty_like(input_tensor)
        return torch.nn.init.normal_(new_tensor)



if __name__ == '__main__':
    pass