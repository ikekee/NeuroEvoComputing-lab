from typing import List

import torch
from torch import nn
import numpy as np
from sklearn.metrics import log_loss


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
        self._subpopulations = [[] for _ in range(h)]
        self.init_model = Model(self._input_size,
                                self._output_size,
                                hidden_dim=h)

    def _run_trials(self, x, y):
        while (self._neurons_num_of_trials < self.trials_num).all():
            # random indexes of neurons in subpopulations
            neurons_indexes = np.random.randint(low=0,
                                                high=self._number_in_subpop,
                                                size=(self._number_of_hidden_neurons,))
            # Getting together all needed chosen neurons and inserting them in neural net
            gathered_neurons = self._subpopulations[
                np.arange(len(self._subpopulations)), neurons_indexes]
            model = Model(self._input_size, self._output_size, self._number_of_hidden_neurons)
            model.change_weights(gathered_neurons)
            # Calculating loss function
            model_out = model.forward(x)
            loss = log_loss(y, model_out)
            # Appending number of trials and loss for each neuron in subpopulation
            for i, index in enumerate(neurons_indexes):
                self._neurons_num_of_trials[i][index] += 1
                self._neurons_cumulative_loss[i][index] += loss
            # Checking for best loss
            if loss < self._best_loss:
                self._best_loss = loss
                self._best_model = model

    def _check_stagnation(self):
        pass

    def _recombination(self):
        self._average_cumulative_losses = np.sort(
            self._neurons_cumulative_loss / self._neurons_num_of_trials)
        for neuron_position in range(self._number_of_hidden_neurons):
            for j in range(int(self._number_in_subpop / 4)):
                # Subpopulation must be big
                random_index = np.random.randint(low=0, high=j)
                neurons = self._subpopulations[neuron_position]
                crossovered_neurons = self._crossover([neurons[j], neurons[random_index],
                                                       neurons[j * 2 + 1], neurons[j * 2 + 2]])
                neurons[j], neurons[random_index], neurons[j * 2 + 1], neurons[j * 2 + 2] = \
                    crossovered_neurons[0], crossovered_neurons[1], \
                    crossovered_neurons[2], crossovered_neurons[3]
            for j in range(int(self._number_in_subpop / 2), self._number_in_subpop):
                self._subpopulations[neuron_position][j] = \
                    self._mutate(self._subpopulations[neuron_position][j])

    def _crossover(self, neurons: List) -> List:
        # Gets 4 neurons: first by the loop, random neuron from quartile, *2 by the loop, *2 + 1 by the loop
        crosspoint = np.random.randint(low=0,
                                       high=len(neurons[0]) - 1)
        neurons[2][:crosspoint] = neurons[0][:crosspoint]
        neurons[3][crosspoint:] = neurons[1][crosspoint:]
        return neurons

    def _mutate(self, neuron):
        if np.random.rand() < self.mutation_rate:
            gen_index_to_mutate = np.random.randint(low=0,
                                                    high=len(neuron) - 1)
            neuron[gen_index_to_mutate] = np.random.standard_cauchy() / 10
        return neuron

        Returns a tensor with random numbers shaped like an input tensor.
        """
        new_tensor = torch.empty_like(input_tensor)
        return torch.nn.init.normal_(new_tensor)



if __name__ == '__main__':
    pass