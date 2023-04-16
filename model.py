from typing import List, Dict, Optional

import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def visualize_metric_per_epoch(metric: List[float], metric_name: str):
    x = range(len(metric))
    y = metric
    plt.figure(figsize=(15, 10))
    plt.grid(True)
    plt.xlabel('Номер итерации')
    if metric_name == 'loss':
        plt.ylabel('Значение функции потерь')
    if metric_name == 'accuracy':
        plt.ylabel('Значение точности')
    if metric_name == 'hidden':
        plt.ylabel('Количество нейронов на скрытом слое')
    plt.plot(x, y)
    plt.savefig(f'{metric_name}.png')


def save_model(model: nn.Module, path: str):
    torch.save(model, path)


def load_model(path) -> nn.Module:
    return torch.load(path)


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.input_size = input_size
        self.output_size = output_size

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, 1, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden()
        # Passing in the input and hidden state into the model and obtaining outputs
        rnn_out, hidden = self.rnn(x, hidden)

        fc_out = self.fc(rnn_out)
        out = self.sigmoid(fc_out)
        out = out.contiguous().view(-1)

        return out, hidden

    def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros((1, 1, self.hidden_dim), dtype=torch.double)
        return hidden

    def evaluate_model(self, x: np.ndarray, y: np.ndarray) -> float:
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = torch.unsqueeze(x, dim=0)
        model_out, _ = self.forward(x)
        loss = log_loss(y, model_out)
        return loss

    def change_weights(self, gathered_neurons: np.ndarray):
        self.rnn.weight_ih_l0 = torch.nn.parameter.Parameter(
            torch.from_numpy(
            gathered_neurons[:, :self.input_size]), requires_grad=False)
        self.rnn.weight_hh_l0 = torch.nn.parameter.Parameter(
            torch.from_numpy(
            gathered_neurons[:, self.input_size:-self.output_size]),
            requires_grad=False)
        self.fc.weight = torch.nn.parameter.Parameter(
            torch.transpose(torch.from_numpy(
            gathered_neurons[:, -self.output_size:]), 0, 1),
            requires_grad=False)

    def evaluate_metrics(self, x: np.ndarray, y: np.ndarray) -> float:
        torch_x = torch.from_numpy(x)
        torch_x = torch.unsqueeze(torch_x, dim=0)
        model_out, _ = self.forward(torch_x)
        model_out = torch.where(model_out >= 0.5, 1.0, 0.0)
        accuracy = accuracy_score(y, model_out.numpy())
        return accuracy


class EspAlgorithm:
    def __init__(self,
                 h: int,
                 n: int,
                 x: pd.DataFrame,
                 y: pd.Series,
                 b: int = 3,
                 trials_num: int = 10.0,
                 mutation_rate: float = 0.3,
                 threshold: float = 1.0):
        """
        h: Number of hidden layer neurons.
        n: Number of individuals for subpopulations.
        b: number of generations before burst mutation is invoked.
        """
        self._number_of_hidden_neurons = h
        self._number_in_subpop = n
        self._generations_before_burst = b
        self.x = x.to_numpy()
        self.y = y.to_numpy()
        self.trials_num = trials_num
        self.mutation_rate = mutation_rate
        self.threshold = threshold
        self._input_size = self.x.shape[1]
        self._output_size = 1
        self._neurons_cumulative_loss = np.zeros((h, n))
        self._neurons_num_of_trials = np.zeros((h, n))
        self._subpopulations = np.random.rand(h, n, self._input_size + h + self._output_size)
        self._best_loss = None
        self._unchanged_generations_num = 0
        self._burst_mutations_in_row = 0
        self._best_model = None
        self._best_neurons = None

    def _run_trials(self, x, y):
        while (self._neurons_num_of_trials < self.trials_num).any():
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
            loss = model.evaluate_model(x, y)
            if self._best_loss is None:
                self._best_loss = loss
            # Appending number of trials and loss for each neuron in subpopulation
            for i, index in enumerate(neurons_indexes):
                self._neurons_num_of_trials[i][index] += 1
                self._neurons_cumulative_loss[i][index] += loss
            # Checking for best loss
            if loss < self._best_loss:
                self._best_loss = loss
                self._best_model = model
                self._best_neurons = gathered_neurons
                self._unchanged_generations_num = 0

    def _check_stagnation(self):
        if self._unchanged_generations_num >= self._generations_before_burst:
            if self._burst_mutations_in_row == 2:
                self._adapt_network_size()
            else:
                self._burst_mutate()
                self._burst_mutations_in_row += 1
                return
        self._burst_mutations_in_row = 0

    def _recombination(self):
        self._average_cumulative_losses = -np.sort(-
            self._neurons_cumulative_loss / self._neurons_num_of_trials)
        for neuron_position in range(self._number_of_hidden_neurons):
            for j in range(int(self._number_in_subpop / 4)):
                # Subpopulation must be big
                random_index = np.random.randint(low=0, high=j + 1)
                neurons = self._subpopulations[neuron_position]
                crossovered_neurons = self._crossover([neurons[j], neurons[random_index],
                                                       neurons[j * 2 + 1], neurons[j * 2 + 2]])
                neurons[-j-1], neurons[-random_index-1], neurons[-j * 2 - 2], neurons[-j * 2 - 3] = \
                    crossovered_neurons[0], crossovered_neurons[1], \
                    crossovered_neurons[2], crossovered_neurons[3]
            for j in range(int(self._number_in_subpop / 2),
                           self._number_in_subpop):
                self._subpopulations[neuron_position][j] = \
                    self._mutate(self._subpopulations[neuron_position][j])

    def _crossover(self, neurons: List[np.ndarray]) -> List[np.ndarray]:
        # Gets 4 neurons: first by the loop, random neuron from quartile, *2 by the loop, *2 + 1 by the loop
        crosspoint = np.random.randint(low=0,
                                       high=len(neurons[0]) - 1)
        neurons[2][:crosspoint] = neurons[0][:crosspoint]
        neurons[3][crosspoint:] = neurons[1][crosspoint:]
        return neurons

    def _mutate(self, neuron: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.mutation_rate:
            gen_index_to_mutate = np.random.randint(low=0,
                                                    high=len(neuron) - 1)
            neuron[gen_index_to_mutate] = np.random.standard_cauchy() / 10
        return neuron

    def _burst_mutate(self):
        neuron_len = self._input_size + self._output_size + self._number_of_hidden_neurons
        for neuron_position in range(self._number_of_hidden_neurons):
            self._subpopulations[neuron_position] =\
                np.array([self._best_neurons[neuron_position] + np.random.standard_cauchy(neuron_len)
                          for _ in range(self._number_in_subpop)])


    def _adapt_network_size(self):
        for neuron_position in range(self._number_of_hidden_neurons):
            smaller_model = Model(
                self._input_size,
                self._output_size,
                self._number_of_hidden_neurons - 1)
            smaller_best_neurons = np.delete(self._best_neurons, neuron_position, axis=0)
            smaller_best_neurons = np.delete(
                smaller_best_neurons, self._input_size + neuron_position, axis=1)
            smaller_model.change_weights(smaller_best_neurons)
            smaller_model_loss = smaller_model.evaluate_model(self.x, self.y)

            if smaller_model_loss < self._best_loss * self.threshold:
                self._number_of_hidden_neurons -= 1
                self._subpopulations = np.delete(self._subpopulations, neuron_position, axis=0)
                self._subpopulations = np.delete(self._subpopulations,
                                                 self._input_size + neuron_position,
                                                 axis=2)
                self._best_model = smaller_model
                self._best_loss = smaller_model_loss
                self._best_neurons = smaller_best_neurons
                return

        new_subpopulation = np.random.rand(1, self._number_in_subpop, self._input_size + self._number_of_hidden_neurons + self._output_size)
        self._subpopulations = np.append(self._subpopulations, new_subpopulation, axis=0)
        self._number_of_hidden_neurons += 1
        position_to_insert = self._input_size + self._number_of_hidden_neurons
        self._subpopulations = \
            np.insert(self._subpopulations,
                      [position_to_insert],
                      np.random.rand(self._number_of_hidden_neurons, self._number_in_subpop, 1), axis=2)
        self._best_model = Model(self._input_size, self._output_size, self._number_of_hidden_neurons)
        new_best_neurons = np.insert(self._best_neurons,
                                     [self._input_size + self._number_of_hidden_neurons],
                                     np.random.rand(self._number_of_hidden_neurons - 1, 1), axis=1)

        new_best_neurons = np.append(
            new_best_neurons,
            [self._subpopulations[-1][np.random.randint(0, self._number_in_subpop)]],
            axis=0)
        self._best_neurons = new_best_neurons
        self._best_model.change_weights(new_best_neurons)
        self._best_loss = self._best_model.evaluate_model(self.x, self.y)

    def run_alg(self, goal_loss: float) -> Optional[Dict[str, List[float]]]:
        history = {'loss': [],
                   'models': [],
                   'accuracy': [],
                   'models_hidden_size': []}

        while True:
            try:
                self._neurons_cumulative_loss = np.zeros((self._number_of_hidden_neurons, self._number_in_subpop))
                self._neurons_num_of_trials = np.zeros((self._number_of_hidden_neurons, self._number_in_subpop))
                beginning_best_loss = self._best_loss
                self._run_trials(self.x, self.y)
                print(self._number_of_hidden_neurons, self._best_loss)
                history['loss'].append(self._best_loss)
                history['models'].append(self._best_model)
                history['models_hidden_size'].append(self._number_of_hidden_neurons)
                history['accuracy'].append(self._best_model.evaluate_metrics(self.x, self.y))
                if beginning_best_loss == self._best_loss:
                    self._unchanged_generations_num += 1
                self._check_stagnation()
                self._recombination()
                if self._best_loss < goal_loss:
                    return history
            except KeyboardInterrupt:
                return history


if __name__ == '__main__':
    pass