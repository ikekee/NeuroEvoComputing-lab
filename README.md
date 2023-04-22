# NeuroEvoComputing-lab
Educational implementation of neuroevolutionary ESP-algorithm for training recurrent neural network.
</br>
### Used packages
---
For working of algorithm and neural net model torch, numpy, pandas, scikit-learn and matplotlib were used.
</br>
### Features
---
- Training recurrent neural network using ESP-algorithm
- Visualisation of training results (metrics, model architecture, best model so far and etc.)
</br>

### ESP-Algorithm
---
ESP can be used to evolve any type of neural network that consists of a single hidden
layer, such as feed-forward, simple recurrent (Elman), fully recurrent, and second-order
networks
Used algorithm can be found in [this paper](https://www.cs.utexas.edu/users/nn/downloads/papers/gomez.phdtr03.pdf).
</br>

### Set up
---
1. Install required packages.
2. You can use .ipynb in repository for tests.
3. If you want to change neural net model architecture to train your own network, pay attention to neural net method "change_weights" and rewrite it for your own model.
