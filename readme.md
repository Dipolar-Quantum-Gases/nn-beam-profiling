# Measuring Laser Beams with a Neural Network

<img src="https://github.com/Dipolar-Quantum-Gases/nn-beam-profiling/blob/master/imgs/thumbnail.jpg?raw=true" alt="drawing" width="150"/> <img src="https://github.com/Dipolar-Quantum-Gases/nn-beam-profiling/blob/master/imgs/thumbnailexp.png?raw=true" alt="drawing" width="150"/>

This is code associated with the paper "[Measuring Laser Beams with a Neural Network](https://doi.org/10.1364/AO.443531)." An ArXiv version of the paper is also available [here](https://arxiv.org/abs/2202.07801).

Currently the repo has two Google Colab notebooks along with supporting Python code.

The [first Colab notebook](https://colab.research.google.com/github/Dipolar-Quantum-Gases/nn-beam-profiling/blob/master/Explore_the_Dataset.ipynb) shows how to download and use the simulated and experimental datasets from the paper. The datasets are located in an Oxford University Research Archive and can be found [here](https://doi.org/10.5287/bodleian:JbDXrnQN1).

The [second Colab notebook](https://colab.research.google.com/github/Dipolar-Quantum-Gases/nn-beam-profiling/blob/master/Neural_Network_Beam_Profiling_Tutorial.ipynb) demonstrates how to create a neural network in Detectron2 that simultaneously detects and measures laser beams from the dataset images.