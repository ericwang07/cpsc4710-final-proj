# 4710/5710 Final Project

Authors: Christian Choi, Jerry Tan, Eric Wang
This project investigates the effectiveness of norm-bounded and minimum norm adversarial attacks on ResNets.

## Usage

The provided jupyter notebook has all the code used for this project. The notebook has installation and import cells, and you can run from the top. If running locally, you will need to install robustbench, foolbox, and tinyimagenet. See `requirements.txt` for the packages.

## Code Structure
The notebook has labelled headers. We start by defining a evaluator class, which serves as the main object for loading model and dataset, performing and evaluating all three attacks (PGD, DF, SDF) through a unified interface. After loading CIFAR-10 and TinyImageNet datasets and the models, we evaluate the baseline model performances.

Next, we have the code for PGD experiments. Then, the source code for the DeepFool and SuperDeepFool attacks, see Alireza Abdollahpoorrostam, Mahed Abroshan, and Seyed-Mohsen Moosavi-Dezfooli. Revisiting DeepFool: generalization and improvement, 2024 and their repository for the implementation details.
