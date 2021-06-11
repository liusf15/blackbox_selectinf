from blackbox_selectinf.usecase.Lasso import LassoClass
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import numpy as np
import argparse
import pickle
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
from selectinf.distributions.discrete_family import discrete_family


n = 1000
n_b = n
p = 4
input_dim = n * (p + 1)
beta = np.array([0, 0, 5, 5]) / np.sqrt(n)
seed = 1
np.random.seed(seed)
X = np.random.randn(n, p)
Y = X @ beta + np.random.randn(n)
lbd = 30
data_type = 'linear'
lassoClass = LassoClass(X, Y, lbd, data_type)
print(lassoClass.sign)

