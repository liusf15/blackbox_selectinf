"""
Condition on selecting a single variable
"""

from blackbox_selectinf.usecase.Lasso import LassoClass
from blackbox_selectinf.learning.learning import learn_select_prob, get_weight, get_CI
import numpy as np
import argparse
import pickle
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from scipy.stats import norm
# from scipy.stats import binom
import matplotlib.pyplot as plt
import torch
from selectinf.distributions.discrete_family import discrete_family
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from argparse import Namespace


parser = argparse.ArgumentParser(description='one stage lasso')
parser.add_argument('--data_type', type=str, default='linear')
parser.add_argument('--basis_type', type=str, default='simple')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--lbd', type=float, default=6)
parser.add_argument('--indep', action='store_true', default=False)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--p', type=int, default=10)
parser.add_argument('--n_b', type=int, default=100)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--max_it', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--logname', type=str, default='log')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--thre', type=float, default=0.99)
parser.add_argument('--consec_epochs', type=int, default=5)
parser.add_argument('--nonnull', action='store_true', default=False)
args = parser.parse_args()
#
#
# args = Namespace(basis_type='simple', batch_size=100, consec_epochs=1, data_type='binary', epochs=1000,
#                  idx=14, indep=False, lbd=10, loadmodel=False, logname='log', max_it=1, modelname='model_',
#                  n=100, n_b=100, nonnull=True, nrep=1, ntrain=1000, p=10, savemodel=False, thre=0.99, verbose=True)


def main():
    n = args.n
    n_b = args.n_b
    ntrain = args.ntrain
    p = args.p
    beta = np.zeros(p)
    if args.nonnull:
        beta[:int(p/4)] = 5 / np.sqrt(n)
    data_type = args.data_type
    basis_type = args.basis_type
    lbd = args.lbd
    logs = [dict() for x in range(args.nrep)]

    j = args.idx
    for j in range(args.idx, args.idx + args.nrep):
        print("Starting simulation", j)
        logs[j - args.idx]['seed'] = j
        np.random.seed(j)
        if data_type == 'linear':
            X = np.random.randn(n, p)
            Y = X @ beta + np.random.randn(n)
        elif data_type == 'binary':
            X = np.random.randn(n, p)
            prob = 1 / (1 + np.exp(- X @ beta))
            Y = np.random.binomial(1, prob, n)
        else:
            raise AssertionError("invalid data_type")
        lassoClass = LassoClass(X, Y, lbd, data_type, basis_type)
        num_select = lassoClass.num_select
        beta_E = beta[lassoClass.E]
        E = lassoClass.E
        X_E = X[:, E]
        print("select:", np.sum(E), E)
        # D_M, D_0 = lassoClass.test_statistic(X, Y, return_D0=True)
        # print(lassoClass.logistic_KKT(D_M, D_0 * np.sqrt(n)))
        # print(lassoClass.linear_KKT(D_M, D_0 * np.sqrt(n)))

        # lee et al
        interval_lee = lassoClass.interval_lee()
        print('lee et al:', interval_lee)
        logs[j - args.idx]['interval_lee'] = interval_lee
        logs[j - args.idx]['covered_lee'] = []
        for i in range(num_select):
            if interval_lee[i, 0] <= beta_E[i] <= interval_lee[i, 1]:
                logs[j - args.idx]['covered_lee'].append(1)
            else:
                logs[j - args.idx]['covered_lee'].append(0)

        Z_data_np = lassoClass.basis(X, Y, basis_type)
        Z_data = torch.tensor(Z_data_np, dtype=torch.float)

        training_data = lassoClass.gen_train_data_select_single(ntrain=ntrain, n_b=n_b)
        Z_train = training_data[0]
        W_train = training_data[1]
        Gamma = training_data[2]
        #

        print("positive data:", np.sum(W_train, 0))
        print("fraction of positive data:", np.mean(W_train, 0))
        logs[j - args.idx]['ones'] = np.mean(W_train, 0)

        interval_nn = np.zeros([p, 2])
        covered_nn = np.zeros(p) - 1
        width_nn = np.zeros(p) - 1
        logs[j - args.idx]['pr_data'] = np.zeros(p) - 1
        if data_type == 'linear':
            theta_data = np.linalg.inv(X.T @ X) @ X.T @ Y
            Sigma = np.linalg.inv(X.T @ X)
        else:
            logi = LogisticRegression(penalty='none', fit_intercept=False, tol=1e-7)
            logi.fit(X, Y)
            theta_data = logi.coef_.squeeze()
            pr = 1 / (1 + np.exp(-X @ theta_data)).reshape(n)
            W = np.diag(pr * (1 - pr))
            Sigma = np.linalg.inv(X.T @ W @ X)

        fig, ax = plt.subplots(ncols=max(num_select, 2), figsize=(4 * num_select, 5))
        counter = 0
        for i in range(p):
            if not E[i]:
                continue
            # train the i-th
            print("inference for {}-th variable".format(i))
            net = None
            max_it = args.max_it
            for it in range(max_it):
                print("recursion", it)
                net, flag, pr_data = learn_select_prob(Z_train, W_train[:, i], Z_data=Z_data, net=net, thre=args.thre,
                                                       consec_epochs=args.consec_epochs, num_epochs=args.epochs,
                                                       batch_size=args.batch_size, verbose=args.verbose, print_every=100)
                if flag == 1:
                    print("Succeeded learning!")
                    break
                if it == max_it - 1:
                    break
                else:  # generate more data
                    print("generate more data")
                    training_data = lassoClass.gen_train_data_select_single(ntrain=ntrain, n_b=n_b)
                    Z_train_new = training_data[0]
                    W_train_new = training_data[1]
                    Z_train = np.concatenate([Z_train, Z_train_new])
                    W_train = np.concatenate([W_train, W_train_new])
                    print("fraction of positive data:", np.mean(W_train))
            logs[j - args.idx]['pr_data'][i] = pr_data
            # inference
            N_0 = Z_data_np - Gamma[:, i] * theta_data[i]
            target_var = Sigma[i, i]
            target_sd = np.sqrt(target_var)
            gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 101)
            target_theta = theta_data[i] + gamma_list
            weight_val = get_weight(net, target_theta.reshape(-1, 1), N_0, Gamma[:, None, i])
            ax[counter].plot(target_theta, weight_val)
            ax[counter].plot(np.ones(2) * theta_data[i], [0, 1])
            counter += 1
            interval_nn[i, :] = get_CI(target_theta, weight_val, target_var, theta_data[i])
            if interval_nn[i, 0] <= beta[i] <= interval_nn[i, 1]:
                covered_nn[i] = 1
            else:
                covered_nn[i] = 0
            width_nn[i] = interval_nn[i, 1] - interval_nn[i, 0]
        plt.savefig('{}_n_{}_p_{}_nb_{}_lbd_{}_{}.png'.format(args.logname, n, p, n_b, lbd, j))
        print(interval_nn[E, :])
        print(covered_nn[E])
        logs[j - args.idx]['interval_nn'] = interval_nn
        logs[j - args.idx]['width_nn'] = width_nn
        logs[j - args.idx]['covered_nn'] = covered_nn

        logs[j - args.idx]['beta_true'] = beta
        logs[j - args.idx]['E'] = E
        logs[j - args.idx]['beta_E'] = beta_E
        logs[j - args.idx]['beta_hat'] = theta_data

        #########################################
        # naive interval
        logs[j - args.idx]['covered_naive'] = []
        interval_naive = np.zeros([num_select, 2])
        target_sd = np.sqrt(np.diag(lassoClass.Sigma1))
        for i in range(num_select):
            interval_naive[i, :] = tuple(
                (norm.ppf(0.025) * target_sd[i], -norm.ppf(0.025) * target_sd[i]) + lassoClass.beta_ls[i])
            if interval_naive[i, 0] <= beta_E[i] <= interval_naive[i, 1]:
                logs[j - args.idx]['covered_naive'].append(1)
            else:
                logs[j - args.idx]['covered_naive'].append(0)
        logs[j - args.idx]['interval_naive'] = interval_naive
        logs[j - args.idx]['width_naive'] = interval_naive[:, 1] - interval_naive[:, 0]
        print(logs[j - args.idx]['covered_naive'])

        path = open('{}_n_{}_p_{}_nb_{}_lbd_{}_{}.pickle'.format(args.logname, n, p, n_b, lbd, j), 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()
        print(logs[j - args.idx])


if __name__ == "__main__":
    main()
