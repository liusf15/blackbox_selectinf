# from importlib import reload
# import blackbox_selectinf.usecase.Lasso
# reload(blackbox_selectinf.usecase.Lasso)
# import blackbox_selectinf.learning.learning
# reload(blackbox_selectinf.learning.learning)
from blackbox_selectinf.usecase.Lasso import TwoStageLasso
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
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression


parser = argparse.ArgumentParser(description='two stage lasso')
parser.add_argument('--data_type', type=str, default='linear')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--lbd', type=float, default=30)
parser.add_argument('--indep', action='store_true', default=False)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--m', type=int, default=500)
parser.add_argument('--n_b', type=int, default=1000)
parser.add_argument('--m_b', type=int, default=500)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--logname', type=str, default='log_')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


def main():
    n = args.n
    m = args.m
    n_b = args.n_b
    m_b = args.m_b
    ntrain = args.ntrain
    p = 4
    beta = np.array([0, 0, 2, 2]) / np.sqrt(n)
    data_type = args.data_type
    lbd = args.lbd
    logs = [dict() for x in range(args.nrep)]

    for j in range(args.idx, args.idx + args.nrep):
        print("Starting simulation", j)
        logs[j - args.idx]['seed'] = j
        np.random.seed(j)
        if data_type == 'linear':
            X_1 = np.random.randn(n, p)
            Y_1 = X_1 @ beta + np.random.randn(n)
            X_2 = np.random.randn(m, p)
            Y_2 = X_2 @ beta + np.random.randn(m)
        elif data_type == 'binary':
            X_1 = np.random.randn(n, p)
            prob = 1 / (1 + np.exp(- X_1 @ beta))
            Y_1 = np.random.binomial(1, prob, n)
            X_2 = np.random.randn(m, p)
            prob_2 = 1 / (1 + np.exp(- X_2 @ beta))
            Y_2 = np.random.binomial(1, prob_2, m)
        else:
            raise AssertionError("invalid data_type")
        X = np.concatenate([X_1, X_2])
        Y = np.concatenate([Y_1, Y_2])
        print(X_2[0, :], Y_2[0])
        lassoClass = TwoStageLasso(X_1, Y_1, X_2, Y_2, lbd, data_type)
        num_select = lassoClass.num_select
        beta_E = beta[lassoClass.E]
        E = lassoClass.E
        print("beta_hat_1", lassoClass.beta_hat_1)
        training_data = lassoClass.gen_train_data(ntrain=ntrain, n_b=n_b, m_b=m_b, remove_D0=args.indep)
        Z_train = training_data['Z_train']
        W_train = training_data['W_train']
        gamma = training_data['gamma']
        Z_data = lassoClass.basis(X, Y)
        if args.indep:
            gamma_D0 = training_data['gamma_D0']
            # D0_train = training_data['D0_train']
            Z_data = Z_data - gamma_D0 @ lassoClass.D_0
            # Z_train = Z_train - D0_train @ gamma_D0.T
        logs[j - args.idx]['ones'] = np.mean(W_train)
        print(np.mean(W_train))
        print(E)
        # train
        net = learn_select_prob(Z_train, W_train, num_epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose, print_every=100)
        pr_data = net(torch.tensor(Z_data, dtype=torch.float))
        print('pr_data', pr_data)
        logs[j - args.idx]['pr_data'] = pr_data
        theta_data = lassoClass.test_statistic(X, Y)
        print("beta_hat", theta_data)
        N_0 = Z_data - gamma @ theta_data
        target_var = np.diag(lassoClass.Sigma1)
        target_sd = np.sqrt(target_var)
        gamma_list = np.linspace(-10 * target_sd, 10 * target_sd, 101)
        interval_nn = np.zeros([num_select, 2])
        logs[j - args.idx]['covered_nn'] = []
        weight_val = np.zeros([num_select, 101])
        for k in range(num_select):
            target_theta_k = theta_data[k] + gamma_list[:, k]
            # after correction
            Gamma_k = lassoClass.Sigma1[:, k] / lassoClass.Sigma1[k, k]
            target_theta = theta_data + np.outer(gamma_list[:, k], Gamma_k)
            # before
            # target_theta = np.tile(theta_data, [101, 1])
            # target_theta[:, k] = target_theta_k
            weight_val[k, :] = get_weight(net, target_theta, N_0, gamma)
            interval = get_CI(target_theta_k, weight_val[k, :], target_var[k], theta_data[k])
            interval_nn[k, :] = interval
            if interval_nn[k, 0] <= beta_E[k] <= interval_nn[k, 1]:
                logs[j - args.idx]['covered_nn'].append(1)
            else:
                logs[j - args.idx]['covered_nn'].append(0)
        logs[j - args.idx]['interval_nn'] = interval_nn
        logs[j - args.idx]['width_nn'] = interval_nn[:, 1] - interval_nn[:, 0]

        ##################################################
        # check learning
        count = 0
        nb = 50
        pval = [[] for x in range(num_select)]
        for ell in range(int(nb / np.mean(W_train))):
            idx_1 = np.random.choice(n + m, n_b, replace=True)
            X_1_b = X[idx_1, :]
            Y_1_b = Y[idx_1]
            idx_2 = np.random.choice(n + m, m_b, replace=True)
            X_2_b = X[idx_2, :]
            Y_2_b = Y[idx_2]
            X_b = np.concatenate([X_1_b, X_2_b])
            Y_b = np.concatenate([Y_1_b, Y_2_b])
            if not np.all(lassoClass.select(X_1_b, Y_1_b) == lassoClass.sign):
                continue
            else:
                count += 1
                d_M = lassoClass.test_statistic(X_b, Y_b)
                observed_target = d_M
                for k in range(num_select):
                    target_theta_k = d_M[k] + gamma_list[:, k]
                    # after correction
                    Gamma_k = lassoClass.Sigma1[:, k] / lassoClass.Sigma1[k, k]
                    target_theta_0 = d_M + np.outer(gamma_list[:, k], Gamma_k)
                    # before
                    # target_theta_0 = np.tile(d_M, [101, 1])
                    # target_theta_0[:, k] = target_theta_k
                    weight_val_0 = get_weight(net, target_theta_0, N_0, gamma)
                    weight_val_2 = weight_val_0 * norm.pdf((target_theta_0[:, k] - observed_target[k]) / target_sd[k])
                    exp_family = discrete_family(target_theta_0.reshape(-1), weight_val_2.reshape(-1))
                    hypothesis = theta_data[k]
                    pivot = exp_family.cdf((hypothesis - observed_target[k]) / target_var[k], x=observed_target[k])
                    pivot = 2 * min(pivot, 1 - pivot)
                    pval[k].append(pivot)
                if count == nb:
                    break
        pval = np.array(pval)
        logs[j - args.idx]['pval'] = pval
        logs[j - args.idx]['false_rej'] = np.sum(pval <= 0.05, 1) / count
        print(pval)
        print("reject:", np.sum(pval <= 0.05, 1) / count)

        ##################################################
        # lee et al
        if data_type == 'linear':
            g = glm.gaussian(X_1, Y_1)
        else:
            g = glm.logistic(X_1, Y_1)
        model = lasso.lasso(g, lbd)
        model.fit()
        summ = model.summary(compute_intervals=True)
        interval_lee = np.zeros([num_select, 2])
        interval_lee[:, 0] = summ.lower_confidence
        interval_lee[:, 1] = summ.upper_confidence
        print(interval_lee)
        logs[j - args.idx]['interval_lee'] = interval_lee
        logs[j - args.idx]['covered_lee'] = []
        for k in range(num_select):
            if interval_lee[k, 0] <= beta_E[k] <= interval_lee[k, 1]:
                logs[j - args.idx]['covered_lee'].append(1)
            else:
                logs[j - args.idx]['covered_lee'].append(0)

        ##################################################
        # interval true

        U = np.array(summ['upper_trunc'])
        L = np.array(summ['lower_trunc'])
        interval_true = np.zeros([num_select, 2])
        weight_val_true = np.zeros([num_select, 101])
        logs[j - args.idx]['covered_true'] = []
        fig, ax = plt.subplots(ncols=num_select, figsize=(4 * num_select, 5))
        for k in range(num_select):
            target_val = theta_data[k] + gamma_list[:, k]
            sigma_k = np.sqrt(m / n * target_var[k])  # what scaling?
            weight_val_true[k, :] = norm.cdf((U[k] - target_val) / sigma_k) - norm.cdf(
                (L[k] - target_val) / sigma_k)
            interval_true[k, :] = get_CI(target_val, weight_val_true[k, :], target_var[k], theta_data[k])
            if interval_true[k, 0] <= beta_E[k] <= interval_true[k, 1]:
                logs[j - args.idx]['covered_true'].append(1)
            else:
                logs[j - args.idx]['covered_true'].append(0)
            # plot
            if num_select == 1:
                plt.plot(target_val, weight_val[k, :], label='nn')
                plt.plot(target_val, weight_val_true[k, :], label='truth', ls='--')
                plt.legend()
            else:
                ax[k].plot(target_val, weight_val[k, :], label='nn')
                ax[k].plot(target_val, weight_val_true[k, :], label='truth', ls='--')
                ax[k].legend()
        plt.savefig(args.logname + str(j) + '.png')
        logs[j - args.idx]['interval_true'] = interval_true
        logs[j - args.idx]['width_true'] = interval_true[:, 1] - interval_true[:, 0]

        ##################################################
        # stage 2 interval
        logs[j - args.idx]['covered_2'] = []
        interval_2 = np.zeros([num_select, 2])
        if data_type == 'binary':
            pr_hat = 1 / (1 + np.exp(-X_2[:, E] @ lassoClass.beta_ls_2)).reshape(-1)
            W = np.diag(pr_hat * (1 - pr_hat))
        else:
            W = np.identity(m)
        var_2 = np.diag(np.linalg.inv(X_2[:, E].T @ W @ X_2[:, E]))
        for k in range(num_select):
            interval_2[k, :] = tuple(
                (norm.ppf(0.025) * np.sqrt(var_2[k]), -norm.ppf(0.025) * np.sqrt(var_2[k])) + lassoClass.beta_ls_2[k])
            if interval_2[k, 0] <= beta_E[k] <= interval_2[k, 1]:
                logs[j - args.idx]['covered_2'].append(1)
            else:
                logs[j - args.idx]['covered_2'].append(0)

        logs[j - args.idx]['interval_2'] = interval_2
        logs[j - args.idx]['width_2'] = interval_2[:, 1] - interval_2[:, 0]

        ##################################################
        # naive interval
        logs[j - args.idx]['covered_naive'] = []
        interval_naive = np.zeros([num_select, 2])

        for k in range(num_select):
            interval_naive[k, :] = tuple(
                (norm.ppf(0.025) * target_sd[k], -norm.ppf(0.025) * target_sd[k]) + lassoClass.beta_ls[k])
            if interval_2[k, 0] <= beta_E[k] <= interval_2[k, 1]:
                logs[j - args.idx]['covered_naive'].append(1)
            else:
                logs[j - args.idx]['covered_naive'].append(0)

        logs[j - args.idx]['interval_naive'] = interval_naive
        logs[j - args.idx]['width_naive'] = interval_naive[:, 1] - interval_naive[:, 0]

        logs[j - args.idx]['beta_true'] = beta
        logs[j - args.idx]['E'] = E
        logs[j - args.idx]['beta_E'] = beta_E
        logs[j - args.idx]['beta_hat'] = theta_data

        path = open(args.logname + str(j) + '.pickle', 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()
    print(logs)


if __name__ == "__main__":
    main()
