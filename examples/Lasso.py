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
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression


parser = argparse.ArgumentParser(description='one stage lasso')
parser.add_argument('--data_type', type=str, default='linear')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--lbd', type=float, default=30)
parser.add_argument('--indep', action='store_true', default=False)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--n_b', type=int, default=600)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--ntrain', type=int, default=5000)
parser.add_argument('--logname', type=str, default='log_')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


def main():
    n = args.n
    n_b = args.n_b
    ntrain = args.ntrain
    p = 4
    beta = np.array([0, 0, 2, 2]) / np.sqrt(n)
    data_type = args.data_type
    lbd = args.lbd
    logs = [dict() for x in range(args.nrep)]
    # if data_type == 'binary':
    #     N = 2000
    #     beta_indep = np.zeros([2, 2, 2, 2, p])
    #     for j in range(1, 16):
    #         ss = "{0:b}".format(j)
    #         E_tmp = [False] * 4
    #         E_tmp[-len(ss):] = np.array([q == '1' for q in ss])
    #         idx = np.zeros(4)
    #         idx[-len(ss):] = [int(q) for q in ss]
    #         idx = np.array(idx, dtype=int)
    #         tmp = 0
    #         for i in range(N):
    #             X = np.random.randn(n, p)
    #             prob = 1 / (1 + np.exp(- X @ beta))
    #             Y = np.random.binomial(1, prob, n)
    #             logreg = LogisticRegression(fit_intercept=False)
    #             logreg.fit(X[:, E_tmp], Y)
    #             beta_new = logreg.coef_.squeeze()
    #             tmp += beta_new / N
    #         beta_indep[idx[0], idx[1], idx[2], idx[3], E_tmp] = tmp

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
        lassoClass = LassoClass(X, Y, lbd, data_type)
        num_select = lassoClass.num_select
        beta_E = beta[lassoClass.E]
        E = lassoClass.E
        # if data_type == 'linear':
        #     # beta_sub = np.linalg.inv(X[:, E].T @ X[:, E]) @ X[:, E].T @ X @ beta # for fixed design
        #     beta_sub = beta_E  # for random design
        # else:
        #     idx_E = np.zeros(4, dtype=int)
        #     idx_E[E] = 1
        #     beta_sub = beta_indep[idx_E[0], idx_E[1], idx_E[2], idx_E[3], E]
        # logs[j - args.idx]['beta_sub'] = beta_sub

        training_data = lassoClass.gen_train_data(ntrain=ntrain, n_b=n_b, remove_D0=args.indep)
        Z_train = training_data['Z_train']
        W_train = training_data['W_train']
        gamma = training_data['gamma']
        Z_data = lassoClass.basis(X, Y)
        if args.indep:
            gamma_D0 = training_data['gamma_D0']
            # D0_train = training_data['D0_train']
            Z_data = Z_data - gamma_D0 @ lassoClass.D_0
            # Z_train = Z_train - D0_train @ gamma_D0.T  # already did this in gen_train_
        logs[j - args.idx]['ones'] = np.mean(W_train)
        print(np.mean(W_train))
        print(E)

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

        # train
        net = learn_select_prob(Z_train, W_train, Z_data=torch.tensor(Z_data, dtype=torch.float), num_epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)
        pr_data = net(torch.tensor(Z_data, dtype=torch.float))
        # print('pr_data', pr_data)
        logs[j - args.idx]['pr_data'] = pr_data
        theta_data = lassoClass.test_statistic(X, Y)
        N_0 = Z_data - gamma @ theta_data
        target_var = np.diag(lassoClass.Sigma1)
        target_sd = np.sqrt(target_var)
        gamma_list = np.linspace(-3 * target_sd, 3 * target_sd, 101)
        interval_nn = np.zeros([num_select, 2])
        logs[j - args.idx]['covered_nn'] = []
        weight_val = np.zeros([num_select, 101])
        for k in range(num_select):
            Gamma_k = lassoClass.Sigma1[:, k] / lassoClass.Sigma1[k, k]
            target_theta_k = theta_data[k] + gamma_list[:, k]
            ######## after correction
            print(theta_data.shape, Gamma_k.shape, gamma_list[:, k].shape)
            print(np.tile(theta_data, [101, 1]).shape)
            target_theta = theta_data + np.outer(gamma_list[:, k], Gamma_k)
            ######## before
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
            idx_b = np.random.choice(n, n_b, replace=True)
            X_b = X[idx_b, :]
            Y_b = Y[idx_b]
            if not np.all(lassoClass.select(X_b, Y_b) == lassoClass.sign):
                continue
            else:
                count += 1
                d_M = lassoClass.test_statistic(X_b, Y_b)
                observed_target = d_M
                for k in range(num_select):
                    target_theta_k = d_M[k] + gamma_list[:, k]
                    target_theta_0 = np.tile(d_M, [101, 1])
                    target_theta_0[:, k] = target_theta_k
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
        # logs[j - args.idx]['pval'] = pval
        logs[j - args.idx]['false_rej'] = np.sum(pval <= 0.05, 1) / count
        # print(pval)
        # print("reject:", np.sum(pval <= 0.05, 1) / count)

        ##################################################
        # true interval
        D_0 = lassoClass.D_0
        prob_gamma = np.zeros([num_select, 101])
        interval_true = np.zeros([num_select, 2])
        logs[j - args.idx]['covered_true'] = []
        fig, ax = plt.subplots(ncols=num_select, figsize=(4 * num_select, 5))
        for i in range(num_select):
            e_i = np.zeros(num_select)
            e_i[i] = 1
            for jj in range(101):
                if data_type == 'linear':
                    prob_gamma[i, jj] = lassoClass.linear_KKT(theta_data + gamma_list[jj, i] * e_i, D_0 * np.sqrt(n))
                else:
                    prob_gamma[i, jj] = lassoClass.logistic_KKT(theta_data + gamma_list[jj, i] * e_i, D_0 * np.sqrt(n))
            D_M_gamma = theta_data + np.outer(gamma_list[:, i], e_i)
            interval_true[i, :] = get_CI(D_M_gamma[:, i], prob_gamma[i, :], target_var[i], theta_data[i])
            if interval_true[i, 0] <= beta_E[i] <= interval_true[i, 1]:
                logs[j - args.idx]['covered_true'].append(1)
            else:
                logs[j - args.idx]['covered_true'].append(0)
            # plot
            if num_select == 1:
                plt.plot(D_M_gamma[:, i], weight_val[i, :], label='nn')
                plt.plot(D_M_gamma[:, i], prob_gamma[i, :], label='truth', ls='--')
                plt.legend()
            else:
                ax[i].plot(D_M_gamma[:, i], weight_val[i, :], label='nn')
                ax[i].plot(D_M_gamma[:, i], prob_gamma[i, :], label='truth', ls='--')
                ax[i].legend()
        plt.savefig(args.logname + str(j) + '.png')

        print('interval_true', interval_true)
        logs[j - args.idx]['interval_true'] = interval_true
        logs[j - args.idx]['width_true'] = interval_true[:, 1] - interval_true[:, 0]

        logs[j - args.idx]['beta_true'] = beta
        logs[j - args.idx]['E'] = E
        logs[j - args.idx]['beta_E'] = beta_E
        logs[j - args.idx]['beta_hat'] = theta_data

        #########################################
        # naive interval
        logs[j - args.idx]['covered_naive'] = []
        interval_naive = np.zeros([num_select, 2])
        for i in range(num_select):
            interval_naive[i, :] = tuple((norm.ppf(0.025) * target_sd[i], -norm.ppf(0.025) * target_sd[i]) + lassoClass.beta_ls[i])
            if interval_naive[i, 0] <= beta_E[i] <= interval_naive[i, 1]:
                logs[j - args.idx]['covered_naive'].append(1)
            else:
                logs[j - args.idx]['covered_naive'].append(0)
        logs[j - args.idx]['interval_naive'] = interval_naive

        path = open(args.logname + str(j) + '.pickle', 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()
        print(logs[j - args.idx])


if __name__ == "__main__":
    main()
