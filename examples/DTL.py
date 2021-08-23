from blackbox_selectinf.usecase.DTL import DropTheLoser
from blackbox_selectinf.learning.learning import (learn_select_prob, get_weight, get_CI)
import DTL_vae
import numpy as np
import argparse
import pickle
from regreg.smooth.glm import glm
from selectinf.algorithms import lasso
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
from selectinf.distributions.discrete_family import discrete_family
from argparse import Namespace


parser = argparse.ArgumentParser(description='DTL')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--selection', type=str, default='mean')
parser.add_argument('--uc', type=float, default=2)
parser.add_argument('--basis_type', type=str, default='naive')
parser.add_argument('--indep', action='store_true', default=False)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--m', type=int, default=500)
parser.add_argument('--n_b', type=int, default=1000)
parser.add_argument('--m_b', type=int, default=500)
parser.add_argument('--nrep', type=int, default=1)
parser.add_argument('--savemodel', action='store_true', default=False)
parser.add_argument('--modelname', type=str, default='model_')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--ntrain', type=int, default=5000)
parser.add_argument('--logname', type=str, default='log_')
parser.add_argument('--loadmodel', action='store_true', default=False)
parser.add_argument('--use_vae', action='store_true', default=False)
parser.add_argument('--nonnull', action='store_true', default=False)
args = parser.parse_args()


def main():
    seed = args.idx
    n = args.n
    m = args.m
    n_b = args.n_b
    m_b = args.m_b
    K = args.K
    uc = args.uc
    selection = args.selection
    ntrain = args.ntrain
    mu_list = np.zeros(K)
    if args.nonnull:
        mu_list[:25] = .1
    logs = [dict() for x in range(args.nrep)]
    for j in range(args.idx, args.idx + args.nrep):
        print("Starting simulation", j)
        logs[j - args.idx]['seed'] = j
        np.random.seed(j)
        X = np.zeros([K, n])
        for k in range(K):
            X[k, :] = np.random.randn(n) + mu_list[k]
        X_bar = np.mean(X, 1)
        if selection == 'mean':
            max_rest = np.sort(X_bar)[-2]
            win_idx = np.argmax(X_bar)
        elif selection == "UC":
            UC = X_bar + uc * np.std(X, axis=1, ddof=1)
            win_idx = np.argmax(UC)
            max_rest = np.sort(UC)[-2]
        else:
            raise AssertionError("invalid selection")
        # Stage 2
        X_2 = np.random.randn(m) + mu_list[win_idx]
        DTL_class = DropTheLoser(X, X_2)
        basis_type = args.basis_type
        Z_data = DTL_class.basis(X, X_2, basis_type)
        theta_data = DTL_class.theta_hat

        print("Generate initial data")
        training_data = DTL_class.gen_train_data(ntrain=ntrain, n_b=n_b, m_b=m_b, basis_type=args.basis_type, remove_D0=args.indep)
        Z_train = training_data['Z_train']
        W_train = training_data['W_train']
        gamma = training_data['gamma']
        print(np.mean(W_train))

        if args.use_vae and np.mean(W_train) <= .1:
            print("Start generating more positive data")
            pos_ind = W_train == 1
            Z_pos = torch.tensor(Z_train[pos_ind, :], dtype=torch.float)
            input_dim = Z_pos.shape[1]
            bottleneck_dim = 10
            vae_model = DTL_vae.VAE(input_dim, bottleneck_dim)
            vae_path = "DTL_VAE_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m)
            output_dim = n * K + m
            decoder = DTL_vae.Decoder(input_dim, output_dim)
            decoder_path = "DTL_decoder_seed_{}_n_{}_K_{}_m_{}.pt".format(seed, n, K, m)
            try:
                vae_model.load_state_dict(torch.load(vae_path))
                decoder.load_state_dict(torch.load(decoder_path))
            except:
                print("no model found, start training")
                DTL_vae.train_networks(n, K, bottleneck_dim, Z_pos, vae_path, decoder_path, output_dim, print_every=100, dec_epochs=2)
                vae_model.load_state_dict(torch.load(vae_path))
                decoder.load_state_dict(torch.load(decoder_path))

            n_vae = ntrain
            Z_vae = vae_model.decode(torch.randn(n_vae, bottleneck_dim))
            X_vae = decoder(Z_vae).detach().numpy()
            X_vae_1 = X_vae[:, :n * K].reshape(-1, K, n)
            X_vae_2 = X_vae[:, n * K:].reshape(-1, m)
            Z_train_vae = np.zeros([n_vae, K + 1])
            W_train_vae = np.zeros(n_vae)
            print("Start generating data using VAE+decoder")
            for i in range(n_vae):
                X_1_b = X_vae_1[i, :, :]
                X_2_b = X_vae_2[i, :]
                X_bar_b = np.mean(X_1_b, 1)
                if np.argmax(X_bar_b) == win_idx:
                    W_train_vae[i] = 1
                Z_train_vae[i, :] = DTL_class.basis(X_1_b, X_2_b, basis_type=basis_type)

            Z_train = np.concatenate([Z_train, Z_train_vae])
            W_train = np.concatenate([W_train, W_train_vae])
            print(Z_train.shape)

        # train
        print("Start learning selection probability")
        net, flag, pr_data = learn_select_prob(Z_train, W_train, Z_data=torch.tensor(Z_data, dtype=torch.float),
                                num_epochs=args.epochs, batch_size=args.batch_size, verbose=True)
        print('pr_data', pr_data)
        logs[j - args.idx]['pr_data'] = pr_data
        logs[j - args.idx]['flag'] = flag
        if args.indep:
            gamma_D0 = training_data['gamma_D0']
            Z_data = Z_data - gamma_D0 @ DTL_class.D_0.reshape(1, )
        N_0 = Z_data - gamma @ theta_data.reshape(1, )

        target_var = 1 / (n + m)
        target_sd = np.sqrt(target_var)
        gamma_list = np.linspace(-20 / np.sqrt(n_b + m_b), 20 / np.sqrt(n_b + m_b), 101)
        target_theta = theta_data + gamma_list
        target_theta = target_theta.reshape(1, 101)
        weight_val = get_weight(net, target_theta, N_0, gamma)
        interval_nn, pvalue_nn = get_CI(target_theta, weight_val, target_var, theta_data, return_pvalue=True)
        logs[j - args.idx]['interval_nn'] = interval_nn
        if interval_nn[0] <= mu_list[DTL_class.win_idx] <= interval_nn[1]:
            logs[j - args.idx]['covered_nn'] = 1
        else:
            logs[j - args.idx]['covered_nn'] = 0
        logs[j - args.idx]['width_nn'] = interval_nn[1] - interval_nn[0]

        logs[j - args.idx]['pvalue_nn'] = pvalue_nn
        print("pvalue", pvalue_nn)

        ##################################################
        # check learning
        count = 0
        nb = 50
        X_pooled = np.concatenate([X[win_idx], X_2])
        pval = []
        for ell in range(int(nb / np.mean(W_train))):
            X_b = np.zeros([K, n_b])
            for k in range(K):
                if k != win_idx:
                    X_b[k, :] = X[k, np.random.choice(n, n_b, replace=True)]
                if k == win_idx:
                    X_b[k, :] = X_pooled[np.random.choice(n + m, n_b, replace=True)]
            if selection == 'mean':
                idx = np.argmax(np.mean(X_b, 1))
            else:
                idx = np.argmax(np.mean(X_b, 1) + uc * np.std(X_b, axis=1, ddof=1))
            if idx != win_idx:
                continue
            else:
                count += 1
                X_2_b = X_pooled[np.random.choice(n + m, m_b, replace=True)]
                d_M = np.mean(np.concatenate([X_b[win_idx], X_2_b]))
                observed_target = d_M
                target_theta_0 = d_M + gamma_list
                target_theta_0 = target_theta_0.reshape(1, 101)
                target_val = target_theta_0
                weight_val = get_weight(net, target_theta_0, N_0, gamma)
                weight_val_2 = weight_val * norm.pdf((target_val - observed_target) / target_sd)
                exp_family = discrete_family(target_val.reshape(-1), weight_val_2.reshape(-1))
                hypothesis = theta_data
                pivot = exp_family.cdf((hypothesis - observed_target) / target_var, x=observed_target)
                pivot = 2 * min(pivot, 1 - pivot)
                pval.append(pivot)
                if count == nb:
                    break
        pval = np.array(pval)
        logs[j - args.idx]['pval'] = pval
        logs[j - args.idx]['false_rej'] = sum(pval <= 0.05) / len(pval)
        print(pval)
        print("reject:", sum(pval <= 0.05) / len(pval))

        ##################################################
        # true interval
        var_0 = 1 / n - 1 / (n + m)
        observed_target = theta_data
        gamma_list = np.linspace(-20 / np.sqrt(n_b + m_b), 20 / np.sqrt(n_b + m_b), 101)
        target_val = gamma_list + theta_data
        prob_gamma_true = []
        for gamma_t in gamma_list:
            if selection == "mean":
                prob_gamma_true.append(norm.sf((np.sort(X_bar)[-2] - theta_data - gamma_t) / np.sqrt(var_0)))
            if selection == "UC":
                tmp = []
                for i in range(1000):
                    tmp.append(np.sqrt(var_0) * np.random.randn(1)
                               + uc / np.sqrt(n - 1) * np.sqrt(np.random.chisquare(df=n - 1)))
                prob_gamma_true.append(sum(tmp > max_rest - theta_data - gamma_t) / len(tmp))
        interval_true, pvalue_true = get_CI(target_val, np.squeeze(prob_gamma_true), target_var, observed_target, return_pvalue=True)
        logs[j - args.idx]['interval_true'] = interval_true
        if interval_true[0] <= mu_list[DTL_class.win_idx] <= interval_true[1]:
            logs[j - args.idx]['covered_true'] = 1
        else:
            logs[j - args.idx]['covered_true'] = 0
        logs[j - args.idx]['width_true'] = interval_true[1] - interval_true[0]
        logs[j - args.idx]['pvalue_true'] = pvalue_true

        plt.figure()
        plt.plot(target_val, weight_val, label="nn")
        plt.plot(target_val, prob_gamma_true, label="truth")
        plt.legend()
        plt.savefig("{}_n_{}_m_{}_K_{}_nb_{}_{}.png".format(args.logname, n, m, K, n_b, j))

        ##################################################
        # stage 2 interval
        interval_2 = tuple((norm.ppf(0.025) / np.sqrt(m), -norm.ppf(0.025) / np.sqrt(m)) + np.mean(X_2))

        logs[j - args.idx]['interval_2'] = interval_2
        if interval_2[0] <= mu_list[DTL_class.win_idx] <= interval_2[1]:
            logs[j - args.idx]['covered_2'] = 1
        else:
            logs[j - args.idx]['covered_2'] = 0
        logs[j - args.idx]['width_2'] = interval_2[1] - interval_2[0]
        pivot = norm.cdf(-np.sqrt(m) * np.mean(X_2))
        logs[j - args.idx]['pvalue_2'] = 2 * min(pivot, 1 - pivot)

        X_cat = np.concatenate([X[win_idx, :], X_2])
        interval_naive = tuple((norm.ppf(0.025) / np.sqrt(n + m), -norm.ppf(0.025) / np.sqrt(n + m)) + np.mean(X_cat))
        width_naive = -2 * norm.ppf(0.025) / np.sqrt(n + m)
        logs[j - args.idx]['interval_naive'] = interval_naive
        logs[j - args.idx]['width_naive'] = width_naive
        if interval_naive[0] <= mu_list[DTL_class.win_idx] <= interval_naive[1]:
            logs[j - args.idx]['covered_naive'] = 1
        else:
            logs[j - args.idx]['covered_naive'] = 0
        pivot = norm.cdf(-np.sqrt(n + m) * np.mean(X_cat))
        logs[j - args.idx]['pvalue_naive'] = 2 * min(pivot, 1 - pivot)

        logs[j - args.idx]['mu_true'] = mu_list[DTL_class.win_idx]

        path = open("{}_n_{}_m_{}_K_{}_nb_{}_{}.pickle".format(args.logname, n, m, K, n_b, j), 'wb')
        pickle.dump(logs[j - args.idx], path)
        path.close()
    print(logs)


if __name__ == "__main__":
    main()
