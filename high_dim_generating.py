import numpy as np


def high_dim_simu(p=3000, n=1500, rho=0.3, var=2, k=30, scenario="A", n_copy=100,rand_seed=1):
    x = np.zeros([n, p, n_copy])
    t = np.zeros([n, n_copy])
    yf = np.zeros([n, n_copy])
    ycf = np.zeros([n, n_copy])
    mu1 = np.zeros([n, n_copy])
    mu0 = np.zeros([n, n_copy])

    set.seed(rand_seed)
    # Parameter generating
    first_half = np.random.normal(size=k / 2)
    second_half = np.random.normal(size=k / 2)
    first_half_tau = np.random.normal(size=k / 2)
    second_half_tau = np.random.normal(size=k / 2)
    first_half_gamma = np.random.normal(size=k / 2)
    second_half_gamma = np.random.normal(size=k / 2)

    for i in range(n_copy):
        mean_vec = np.zeros(p)
        if scenario == "A":
            beta = np.concatenate((first_half, second_half, np.zeros(p - k)))
            beta_tau = np.concatenate((first_half_tau, second_half_tau, np.zeros(p - k)))
            gamma = np.concatenate((first_half_gamma, second_half_gamma, np.zeros(p - k)))
        elif scenario == "B":
            beta = np.concatenate((first_half, second_half, np.zeros(p - k)))
            beta_tau = np.concatenate((first_half_tau, second_half_tau, np.zeros(p - k)))
            gamma = np.concatenate((first_half_gamma, np.zeros(k / 2), second_half_gamma, np.zeros(p - k - k / 2)))
        else:
            beta = np.concatenate((np.zeros(p - k), first_half, second_half))
            beta_tau = np.concatenate((np.zeros(p - k), first_half_tau, second_half_tau))
            gamma = np.concatenate((first_half_gamma, second_half_gamma, np.zeros(p - k)))

        Sigma_x = (np.ones([p, p]) * rho + np.identity(p) * (1 - rho)) * var

        x[:, :, i] = np.random.multivariate_normal(mean_vec, Sigma_x, n)
        prob_t = 1 / (1 + np.exp(-np.matmul(x[:, :, i], gamma)))

        t[:, i] = np.random.binomial(1, prob_t, size=n)

        mu0[:, i] = np.matmul(x[:, :, i], beta)
        mu1[:, i] = mu0[:, i] + np.matmul(x[:, :, i], beta_tau)
        if i == 0:
            ate = np.mean(mu1[:, i] - mu0[:, i])
            t_id = np.where(t[:, i] == 1)
            att = np.mean(mu1[t_id, i] - mu0[t_id, i])
        noise = np.random.normal(size=n)
        yf[:, i] = t[:, i] * mu1[:, i] + (1 - t[:, i]) * mu0[:, i] + noise
        ycf[:, i] = t[:, i] * mu0[:, i] + (1 - t[:, i]) * mu1[:, i] + noise
        if i%10==0:
            print (i, "th finished")

    return {'x': x, 't': t, 'mu0': mu0, 'mu1': mu1, 'yf': yf, 'ycf': ycf, 'ate': ate, 'att': att}

train_A=high_dim_simu(n=600,p=2000,k=20,scenario="A",n_copy=100)
np.savez("high_dim_100_train_A.npz",**train_A)
test_A=high_dim_simu(n=200,p=2000,k=20,scenario="A",n_copy=100)
np.savez("high_dim_100_test_A.npz",**test_A)
train_B=high_dim_simu(n=600,p=2000,k=20,scenario="B",n_copy=100)
np.savez("high_dim_100_train_B.npz",**train_B)
test_B=high_dim_simu(n=200,p=2000,k=20,scenario="B",n_copy=100)
np.savez("high_dim_100_test_B.npz",**test_B)
train_C=high_dim_simu(n=600,p=2000,k=20,scenario="C",n_copy=100)
np.savez("high_dim_100_train_C.npz",**train_C)
test_C=high_dim_simu(n=200,p=2000,k=20,scenario="C",n_copy=100)
np.savez("high_dim_100_test_C.npz",**test_C)
