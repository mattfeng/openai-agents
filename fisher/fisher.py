import numpy as np

def inv_fisher(xvals, sigmavals):
    npar = 2

    F = np.zeros([npar, npar])
    for x, sigma in zip(xvals, sigmavals):
        for i in range(npar):
            if i == 0:
                dfdpi = x
            else:
                dfdpi = 1
            for j in range(npar):
                if j == 0:
                    dfdpj = x
                else:
                    dfdpj = 1
                F[i, j] += (sigma ** -2) * dfdpi * dfdpj

    print(np.mat(F).I)  # invert the matrix

inv_fisher([-1, 1, 1], [0.1, 0.1, 0.1])