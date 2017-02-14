# Author: Patrick A. O'Neil

import numpy as np

from util import *

class MultiParameterCube(object):
    def __init__(self, nparams, dim):
        self.dim = dim
        self.nparams = nparams
        self.cube = [np.zeros(dim) for i in range(nparams)]

    def copy_data(self, data, param):
        self.cube[param] = data

    def convert_to_scalar_cube(self, K=3,  method='std'):
        sc = np.zeros(self.dim)
        for i in range(np.prod(self.dim)):
            P = np.unravel_index(i, self.dim)
            N = get_neighbors(P, self.dim, K=k)
            for i in range(self.nparams):
                V = []
                for n in N:
                    V.append(self.cube[i][n])
                V.append(self.cube[i][P])
                u = np.mean(V)
                s = np.sum(np.power(np.array(V) - u, 2)) / float(len(N) + 1)
                sc[P] = s
        return sc
