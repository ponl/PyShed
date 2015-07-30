# Author: Patrick A. O'Neil

import numpy as np

class MultiParameterCube(object):
    def __init__(self, nparams, dim):
        self.dim = dim
        self.nparams = nparams
        self.cube = [np.zeros(dim) for i in xrange(nparams)] 
        
    def copy_data(self, data, param):
        self.cube[param] = data

    def get_neighbors(self, P, k=1):
        N = []
        for i in xrange(-k,k+1):
            if P[0] + i < 0 or P[0] + i >= self.dim[0]:
                continue
            for j in xrange(-k,k+1):
                if P[1] + j < 0 or P[1] + j >= self.dim[1]:
                    continue
                for k in xrange(-k,k+1):
                    if P[2] + k < 0 or P[2] + k >= self.dim[2]:
                        continue
                    if i == 0 and j == 0 and k == 0:
                        continue
                    p = (P[0] + i, P[1] + j, P[2] + k)
                    N.append(p)
        return N

    def convert_to_scalar_cube(self, k=3,  method='std'):
        sc = np.zeros(self.dim)
        for i in xrange(np.prod(self.dim)):
            P = np.unravel_index(i, self.dim)
            N = self.get_neighbors(P, k=k)
            for i in xrange(self.nparams):
                V = []
                for n in N:
                    V.append(self.cube[i][n])
                V.append( self.cube[i][P] )
                u = np.mean( V ) 
                s = np.sum( np.power(np.array(V) - u, 2) ) / float(len(N) + 1)
                sc[P] = s
        return sc
