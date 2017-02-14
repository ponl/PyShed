# Author: Patrick A. O'Neil

import numpy as np
from collections import defaultdict

import util

class PersistenceWatershed(object):
    def __init__(self, arr):
        self.WSHED = 0
        self.INIT = -1
        self.arr = arr
        self.dim = self.arr.shape
        self.lab = np.empty(self.dim, dtype=int)
        self.lab.fill(self.INIT)
        self.dual = {}
        self.current_label = 1
        self.maxes = []
        self.mt = None

    def train(self):
        rank = np.argsort(self.arr.flatten())
        for r in rank[::-1]:
            P = np.unravel_index(r, self.dim)
            L = self.get_neighbor_labels(P)
            if len(L) == 0:
                # assign new label
                l = self.current_label
                self.lab[P] = l
                self.dual[l] = defaultdict(float)
                self.maxes.append((l, self.arr[P]))
                self.current_label += 1
            elif len(L) == 1:
                # assign existing label
                self.lab[P] = list(L)[0]
            else:
                # - Connect adjacent labels
                for l0 in L:
                    for l1 in L:
                        if l0 == l1:
                            continue
                        w = self.arr[P]
                        [n0, n1] = [min(l0,l1), max(l0,l1)]
                        self.dual[n0][n1] = max( self.dual[n0][n1], w)
                # - Assign watershed
                self.lab[P] = self.WSHED

        edges = []
        for n0 in self.dual:
            for n1 in self.dual[n0]:
                edges.append((n0, n1, self.dual[n0][n1]))
        self.mt = util.merge_tree(self.maxes, edges)

    def apply_threshold(self, t):
        relabel = {i: i for i in range(self.current_label)}
        for e in self.mt:
            if e[2] < t:
                relabel[e[0]] = e[1]
        rlab = np.empty(self.dim, dtype=int)
        wvox = []
        for i in range(len(self.lab.flatten())):
            P = np.unravel_index(i, self.dim)
            l = self.lab[P]
            if l == self.WSHED:
                rlab[P] = 0
                wvox.append(P)
                continue
            rlab[P] = relabel[self.lab[P]]

        for P in wvox:
            L = self.get_neighbor_labels(P, rlab)
            if len(L) == 1:
                rlab[P] = list(L)[0]
        return rlab

    def get_neighbor_labels(self, P, lab = None):
        N = util.get_neighbors(P, self.dim)
        L = set()
        for n in N:
            if lab is not None:
                l = lab[n]
            else:
                l = self.lab[n]
            if not l == self.INIT and not l == self.WSHED:
                L.add(l)
        return L
