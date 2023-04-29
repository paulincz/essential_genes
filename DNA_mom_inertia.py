# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:09:24 2021

@author: pauli
"""

import numpy as np
from scipy import spatial

class DNA:
    def __init__(self, seq, dim, desc1):
        self.seq = seq
        self.dim = dim
        self.N = len(seq)
        self.desc1 = desc1
        
    def DNA_coord(self):

        DNA_coord = np.zeros((self.N, self.dim))

        if self.dim == 2:

            DNA_dict = {'A':np.array([-1,0]),'C':np.array([0,1]),'G':np.array([1,0]),'T':np.array([0,-1])}

            for idx, elem in enumerate(self.seq):
                vec = DNA_dict[elem]
                DNA_coord[idx] = DNA_coord[idx-1] + vec

        elif self.dim == 3:

            DNA_dict = {'A':np.array([-1,0,1]),'C':np.array([0,1,1]),'G':np.array([1,0,1]),'T':np.array([0,-1,1])}

            for idx, elem in enumerate(self.seq):
                vec = DNA_dict[elem]
                DNA_coord[idx] = DNA_coord[idx-1] + vec

        return DNA_coord

    def DNA_mc(self):
        coord = self.DNA_coord()
        cm_coord = np.sum(coord, axis=0)/len(coord)

        return cm_coord

    def DNA_I_ten(self):

        center_mass = self.DNA_mc()
        coord = self.DNA_coord()
        coord_upd = np.transpose(coord - center_mass)

        I_ten = np.zeros((self.dim, self.dim))

        if self.dim == 2:
            for i in range(2):
                for j in range(2):
                    if i == 0 and j == 0:
                        I_ten[0][0] = np.sum((coord_upd[1]**2))
                    elif i == 1 and j == 1:
                        I_ten[1][1] = np.sum((coord_upd[0]**2))
                    else:
                        I_ten[i][j] = -np.sum(coord_upd[0]*coord_upd[1])

        elif self.dim == 3:
            for i in range(3):
                for j in range(3):
                    if i == 0 and j == 0:
                        I_ten[0][0] = np.sum((coord_upd[1]**2)+(coord_upd[2]**2))
                    elif i == 1 and j == 1:
                        I_ten[1][1] = np.sum((coord_upd[0]**2)+(coord_upd[2]**2))
                    elif i == 2 and j ==2:
                        I_ten[2][2] = np.sum((coord_upd[0]**2)+(coord_upd[1]**2))
                    else:
                        I_ten[i][j] = -np.sum(coord_upd[i]*coord_upd[j])

        return I_ten

    def DNA_descriptors(self):

        DNA_I = self.DNA_I_ten()
        DNA_cent_mass = self.DNA_mc()
        
        eig, v = np.linalg.eig(np.array(DNA_I)) #those aren't sorted correctly

        #a solution to sort the eigenvalues correctly (hope this one is a correct solution) 

        d = np.sum(DNA_I, axis=0).tolist()
        w = list(enumerate(d))
        w.sort(key = lambda x: x[1])  

        eig2 = [x for _, x in sorted(zip([x[0] for x in w],eig), key=lambda pair: pair[0])] #the eigenvalues are now in pretty much the same order as they were in tables in the article

        #creating descriptor matrix

        D = np.zeros(self.dim)

        if self.dim == 2 and self.desc1 == False:
            for i in range(len(D)):
                D[i] = DNA_cent_mass[i]/eig2[i]

            return D

        elif self.dim == 3:
            for i in range(len(D)):
                D[i] = np.sqrt(eig2[i]/self.N)

            #Cij matrix
            C = np.zeros((3,3))

            for i in range(3):
                for j in range(3):
                    C[i][j] = spatial.distance.cosine(DNA_cent_mass[i], eig2[j])

            return D, C