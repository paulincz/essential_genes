# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:13:00 2021

@author: pauli
"""

import numpy as np
import math

class ComplexityEntropy:
    
    def __init__(self, seq, L = 12, all_or_sub = 0, K1=2.2):
        self.seq = seq
        self.L = L
        self.all_or_sub = all_or_sub
        self.K1 = K1
       
    def matrix_maker(self):

        s = len(self.seq)
        if self.all_or_sub == 1:    
            subseq = [self.seq[i:i+self.L] for i in range(s)]
            filtered_subseq = list(filter(lambda x: len(x) == self.L, subseq))
        
            mat = np.zeros((s,20))
            amino_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}
            for idx, k in enumerate(filtered_subseq):
                for i in range(len(k)):
                    index = amino_dict[k[i]]
                    mat[idx][index] += 1
    
            return mat, filtered_subseq
        else:
            mat = np.zeros(s)
    
            used = []
            for i in range(s):
                if self.seq[i] not in used:
                    mat[i] += 1
                    used.append(self.seq[i])
                else:
                    first_used = used.index(self.seq[i])
                    mat[first_used] += 1
    
            return mat
        
    def log_omega(self):
        
        if self.all_or_sub == 1:
            matrix, subseq = self.matrix_maker()
            m, n = matrix.shape[0], matrix.shape[1]
            
            mat_fact = []
        
            for i in range(m):
                row = []
                for j in range(n):
                    mat_factorial = sum(math.log(ii) for ii in range(1, int(matrix[i][j]) + 1))
                    if not mat_factorial == 0:
                        row.append(mat_factorial)
                    else:
                        row.append(1)
                mat_fact.append(row)
                
            product = []
        
            for i in range(len(mat_fact)):
                num = mat_fact[i]
                if len(num) > 1:
                    prod = np.prod(num)
                    product.append(prod)
                elif len(num) == 1:
                    product.append(num.pop())
        
            product = np.array(product)
            
            om = (np.ones((m))*sum(math.log(ii) for ii in range(1, self.L + 1)))
            om /= product
        
        return om
        
    
    def K1_complexity(self): #dla całej sekwencji i dla podsekwencji
        
        if self.all_or_sub == 1:
            matrix, subseq = self.matrix_maker()
        
            K1_mat = (1/self.L)*self.log_omega()
        
            up_subseq = []
            for i in range(len(subseq)):
                if K1_mat[i] <= self.K1:
                    up_subseq.append(subseq[i])
            
            return K1_mat, up_subseq
        
        else:
            x = []
            matrix = self.matrix_maker()
            K1_mat = (1/self.L)*np.log(matrix)
            

            for i in K1_mat:
                if i <= self.K1:
                    if not math.isinf(i):
                        x.append(i)
                                     
            return x
        
    def K2_entropy(self): #dla całej sekwencji
        
        matrix = self.matrix_maker()
        c = len(self.seq)
        
        K2 = 0
        
        for i in range(c):
            if not matrix[i] == 0: 
                K2 += -((matrix[i]/c)*np.log(matrix[i]/c))
                  
        return K2