# -*- coding: utf-8 -*-
"""
@author: Scott Blyth
@studentid: 32501013
"""

from GI import Genome
from random import random, randint
import numpy as np

def random_vec(n, vec):
    # generate random vector with bounds defined by vec
    return np.array([2*vec[i]*random()-vec[i] for i in range(n)])

class disease(Genome): 
    
    def __init__(self, l1,l2,l3,l4, mutation_rate = (0.00001, 0.00001, 0.00001, 0.00001), lock_l1=False):
        # l1 : infection rate
        # l2 : recovery rate
        # l3 : immunity 
        # l4 : mortality rate
        self.l1,self.l2,self.l3,self.l4 = l1,l2,l3,l4 
        self.lock_l1 = lock_l1
        self.mutation_rate = mutation_rate
        
    def to_phenotype(self):
        return np.array([self.l1,self.l2,self.l3,self.l4])
        
    def get_params(self):
        return self.l1,self.l2,self.l3,self.l4

    def copy(self):
        return disease(self.l1,self.l2,self.l3,self.l4)
    
    def mutate(self): 
        # shifts each parameter by random amount
        k = random_vec(4, self.mutation_rate)
        new_vec = self.to_phenotype()+k
        l1,l2,l3,l4 = new_vec
        # rerurns new disease
        return disease(abs(l1),abs(l2),abs(l3),abs(l4))
        
    def crossover(self, other):
        # 1 in 2 chance of just returning self
        if randint(1,2) == 1:
            return self
        diseases = [self.to_phenotype(), other.to_phenotype()]
        l1,l2,l3,l4 = [diseases[randint(0,1)][i] for i in range(4)]
        return disease(l1,l2,l3,l4)



