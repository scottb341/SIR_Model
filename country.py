# -*- coding: utf-8 -*-
"""
@author: Scott Blyth
@studentid: 32501013
"""
import numpy as np
from disease import disease
from GI import Environment, evolve
from random import random
from numpy.linalg import matrix_power
from pprint import pprint
import math
from scipy.integrate import odeint

# utils 

def all_zeros(lst):
    for val in lst:
        if val != 0:
            return False
    return True

# from workshop/lectures
def gillespie(Y0, t0, t_max=100, max_iter=10**4):
    y, t = Y0, t0
    #Y, T = np.array(Y0.get_info()), [t]
    i = 0
    events,event_consequences = y.get_events() 
    p_n = len(events)
    while t<t_max and i <= max_iter:
        events,event_consequences = y.get_events()
        p = propensities(t, p_n, events=events)
        if all_zeros(p):
            return
        p_rel = p/sum(p)
        tte = [time_to_event(p[i]) for i in range(len(p))]
        idx = np.random.choice(range(len(p)), p=p_rel)
        event, dt = event_consequences[idx], tte[idx]
        event(t)
        t += dt    
        i += 1

time_to_event = lambda p: (-1/p)*np.log(np.random.random())

def propensities(t, p_n, events):
    e_ = []
    for event in events:
        e = event(t=t)
        e_.append(e*p_n)
    return e_

class Country:
    mu = 5
    
    def __init__(self, id: int, birth_rate, y_0, infection : disease, lockdown=False, lockdown_threshold=0.1, 
                 plot=False, lifiting_threshold=0.01): 
        # self.current = [S,I,R,D]
        self.current = y_0 
        self.total_population = sum(y_0)
        self.disease = infection 
        self.birth_rate = birth_rate
        self.neighbours = []
        self.id = id
        self.history = np.array([y_0])
        self.times = [0]
        self.plot = plot
        self.closed_borders = False 
        self.lockdown = lockdown # will this country go into lockdown or not?
        self.lockdown_factor = 1
        self.lockdown_threshold = lockdown_threshold
        self.lifiting_threshold = lifiting_threshold
        
        self.props = None
        self.events = None
    
    def get_info(self): 
        return np.copy(self.current)
        
    # connects this country to the given country
    # the propensity to move from this country to another 
    # is defined by l_i_j
    def add_neighbour(self, l_i_j, l_j_i, country):
        self.neighbours.append((l_i_j, l_j_i, country))
        
    # moves person from box self.current[i] to box self.current[j]
    # note self.current = [S,I,R,D]
    def move_i_to_j(self, i, j, t):
        # ensures box is never negative
        if self.current[i] == 0:
            return
        deaths = self.current[3] 
        # detects if this country is in lockdown
        # if so, set lockdown factor (reduces the infectivity within this country)
        if self.lockdown and self.current[1]/self.total_population >= self.lockdown_threshold:
            self.lockdown_factor = 0.01
        # detects if lockdown is going to be lifted
        if self.lockdown and self.current[1]/self.total_population < self.lifiting_threshold:
            self.lockdown_factor = 1 
        self.current[i] -= 1 # S
        self.current[j] += 1 # I 
        # if the graph is to be plotted,
        # save this change
        if self.plot:
            self.times.append(t)
            self.history = np.vstack([self.history, self.get_info()])
      
    # moves person from this country in the self.current[SIRD_index] box
    # to self.neighbours[neighbour] country - travelling mechanic
    def move_to_country(self, SIRD_index, neighbour, t):
        if self.current[SIRD_index] == 0:
            return
        l1,l2,country = self.neighbours[neighbour]
        country.current[SIRD_index] += 1
        self.current[SIRD_index] -= 1
        # if the graph is to be plotted,
        # save this change
        if self.plot:
            self.times.append(t)
            self.history = np.vstack([self.history, self.get_info()])
    
    def get_SIRD_events(self): 
        # computes the propensities and events for 
        # infection spread within this country. 
        S,I,R,D = self.current
        l1,l2,l3,l4 = self.disease.get_params()
        # the infectivity, which is moving people from S to I,
        # is reduced by a factor self.lockdown_factor
        S_to_I = lambda t : l1*self.get_info()[0]*self.get_info()[1]*self.lockdown_factor
        I_to_R = lambda t : l2*self.get_info()[1]
        R_to_S = lambda t : l4*self.get_info()[2]
        I_to_D = lambda t : l3*self.get_info()[1]
        props = np.array([S_to_I, I_to_R, R_to_S, I_to_D])
        # consequences
        event_S_I = lambda t : self.move_i_to_j(0, 1, t)
        event_I_R = lambda t : self.move_i_to_j(1, 2, t)
        event_R_S = lambda t : self.move_i_to_j(2, 0, t)
        event_I_D = lambda t : self.move_i_to_j(1, 3, t)
        events = np.array([event_S_I, event_I_R, event_R_S, event_I_D])
        return props,events

    # returns the propnesities and events for each 
    # movement operation frm country to country
    def country_to_country(self): 
        # returns the functions that return the events at the time t
        def func(s, i): 
            def func2(t): 
                return self.move_to_country(s, i, t)
            return func2
        lst = []
        # collect all of the propentity functions 
        # for each edge between this country and neighbouring countries.
        for l1,l2,c in self.neighbours: 
            lst.append(lambda t : l1*self.get_info()[0]) # S 
            lst.append(lambda t : l1*self.get_info()[1]/Country.mu) # I 
            lst.append(lambda t : l1*self.get_info()[2]) # R 
        props = np.array(lst)
        events = []
        # collect all of the event functions 
        # for each edge between this country and neighbouring countries.
        for i,val in enumerate(self.neighbours):
            l1,l2,c = val
            events.append(func(0, i))
            events.append(func(1,i))
            events.append(func(2,i))
        events = np.array(events)
        
        return props,events
        
    # get all of the propensities and events for this country 
    # into two lists, where elements of the same index in events 
    # line up with propensities of the same index in propensities
    def get_events(self): 
        p1,e1 = self.get_SIRD_events()
        p2,e2 = self.country_to_country()
        if len(p2) == 0:
            return p1,e1
        p = np.concatenate([p1,p2])
        e = np.concatenate([e1,e2])
        return p,e
    
def random_interval(i, j):
    return (j-i)*random()+i 
    
def random_vec(n, i,j):
    return [(j-i)*random()+i for _ in range(n)]
    
class GridEnvironemnt(Environment): 
    def __init__(self, country_l, dimensions, lockdown=False):
        self.country_l = country_l
        self.n = dimensions 
        self.lockdown = lockdown
        
        # defines legal bounds -> infections 
        # outside these bounds have fitness 0
        self.l1_bounds = (0.000025, 0.00003)
        self.l2_l3_bounds = (0.01, 0.2) 

        self.l4_bounds = (0.0001, 0.0002) 
        # used to increase efficiecny 
        # if already computed fitness, grab it from cache
        self.cache_fitness = {}
        
    def in_bounds(self, l1,l2,l3,l4):
        cond1 = self.l1_bounds[0] <= l1 <= self.l1_bounds[1]
        cond2 = self.l2_l3_bounds[0] <= l2+l3 <= self.l2_l3_bounds[1]
        cond3 = self.l4_bounds[0] <= l4 <= self.l4_bounds[1]
        return cond1 and cond2 and cond3
    
    def random_population(self): 
        # generates random disease that is within the 
        # pre defined bounds.
        l1 = random_interval(self.l1_bounds[0], self.l1_bounds[1])
        l2 = random_interval(self.l2_l3_bounds[0], self.l2_l3_bounds[1]) 
        l3 = random_interval(self.l2_l3_bounds[0], self.l2_l3_bounds[1]) 
        l4 = random_interval(self.l4_bounds[0], self.l4_bounds[1])
        return disease(l1, l2, l3, l4)
    
    
    def fitness_aux(self ,l1,l2,l3,l4): 
        # computes the fitness of the disease defined by (l1,l2,l3,l4)
        if not self.in_bounds(l1,l2,l3,l4):
            return 1
        d = disease(l1,l2,l3,l4)
        countries,grid = create_grid(d, self.country_l, (self.n,self.n), self.lockdown) 
        grid[0][0].current[1] = 5
        w = World(countries)
        gillespie(w, 0,  t_max=365, max_iter=2*10**5)
        num_deaths = 0 
        for c in countries:
            num_deaths += c.get_info()[3]
        self.cache_fitness[(l1,l2,l3,l4)] = 0 if num_deaths < 0 else num_deaths
        return 0 if num_deaths < 0 else num_deaths
    
    def fitness(self, sol): 
        # if already computed fitness, get it from cache
        if sol in self.cache_fitness:
            return self.cache_fitness[sol]
        l1,l2,l3,l4 = sol.to_phenotype()
        self.cache_fitness[sol] = self.fitness_aux(l1, l2, l3, l4)
        return self.cache_fitness[sol]
    
class World:
    def __init__(self, countries):
        self.countries  = countries
        # used to cache propensity and event functions
        self.props = None
        self.events = None
        
    def get_events(self):
        if self.props is not None:
            return self.props,self.events
        props = []
        events = []
        for c in self.countries: 
            p,e = c.get_events() 
            props += list(p) 
            events += list(e)
        props = np.array(props)
        events = np.array(events)
        # cache propensity and event functions
        self.props = props 
        self.events = events
        return props,events
    
    def get_info(self):
        info = []
        for c in self.countries:
            info.append(c.get_info())
        return info
            
def adj_mat_world(adj_matrix, max_propensity, populations, disease): 
    n = len(adj_matrix)
    countries = [Country(i, 0, populations[i], disease) for i in range(n)]
    for i in range(n):
        for j in range(n):
            p = adj_matrix[i][j]
            if p != 0:
                countries[i].add_neighbour(max_propensity*p,0, countries[j])
    return World(countries)

def flow_mat_to_world(flow_matrix, populations, disease):
    n = len(flow_matrix)
    countries = [Country(i, 0, populations[i], disease) for i in range(n)]
    for i in range(n):
        for j in range(n):
            f = flow_matrix[i][j]
            if f != 0:
                countries[i].add_neighbour(f,0, countries[j])
    return World(countries)



def create_grid(disease, country_l, dimensions, plot_output=False, 
                        lockdown=False, lockdown_threshold=0.1, lifiting_threshold=0.01):
    # create grid of countries 
    n,m = dimensions 
    country_grid = [[None for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            c = Country(i*m+j, 0, np.array([10**4, 0,0,0]), disease, plot=plot_output, lockdown=lockdown,
                        lockdown_threshold=lockdown_threshold, lifiting_threshold=lifiting_threshold)
            country_grid[i][j] = c
    # get vertical and adjacent neighbours
    def get_neighbours(i, j): 
        neighbours = [(i, j-1), (i, j+1),
                      (i-1, j), (i+1,j)]
        return [(i,j) for i,j in neighbours if 0 <= i < n and 0 <= j < m]
    countries = []
    for i in range(n):
        for j in range(m):
            countries.append(country_grid[i][j])
            for i_n,j_n in get_neighbours(i,j):
                country_grid[i][j].add_neighbour(country_l, country_l, country_grid[i_n][j_n])
    return countries, country_grid


# stability stuff

def flatten(vec):
    return vec / sum(vec) 

def flow_probability(l):
    if l == 0:
        return 0
    gamma = 1/l 
    numerator = 2*gamma-math.sqrt(4*gamma+1)+1
    denominator = 2*gamma
    return numerator/denominator
    
def round_matrix(mat, n):
    res = np.copy(mat) 
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res[i][j] = round(mat[i][j], n)
    return res

def sum_col(matrix, i):
    s = 0 
    for row in matrix:
        s += row[i] 
    return s

def stability_point(flow_matrix): 
    # make matrix 
    n = len(flow_matrix)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][0] = sum_col(flow_matrix, i)
        for j in range(n):
            if j != i:
                matrix[i][j] = -flow_matrix[i][j]
    matrix = np.array(matrix) 
    ones = np.array([1,0,0])
    S = np.linalg.solve(matrix, ones)
    return S

def PFM(flow_matrix):
    n = len(flow_matrix)
    size = n**2 + n
    Q = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(n):
        flow = flow_matrix[i]
        # compute probabilties 
        probabilties = flow / sum(flow)
        for j in range(n):
            if i == j:
                continue
            index = n+i*n+j
            if flow[j] == 0:
                flow_prob = 1
            else:
                gamma = 1/flow[j]
                flow_prob = gamma/(1+gamma)
            Q[i][index] = probabilties[j]
            Q[index][index] = flow_prob
            Q[index][j] = 1-flow_prob
    return np.array(Q)
            
        


def jacobi_newton(J_f, F, x, iterations=25): 
    # J_f is a function 
    x = x / np.linalg.norm(x)
    J_0 = J_f(x) 
    F_0 = F(x)
    for _ in range(iterations): 
        dx = np.linalg.solve(J_0, F_0) 
        x = x - dx
        J_0 = J_f(x)
        F_0 = F(x)
    return x


def find_stable(Q, population, n=100):
    f = int((-1+math.sqrt(1+4*len(Q)))/2)
    Q_star = matrix_power(Q, n)
    stable = np.ones(f)
    for i in range(f):
        stable[i] = Q_star[0][i]
        for j in range(f): 
            stable[i] += Q_star[0][f+i*f+j]
    return population*stable

def PFM_Jacobi(flow_matrix, i): 
    n = len(flow_matrix)
    def func(P):
        J = [[0 for _ in range(n)] for _ in range(n)]
        # first bit 
        for j in range(n):
            if j != i and flow_matrix[i][j] != 0:
                J[j][j] = P[i]
            if flow_matrix[i][j] == 0:
                J[j][j] = 1
        # second bit 
        for j in range(n):
            J[i][j] = 1
        # third bit
        for j in range(n):
            if j != i and flow_matrix[i][j] != 0:
                l = 1/flow_matrix[i][j]
                J[j][i] = P[j]-2*l*(P[i]-1)
        return np.array(J) 
    return func

def PFM_Func(flow_matrix, i):
    n = len(flow_matrix) 
    def func(P):
        F = [0 for _ in range(n)]
        for j in range(n):
            if j != i and flow_matrix[i][j] != 0:
                l = 1/flow_matrix[i][j]
                F[j] = P[j]*P[i]-l*(P[i]-1)*(P[i]-1)
            if flow_matrix[i][j] == 0:
                F[j] = 0
        F[i] = sum(P)-1
        return np.array(F)
    return func

def flow_to_markov_chain(flow_matrix, iterations=25): 
    n = len(flow_matrix)
    Q = [[0 for _ in range(n)] for _ in range(n)]
    for i,f in enumerate(flow_matrix):
        J_f = PFM_Jacobi(flow_matrix, i)
        F = PFM_Func(flow_matrix, i)
        P = np.array([1/n for _ in range(n)])
        x = jacobi_newton(J_f, F, P, iterations)
        for j in range(n):
            Q[i][j] = x[j]
    return np.array(Q)

# used for plotting data
def add_curves(curves):
    time_steps = [] 
    for curve in curves:
        time_steps += [t for t,y in curve]
    time_steps = sorted(time_steps)
    resulting_curve = []
    current_indices = [0 for _ in range(len(curves))]
    for t in time_steps:
        # find t value of 
        sum_ = 0
        for i,curve in enumerate(curves):
            index = current_indices[i]
            if index >= len(curve):
                sum_ += curve[index-1][1]
                break
            sum_ += curve[index][1]
            if curve[index][0] <= t:
                current_indices[i] += 1
        resulting_curve.append((t, sum_))
    return [t for t,y in resulting_curve], [y for t,y in resulting_curve]

def dsdt(flow_matrix):
    n = len(flow_matrix)
    def f(y,t):
        s = [None]*n 
        for i in range(n): 
            inflow = sum(flow_matrix[j][i]*y[j] for j in range(n))
            outflow = y[i]*sum(flow_matrix[i][j] for j in range(n))
            s[i] = inflow-outflow 
        return np.array(s)
    return f 

def get_solver(flow_matrix, num_days, y0, n=50):
    f = dsdt(flow_matrix)
    t = np.linspace(0, num_days, n)
    sol = odeint(f, y0, t)
    return sol

def simulate_markov(Q, i, j):
    current = i
    n = len(Q)
    length = 0
    while True:
        next = np.random.choice(range(n), p=Q[current])
        if next == j:
            return length 
        current = next
        length += 1

def get_x(lst):
    return [i for i in range(len(lst))]

    
if __name__ == "__main__":
    d = disease(0.00005, 0,0,0)
    flow_matrix = np.array([[0, 0.1, 0.005], 
    [0.005, 0, 0.001], 
    [0.0003, 0.0004, 0]])
    w = flow_mat_to_world(flow_matrix, 
        [[5000, 0,0,0], [5000,0,0,0], 
         [0,0,0,0]], d)
    Q = PFM(flow_matrix)
    Q_r = round_matrix(Q, 2)



