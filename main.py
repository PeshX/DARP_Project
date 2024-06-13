import numpy as np
import gurobipy as gb
import pygad as ga
import networkx as nx
from instance_gen import *
from solver import *

# ALGORITHM PARAMETERS 
N = 50 #nb of individuals in the initial population
nb_generations = 100
proba_mutation = 0.2 #proba for an ind to be muted 
proba_mutation_gene = 0.3 #proba for a gene to be muted 
proba_crossing = 0.4

# INSTANCE CREATION 
num_nodes = 10 # pickout/dropout points 
min_weight_fuel, max_weight_fuel = 1,10
min_weight_time, max_weight_time = 20,100
min_degree = 2

G = createGraphInstance(num_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree)

# TEST 
T1 = [1, 3, 5]
T2 = [4, 2]
individual = [T1, T2]
res = fitness(individual, G)
print(res)



