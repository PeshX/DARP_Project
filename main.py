import numpy as np
import gurobipy as gb
import pygad as ga
import networkx as nx
from instance_gen import *
from solver import *

# ALGORITHM PARAMETERS 
N = 10 #nb of individuals in the initial population
nb_generations = 100
proba_mutation = 0.2 #proba for an ind to be muted 
proba_mutation_gene = 0.3 #proba for a gene to be muted 
proba_crossing = 0.4

# INSTANCE CREATION 
nb_passengers = 15
nb_of_diff_starting_points = 15
nb_of_diff_destinations = 10
nb_of_vehicles = 5
vehicles_capacities = np.random.choice(range(1, 6), nb_of_vehicles).tolist()

num_nodes = nb_of_diff_destinations + nb_of_diff_starting_points # pickout/dropout points 
min_weight_fuel, max_weight_fuel = 1,10
min_weight_time, max_weight_time = 20,100
min_degree = 2

G = createGraphInstance(num_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree)

# TEST 
T1 = [1, 3, 5]
T2 = [4, 2]
individual = [T1, T2]
#res = fitness(individual, G)
#print("fitness of individual:", res)


initial_pop =  generate_initial_pop(N, nb_passengers, vehicles_capacities)

# SELECTION PROCESS 

selected_individuals_by_roulette = roulette_wheel_selection(initial_pop, fitness, G) 
selected_individuals_by_tournament = tournament_selection(initial_pop, fitness, G, 3)

print("WITH ROULETTE SELECTION, THE SELECTIONED INDIVIDUALS ARE:")
for ind in selected_individuals_by_roulette: 
    print(ind)

print("WITH TOURNAMENT SELECTION, THE SELECTED INDIVIDUALS ARE:")
for ind in selected_individuals_by_tournament: 
    print(ind)

"""
To compare both approach, we can compare mean of fitnesses of the selected individuals  
The bigger the sum is, the worst are the individual, that's a first metric
"""
mean_fitness_roulette = compute_mean_fitness(selected_individuals_by_roulette, fitness, G)
mean_fitness_tournament = compute_mean_fitness(selected_individuals_by_tournament, fitness, G)

print("mean fitness roulette: ", mean_fitness_roulette)
print("mean fitness tournament: ", mean_fitness_tournament)

# GENETIC ALGORITHM 
nb_iterations = 0 
initial_population = generate_initial_pop(nb_individuals=N, nb_passengers=nb_passengers, capacities=vehicles_capacities)

parent_population = initial_population

# data for comparison 
best_fitness_from_each_gen = []
mean_fitness_first_X_ind_from_each_gen = []
mean_of_fitness_whole_pop_from_each_gen = []

# NB : try with different selection processes 

while (nb_iterations <= nb_generations): 
    child_population = generate_next_generation(parent_population, fitness, N) 
    parent_population = child_population 
    nb_iterations += 1

# REMEMBER TO FETCH DATA FOR THE PLOTS 


