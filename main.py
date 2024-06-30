import numpy as np
import gurobipy as gb
import pygad as ga
import networkx as nx
from instance_gen import *
from solver import *
import matplotlib.pyplot as plt

# # ALGORITHM PARAMETERS 
# N = 20 #nb of individuals in the initial population
# nb_generations = 500
# proba_mutation = 0.2 #proba for an ind to be muted 
# proba_mutation_gene = 0.3 #proba for a gene to be muted 
# proba_crossing = 0.4

# # INSTANCE CREATION 
# nb_passengers = 15
# nb_of_diff_starting_points = 15
# nb_of_diff_destinations = 10
# nb_of_vehicles = 5
# vehicles_capacities = np.random.choice(range(1, 6), nb_of_vehicles).tolist()

# num_nodes = nb_of_diff_destinations + nb_of_diff_starting_points # pickout/dropout points 
# min_weight_fuel, max_weight_fuel = 1,10
# min_weight_time, max_weight_time = 20,100
# min_degree = 2

# G = createGraphInstance(num_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree)

# initial_pop =  generate_initial_pop(N, nb_passengers, vehicles_capacities)

# # SELECTION PROCESS 

# selected_individuals_by_roulette = roulette_wheel_selection(initial_pop, fitness, G) 
# selected_individuals_by_tournament = tournament_selection(initial_pop, fitness, G, 3)

# print("WITH ROULETTE SELECTION, THE SELECTIONED INDIVIDUALS ARE:")
# for ind in selected_individuals_by_roulette: 
#     print(ind)

# print("WITH TOURNAMENT SELECTION, THE SELECTED INDIVIDUALS ARE:")
# for ind in selected_individuals_by_tournament: 
#     print(ind)

# """
# To compare both approach, we can compare mean of fitnesses of the selected individuals  
# The bigger the sum is, the worst are the individual, that's a first metric
# """
# mean_fitness_roulette = compute_mean_fitness(selected_individuals_by_roulette, fitness, G)
# mean_fitness_tournament = compute_mean_fitness(selected_individuals_by_tournament, fitness, G)

# print("mean fitness roulette: ", mean_fitness_roulette)
# print("mean fitness tournament: ", mean_fitness_tournament)

# # GENETIC ALGORITHM 

# # 1 - INITIALIZATION 
# nb_iterations = 0 
# selection = "roulette" #"tournament"
# initial_population = generate_initial_pop(nb_individuals=N, nb_passengers=nb_passengers, capacities=vehicles_capacities)

# parent_population = initial_population

# # 2 - PREPARE DATA FOR PERFORMANCE ANALYTICS 
# best_fitness_from_each_gen = []
# mean_fitness_first_X_ind_from_each_gen = []
# X = 5
# mean_of_fitness_whole_pop_from_each_gen = []

# # 3 - RUNNING THE ALGORITHM 
# while (nb_iterations <= nb_generations):     

#     child_population = generate_next_generation(parent_population=parent_population, fitness=fitness, nb_individuals=N, selection_process=selection, proba_mutation=proba_mutation, graph=G)
#     parent_population = child_population 
#     nb_iterations += 1

#     # FETCHING DATA FOR PERFORMANCE MEASUREMENTS 

#     # KPI1 : Fitness of the best individual of each generation 
#     sorted_child_population = sorted(child_population, key=lambda ind: fitness(ind, G))
#     test = [fitness(ind, G) for ind in sorted_child_population]
#     #print(test)
#     #best_ind_from_child_pop = sorted_child_population[0]
#     #best_fitness = fitness(best_ind_from_child_pop,G) 
#     first_non_zero = next((value for value in test if value != 0), None)
#     best_fitness = first_non_zero
#     best_fitness_from_each_gen.append(best_fitness)

#     # KPI2 : Fitness of the first X individuals of each generation for the mean 
#     top_individuals = sorted_child_population[:X]
#     top_fitnesses = [fitness(ind, G) for ind in top_individuals]
#     mean_of_the_fitness_of_first_X_individuals = np.mean(top_fitnesses)
#     mean_fitness_first_X_ind_from_each_gen.append(mean_of_the_fitness_of_first_X_individuals)

#     # KPI3 : Fitness of all individuals of each generation 
#     fitnesses_of_whole_gen = [fitness(ind, G) for ind in child_population]
#     mean_fitness_of_whole_pop = np.mean(fitnesses_of_whole_gen)
#     mean_of_fitness_whole_pop_from_each_gen.append(mean_fitness_of_whole_pop)


# # PLOTTING 
# generation_indices = range(nb_generations+1)

# # NB : try with different selection processes 

# # PLOT 1 : evolution of the fitness along the reproduction process (best fitness for each gen)
# plt.plot(generation_indices, best_fitness_from_each_gen, label="Evolution of the best fitness of each generation")
# plt.xlabel('Generation')
# plt.ylabel('Best fitness')
# plt.show()

# # PLOT 2 : evolution of the mean of the fitness of the X best individuals along the reproduction process 
# plt.plot(generation_indices, mean_fitness_first_X_ind_from_each_gen, label='Evolution of the mean of the fitness of the f"{X} first best individuals')
# plt.xlabel('Generation')
# plt.ylabel('Mean of fitnesses of the f{X} best individuals of the generation')
# plt.show()

# # PLOT 3 : evolution of the mean of the fitness of all individuals along the reproduction process 
# plt.plot(generation_indices, mean_of_fitness_whole_pop_from_each_gen, label="Evolution of the mean fitness of the whole population")
# plt.xlabel('Generation')
# plt.ylabel('Mean of fitnesses of all individuals')
# plt.show()

# # PLOT 4 : all 3 curves 
# plt.figure(figsize=(10,8))
# plt.plot(generation_indices[::50], best_fitness_from_each_gen[::50], 'b', label="Evolution of the best fitness")
# plt.plot(generation_indices[::50], mean_fitness_first_X_ind_from_each_gen[::50], 'r', label=f"Evolution of the mean fitness of the {X} best individuals")
# plt.plot(generation_indices[::50], mean_of_fitness_whole_pop_from_each_gen[::50], 'g', label="Evolution of the mean fitness of the whole population")
# plt.xlabel('Génération')
# plt.title(f"Evolution des fitness au cours des {nb_generations} générations pour une population de {N} individus")
# plt.legend()
# plt.show()


# Create passengers
Transfer, Passenger, Nodes = createPassengersTransfersBatch(2,2)

print(TransferNodesSequence(Transfer, Passenger, 1, [1,2,0]))



# # Create graph according to user
# G = createGraphInstance(Nodes,1,10,20,100,2)

# # Set weights for fuel cost and time cost
# w_f = 0.4  # Weight for fuel cost
# w_t = 0.6  # Weight for time cost

# # Find the shortest path using the combined weight function

# source = Nodes[1]
# target = Nodes[4]
# path = nx.dijkstra_path(G, source=source, target=target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))
# path_length = nx.dijkstra_path_length(G, source=source, target=target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)) 

# print(f"Path from node {source} to node {target} is: {path}")

# # Position the nodes using a layout
# pos = nx.spring_layout(G)

# # Draw the graph
# plt.figure()
# nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')

# # Prepare edge labels with both weights
# edge_labels = {(u, v): f'({d["fuel_cost"]}, {d["time_cost"]})' for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# # Display the plot with informative box
# plt.title('Graph of the current scenario')
# plt.show()

