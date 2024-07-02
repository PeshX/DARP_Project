import os
import numpy as np
import matplotlib.pyplot as plt
from instance_gen import *
from solver import *

# Algorithm parameters 
N = 20                      #nb of individuals in the initial population
nb_generations = 500        # nb of new generations in the population
proba_mutation = 0.4        #proba for an ind to be muted 
w_f = 0.4                   # Weight for fuel cost in the fitness
w_t = 0.6                   # Weight for time cost in the fitness
nb_passengers = 20          # Number of passengers  
nb_transfers = 4            # Number of transfers -> consider a capacity from 5 to 10
min_fuel = 1                # min fuel weight for edges
max_fuel = 10               # max fuel weight froi edges
min_time = 5                # min time weight for edges
max_time = 60               # max time weight for edges
min_degree = 2              # min degree of graph's nodes
user_path = r'C:\Users\marco\Documents\GitHub\DARP_Project\plots' # TODO: fill used path with 'r' at the beginning

# Instance creation
Transfer, Passenger, Nodes = createPassengersTransfersBatch(nb_passengers,nb_transfers)
G = createGraphInstance(Nodes,min_fuel,max_fuel,min_time,max_time,min_degree)

# Fetch transfers' capacities
vehicles_capacities = [Transfer[transfer][2] for transfer in Transfer]  

# Generate initial population
initial_pop = generate_initial_pop(N, len(Passenger), vehicles_capacities)

# Selection processes
selected_individuals_by_roulette = RouletteWheelSelection(initial_pop, Fitness, G, Transfer, Passenger, w_f, w_t) 
selected_individuals_by_tournament = TournamentSelection(initial_pop, Fitness, G, Transfer, Passenger, w_f, w_t, 3)

"""
---- UNCOMMENT TO COMPARE THE TWO SELECTION APPROACHES ---- 

print("WITH ROULETTE SELECTION, THE SELECTIONED INDIVIDUALS ARE:")
for ind in selected_individuals_by_roulette: 
    print(ind)

print("WITH TOURNAMENT SELECTION, THE SELECTED INDIVIDUALS ARE:")
for ind in selected_individuals_by_tournament: 
    print(ind)

To compare both approach, we can compare mean of fitnesses of the selected individuals  
The bigger the sum is, the worst are the individual, that's a first metric

mean_fitness_roulette = compute_mean_fitness(selected_individuals_by_roulette, Fitness, G, Transfer, Passenger, w_f, w_t)
mean_fitness_tournament = compute_mean_fitness(selected_individuals_by_tournament, Fitness, G, Transfer, Passenger, w_f, w_t)

print("mean fitness roulette: ", mean_fitness_roulette)
print("mean fitness tournament: ", mean_fitness_tournament)
"""

# GENETIC ALGORITHM -------------------------------------------------------------------------------------------------------------

# 1 - Initialization 

nb_iterations = 0 
selection1 = "roulette" 
selection2 = "tournament"
initial_population = generate_initial_pop(nb_individuals=N, nb_passengers=len(Passenger), capacities=vehicles_capacities)

# Roulette selection 
parent_population1 = initial_population

# Tournament selection 
parent_population2 = initial_population

# 2 - Prepare data for performance analytics 

# Roulette selection
best_fitness_from_each_gen1 = []
mean_fitness_first_X_ind_from_each_gen1 = []
X = 5
mean_of_fitness_whole_pop_from_each_gen1 = []

# Tournament selection
best_fitness_from_each_gen2 = []
mean_fitness_first_X_ind_from_each_gen2 = []
mean_of_fitness_whole_pop_from_each_gen2 = []

# 3 - Running the algorithm
while (nb_iterations <= nb_generations):     

    # Roulette selection
    child_population1 = GenerateNextGeneration(parent_population1, Fitness, N, selection1, proba_mutation, G, Transfer, Passenger, w_f, w_t)
    parent_population1 = child_population1

    # Tournament selection
    child_population2 = GenerateNextGeneration(parent_population2, Fitness, N, selection2, proba_mutation, G, Transfer, Passenger, w_f, w_t)
    parent_population2 = child_population2 

    nb_iterations += 1

    # Fech data for performance metrics

    # KPI1 : Fitness of the best individual of each generation 
    # Roulette selection 
    sorted_child_population1 = sorted(child_population1, key=lambda ind: Fitness(ind, G, Transfer, Passenger, w_f, w_t))
    best_ind_from_child_pop1 = sorted_child_population1[0]
    best_fitness1 = Fitness(best_ind_from_child_pop1, G, Transfer, Passenger, w_f, w_t)
    best_fitness_from_each_gen1.append(best_fitness1)
    
    # Tournament selection
    sorted_child_population2 = sorted(child_population2, key=lambda ind: Fitness(ind, G, Transfer, Passenger, w_f, w_t))
    best_ind_from_child_pop2 = sorted_child_population2[0]
    best_fitness2 = Fitness(best_ind_from_child_pop2, G, Transfer, Passenger, w_f, w_t)
    best_fitness_from_each_gen2.append(best_fitness2)

    # KPI2 : Fitness of the first X individuals of each generation for the mean 
    # Roulette selection
    top_individuals1 = sorted_child_population1[:X]
    top_fitnesses1 = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in top_individuals1]
    mean_of_the_fitness_of_first_X_individuals1 = np.mean(top_fitnesses1)
    mean_fitness_first_X_ind_from_each_gen1.append(mean_of_the_fitness_of_first_X_individuals1)

    # Tournament selection 
    top_individuals2 = sorted_child_population2[:X]
    top_fitnesses2 = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in top_individuals2]
    mean_of_the_fitness_of_first_X_individuals2 = np.mean(top_fitnesses2)
    mean_fitness_first_X_ind_from_each_gen2.append(mean_of_the_fitness_of_first_X_individuals2)

    # KPI3 : Fitness of all individuals of each generation 
    # Roulette selection
    fitnesses_of_whole_gen1 = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in child_population1]
    mean_fitness_of_whole_pop1 = np.mean(fitnesses_of_whole_gen1)
    mean_of_fitness_whole_pop_from_each_gen1.append(mean_fitness_of_whole_pop1)

    # Tournament selection 
    fitnesses_of_whole_gen2 = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in child_population2]
    mean_fitness_of_whole_pop2 = np.mean(fitnesses_of_whole_gen2)
    mean_of_fitness_whole_pop_from_each_gen2.append(mean_fitness_of_whole_pop2)


# Plotting
generation_indices = range(nb_generations+1)

# PLOT 1 : evolution of the fitness along the reproduction process (best fitness for each gen)
# Roulette selection
plt.plot(generation_indices[::25], best_fitness_from_each_gen1[::25], label="Evolution of the best fitness of each generation")
plt.xlabel('Generation')
plt.ylabel('Best fitness')
plt.title(f"Evolution of the best fitness for a population of {N} individuals")
figure_path = os.path.join(user_path, 'plot1_r.png')
plt.savefig(figure_path)
plt.close()

# Tournament selection 
plt.plot(generation_indices[::25], best_fitness_from_each_gen2[::25], label="Evolution of the best fitness of each generation")
plt.xlabel('Generation')
plt.ylabel('Best fitness')
plt.title(f"Evolution of the best fitness for a population of {N} individuals")
figure_path = os.path.join(user_path, 'plot1_t.png')
plt.savefig(figure_path)
plt.close()

# PLOT 2 : evolution of the mean of the fitness of the X best individuals along the reproduction process 
# Roulette selection
plt.plot(generation_indices[::25], mean_fitness_first_X_ind_from_each_gen1[::25], label='Evolution of the mean of the fitness of the f"{X} first best individuals')
plt.xlabel('Generation')
plt.ylabel(f'Mean of fitnesses of the {X} best individuals')
plt.title(f"Evolution of the mean fitness of the {X} best individuals")
figure_path = os.path.join(user_path, 'plot2_r.png')
plt.savefig(figure_path)
plt.close()

# Roulette selection
plt.plot(generation_indices[::25], mean_fitness_first_X_ind_from_each_gen2[::25], label='Evolution of the mean of the fitness of the f"{X} first best individuals')
plt.xlabel('Generation')
plt.ylabel(f'Mean of fitnesses of the {X} best individuals')
plt.title(f"Evolution of the mean fitness of the {X} best individuals")
figure_path = os.path.join(user_path, 'plot2_t.png')
plt.savefig(figure_path)
plt.close()

# PLOT 3 : evolution of the mean of the fitness of all individuals along the reproduction process 
# Roulette selection
plt.plot(generation_indices[::25], mean_of_fitness_whole_pop_from_each_gen1[::25], label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Mean of fitnesses of all individuals')
plt.title(f"Evolution of the mean fitness of the whole population")
figure_path = os.path.join(user_path, 'plot3_r.png')
plt.savefig(figure_path)
plt.close()

# Tournament selection
plt.plot(generation_indices[::25], mean_of_fitness_whole_pop_from_each_gen2[::25], label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Mean of fitnesses of all individuals')
plt.title(f"Evolution of the mean fitness of the whole population")
figure_path = os.path.join(user_path, 'plot3_t.png')
plt.savefig(figure_path)
plt.close()

# PLOT 4 : all 3 curves 
# Roulette selection
plt.figure(figsize=(14,12))
plt.plot(generation_indices[::25], best_fitness_from_each_gen1[::25], 'b', label="Evolution of the best fitness")
plt.plot(generation_indices[::25], mean_fitness_first_X_ind_from_each_gen1[::25], 'r', label=f"Evolution of the mean fitness of the {X} best individuals")
plt.plot(generation_indices[::25], mean_of_fitness_whole_pop_from_each_gen1[::25], 'g', label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title(f"Evolution of the fitnesses over the {nb_generations} generations for a population of {N} individuals")
plt.legend()
figure_path = os.path.join(user_path, 'plot4_r.png')
plt.savefig(figure_path)
plt.close()

# Tournament selection
plt.figure(figsize=(14,12))
plt.plot(generation_indices[::25], best_fitness_from_each_gen2[::25], 'b', label="Evolution of the best fitness")
plt.plot(generation_indices[::25], mean_fitness_first_X_ind_from_each_gen2[::25], 'r', label=f"Evolution of the mean fitness of the {X} best individuals")
plt.plot(generation_indices[::25], mean_of_fitness_whole_pop_from_each_gen2[::25], 'g', label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title(f"Evolution of the fitnesses over the {nb_generations} generations for a population of {N} individuals")
plt.legend()
figure_path = os.path.join(user_path, 'plot4_t.png')
plt.savefig(figure_path)
plt.close()


# PLOT 5 : graph of the scenario (ONLY FOR SMALL INSTANCES FOR THE REPORT'S SAKE)

# pos = nx.spring_layout(G)
# plt.figure()
# nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
# # Prepare edge labels with both weights
# edge_labels = {(u, v): f'({d["fuel_cost"]}, {d["time_cost"]})' for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.title('Graph of the current scenario')
# figure_path = os.path.join(user_path, 'graph.png')
# plt.savefig(figure_path)


# PRINTING BEST SOLUTIONS OBTAINED ---------------------------------------------------------------------------------------

# Roulette selection 
final_pop_sorted_by_fitness1 = sorted(child_population1, key=lambda ind: Fitness(ind, G, Transfer, Passenger, w_f, w_t))
best_solution1 = final_pop_sorted_by_fitness1[0]
best_solution_fitness1 = Fitness(best_solution1, G, Transfer, Passenger, w_f, w_t)
print("Best solution obtained with the roulette wheel selection :", best_solution1)
print("Fitness of the best solution:", best_solution_fitness1)

# Tournament selection 
final_pop_sorted_by_fitness2 = sorted(child_population2, key=lambda ind: Fitness(ind, G, Transfer, Passenger, w_f, w_t))
best_solution2 = final_pop_sorted_by_fitness2[0]
best_solution_fitness2 = Fitness(best_solution2, G, Transfer, Passenger, w_f, w_t)
print("Best solution obtained with the tournament selection :", best_solution2)
print("Fitness of the best solution:", best_solution_fitness2)