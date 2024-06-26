import os
import numpy as np
import matplotlib.pyplot as plt
from instance_gen import *
from solver import *

# Algorithm parameters 
N = 20                      #nb of individuals in the initial population
nb_generations = 100        # nb of new generations in the population
proba_mutation = 0.4        #proba for an ind to be muted 
w_f = 0.4                   # Weight for fuel cost
w_t = 0.6                   # Weight for time cost

# Instance creation
Transfer, Passenger, Nodes = createPassengersTransfersBatch(20,4)
G = createGraphInstance(Nodes,1,10,20,100,2)

# Fetch transfers' capacities
vehicles_capacities = [Transfer[transfer][2] for transfer in Transfer]  

# Generate initial population
initial_pop = generate_initial_pop(N, len(Passenger), vehicles_capacities)

# Selection processes
selected_individuals_by_roulette = RouletteWheelSelection(initial_pop, Fitness, G, Transfer, Passenger, w_f, w_t) 
selected_individuals_by_tournament = TournamentSelection(initial_pop, Fitness, G, Transfer, Passenger, w_f, w_t, 3)
print(selected_individuals_by_tournament)

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
selection = "roulette" #tournament"
initial_population = generate_initial_pop(nb_individuals=N, nb_passengers=len(Passenger), capacities=vehicles_capacities)
parent_population = initial_population

# 2 - Prepare data for performance analytics 
best_fitness_from_each_gen = []
mean_fitness_first_X_ind_from_each_gen = []
X = 5
mean_of_fitness_whole_pop_from_each_gen = []

# 3 - Running the algorithm
while (nb_iterations <= nb_generations):     

    child_population = GenerateNextGeneration(parent_population, Fitness, N, selection, proba_mutation, G, Transfer, Passenger, w_f, w_t)
    #print(child_population)
    parent_population = child_population 
    nb_iterations += 1

    # Fech data fro performance metrics

    # KPI1 : Fitness of the best individual of each generation 
    sorted_child_population = sorted(child_population, key=lambda ind: Fitness(ind, G, Transfer, Passenger, w_f, w_t))
    test = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in sorted_child_population]
    #print(test)
    best_ind_from_child_pop = sorted_child_population[0]
    best_fitness = Fitness(best_ind_from_child_pop, G, Transfer, Passenger, w_f, w_t)
    #print(best_fitness)
    #first_non_zero = next((value for value in test if value != 0), None)
    #best_fitness = first_non_zero
    best_fitness_from_each_gen.append(best_fitness)

    # KPI2 : Fitness of the first X individuals of each generation for the mean 
    top_individuals = sorted_child_population[:X]
    top_fitnesses = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in top_individuals]
    mean_of_the_fitness_of_first_X_individuals = np.mean(top_fitnesses)
    mean_fitness_first_X_ind_from_each_gen.append(mean_of_the_fitness_of_first_X_individuals)

    # KPI3 : Fitness of all individuals of each generation 
    fitnesses_of_whole_gen = [Fitness(ind, G, Transfer, Passenger, w_f, w_t) for ind in child_population]
    mean_fitness_of_whole_pop = np.mean(fitnesses_of_whole_gen)
    mean_of_fitness_whole_pop_from_each_gen.append(mean_fitness_of_whole_pop)


# Plotting
generation_indices = range(nb_generations+1)

# NB : try with different selection processes 

#print(best_fitness_from_each_gen)
#print(mean_fitness_first_X_ind_from_each_gen)
#print(mean_of_fitness_whole_pop_from_each_gen)

# PLOT 1 : evolution of the fitness along the reproduction process (best fitness for each gen)
plt.plot(generation_indices[::5], best_fitness_from_each_gen[::5], label="Evolution of the best fitness of each generation")
plt.xlabel('Generation')
plt.ylabel('Best fitness')
plt.title(f"Evolution of the best fitness over the {nb_generations} generations for a population of {N} individuals")
#figure_path = os.path.join(r'C:\Users\marco\Documents\GitHub\DARP_Project\plots', 'best_fitness.png')
figure_path = os.path.join(r'C:\Users\mathi\OneDrive\Documents\0_ECOLE\2_POLITO\ORTA\FINAL PROJECT', 'plot1.png')
plt.savefig(figure_path)
plt.close()

# PLOT 2 : evolution of the mean of the fitness of the X best individuals along the reproduction process 
plt.plot(generation_indices[::5], mean_fitness_first_X_ind_from_each_gen[::5], label='Evolution of the mean of the fitness of the f"{X} first best individuals')
plt.xlabel('Generation')
plt.ylabel(f'Mean of fitnesses of the {X} best individuals of the generation')
plt.title(f"Evolution of the mean fitness of the {X} best individuals over the {nb_generations} generations for a population of {N} individuals")
#figure_path = os.path.join(r'C:\Users\marco\Documents\GitHub\DARP_Project\plots', 'mean_best_fitness.png')
figure_path = os.path.join(r'C:\Users\mathi\OneDrive\Documents\0_ECOLE\2_POLITO\ORTA\FINAL PROJECT', 'plot2.png')
plt.savefig(figure_path)
plt.close()

# PLOT 3 : evolution of the mean of the fitness of all individuals along the reproduction process 
plt.plot(generation_indices[::5], mean_of_fitness_whole_pop_from_each_gen[::5], label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Mean of fitnesses of all individuals')
plt.title(f"Evolution of the mean fitness over the {nb_generations} generations for a population of {N} individuals")
#figure_path = os.path.join(r'C:\Users\marco\Documents\GitHub\DARP_Project\plots', 'mean_fitness.png')
figure_path = os.path.join(r'C:\Users\mathi\OneDrive\Documents\0_ECOLE\2_POLITO\ORTA\FINAL PROJECT', 'plot3.png')
plt.savefig(figure_path)
plt.close()

# PLOT 4 : all 3 curves 
plt.figure(figsize=(14,12))
plt.plot(generation_indices[::5], best_fitness_from_each_gen[::5], 'b', label="Evolution of the best fitness")
plt.plot(generation_indices[::5], mean_fitness_first_X_ind_from_each_gen[::5], 'r', label=f"Evolution of the mean fitness of the {X} best individuals")
plt.plot(generation_indices[::5], mean_of_fitness_whole_pop_from_each_gen[::5], 'g', label="Evolution of the mean fitness of the whole population")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title(f"Evolution of the fitnesses over the {nb_generations} generations for a population of {N} individuals")
plt.legend()
#figure_path = os.path.join(r'C:\Users\marco\Documents\GitHub\DARP_Project\plots', 'fitness_evolution.png')
figure_path = os.path.join(r'C:\Users\mathi\OneDrive\Documents\0_ECOLE\2_POLITO\ORTA\FINAL PROJECT', 'plot4.png')
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
# figure_path = os.path.join(r'C:\Users\marco\Documents\GitHub\DARP_Project\plots', 'graph.png')
# figure_path = os.path.join(r'C:\Users\mathi\OneDrive\Documents\0_ECOLE\2_POLITO\ORTA\FINAL PROJECT', 'plot5.png')
# plt.savefig(figure_path)

