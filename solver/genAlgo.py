import random
import numpy as np
from .routing import RoutingAlgorithm

# FITNESS FUNCTION 

def Fitness(individual, graph, transfer_LUT, passenger_LUT, w_f, w_t):

    # Define fitness of the individual 
    individual_fitness = 0

    n_transfer = 1 # Transfer ID

    # Iterate over the chromosomes (transfers) in the individual
    for chromosome in individual:

        # Retrieve the overall path of the transfer
        transfer_path = RoutingAlgorithm(chromosome, graph, n_transfer, transfer_LUT, passenger_LUT, w_f, w_t)

        chromosome_cost = compute_costs_transfer(transfer_path, graph)

        # Add a penalty for every passenger if he has arrived later that its request (stored in the dictionary)
        penalty_cost = compute_penalty_transfer(transfer_path, graph, chromosome, passenger_LUT)

        # Sum up to the individual fitness
        individual_fitness += chromosome_cost + penalty_cost

        # Increase transfer number according to index of chromosome + 1
        n_transfer += 1

    return individual_fitness

def compute_costs_transfer(transfer_path, graph): 
    # Given the overall path of the transfer, we compute its cost (fuel + time)
    total_cost = 0

    filtered_transfer_path = filter_transfer(transfer_path) 
    transfer = filtered_transfer_path

    for s in range(len(transfer)-1) : 
        stop1 = transfer[s]
        stop2 = transfer[s+1]

        edge_data = graph.get_edge_data(stop1, stop2) # fetch the weights from the graph for the edge (stop1, stop2)

        fuel, time = edge_data['fuel_cost'], edge_data['time_cost']
        
        total_cost += fuel + time 
    return total_cost

    return 0 

def filter_transfer(transfer_path): 
    filtered_transfer = []
    i = 0 
    while i < len(transfer_path): 
        if transfer_path[i] == 'EOP': 
            i+=2 
        else: 
            filtered_transfer.append(transfer_path[i])
            i+=1 

    return filtered_transfer 

def compute_penalty_transfer(transfer_path, graph, chromosome, passengers_dict): 

    total_penalty = 0

    filtered_transfer = filter_transfer2(transfer_path) #list of sublists for each passenger 
    transfer_path = filtered_transfer

    passengers_in_chromosome = [i for i in chromosome if i != 0] #cleaning the zeros 
    cpt = 0

    for passenger_path in transfer_path : 
        penalty_cost = 0
        passenger_index = passengers_in_chromosome[cpt]

        for s in range(len(passenger_path)-1) : 
            stop1 = passenger_path[s]
            stop2 = passenger_path[s+1]

            edge_data = graph.get_edge_data(stop1, stop2) # fetch the weights from the graph for the edge (stop1, stop2)

            time = edge_data['time_cost']
            
        time_request = passengers_dict[passenger_index][2]

        penalty_cost = time - time_request
        total_penalty += penalty_cost

        cpt+=1

    return total_penalty

def filter_transfer2(transfer_path):
    
    input_list = transfer_path[1:]

    try:
        last_eop_index = len(input_list) - 1 - input_list[::-1].index('EOP')
    except ValueError:
        # EOP NOT FOUND, I just return the list without the first element 
        return []

    list_before_last_eop = input_list[:last_eop_index] #get the list before the last 'EOP'

    # splitting the list before the last EOP 
    result = []
    temp_list = []
    for i in list_before_last_eop:
        if i == 'EOP':
            if temp_list:
                result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    if temp_list:
        result.append(temp_list)

    return result

def compute_mean_fitness(individuals, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t):
    """
    Computes the mean fitness of a list of individuals.
    
    @param individuals: a list of individuals/solutions
    @param fitness: the numeric function that evaluates the "goodness" of a solution
    @param graph: the instance
    
    @return: the mean fitness of the individuals
    """

    fitnesses = [Fitness(ind, graph, transfer_LUT, passenger_LUT, w_f, w_t) for ind in individuals]
    mean_fitness = np.mean(fitnesses)
    
    return mean_fitness

# MUTATION

def MutationCustomDARPT(child, p):
        # Overcoming probability of mutation
        if p > random.uniform(0.0,1.0):
            # Randomly choose a transfer to be mutated
            idMutatedTransfer = random.randint(0,len(child)-1)
            # Apply custom mutation to the selected transfer -> shuffle the passengers in it
            random.shuffle(child[idMutatedTransfer])

        return child

# RECOMBINATION

def CrossoverCustomDARPT(parent1, parent2):

    # Save transfers dimension, equal for both parents
    dims = [len(parent1[i]) for i in range(len(parent1))]

    # Flatten list of lists in a single list
    par1 = [x for xs in parent1 for x in xs]
    par2 = [x for xs in parent2 for x in xs]

    # Pick the size of the flattened list
    size = len(par1)

    # Choose two random crossover points
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    # Initialize child with zeros
    child_flat = [0] * size
    
    # Copy a slice from the first parent
    child_flat[cxpoint1:cxpoint2] = par1[cxpoint1:cxpoint2]
    
    # Create a list of the remaining elements from the second parent
    remaining_elements = [item for item in par2 if item not in child_flat]
    
    # Fill remaining slots with the remaining elements from the second parent
    current_index = 0
    i = 0
    while i < (size) and current_index < len(remaining_elements):
        # Add random check for filling the list
        if random.uniform(0.0,1.0) > 0.5: # Fill starting from the beginning
            # Fill only if the i-th element is a zero
            if child_flat[i] == 0:
                child_flat[i] = remaining_elements[current_index]
                current_index += 1
        else: # Fill starting from the end
            # Fill only if the i-th element is a zero
            if child_flat[size-1-i] == 0:
                child_flat[size-1-i] = remaining_elements[current_index]
                current_index += 1
        i += 1

    # Reconstruct list of lists for child
    child = []
    start_index = 0
    for length in dims:
        sublist = child_flat[start_index:start_index + length]
        child.append(sublist)
        start_index += length

    return child
    
# SELECTION 

# To compare different strategies, we will implement a roulette wheel selection and a tournament selection 
# Then, we'll get to decide which one is the most suitable for our DARP problem 

# 1 - ROULETTE WHEEL SELECTION 

def roulette_wheel_selection(population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t): 
    """
    @param population, a pool of individuals/solutions 
    @param fitness, the numeric function that evaluates the "goodness" of a solution 
    @param graph, the instance 
    return a list of selected solutions/individuals, according to the roulette wheel selection
            ##and a list of the corresponding ids 
    """

    #individuals_by_id = {}
    selected_ind = []
    #selected_ids = []
    #sum_of_fitnesses = 0 
    fitnesses = []

    for ind in population: 
        fitnesses.append(Fitness(ind, graph, transfer_LUT, passenger_LUT, w_f, w_t))
        #print("fitnesses list", fitnesses)

    sum_probabilities = 0
    probabilities = {}

    sum_of_fitnesses = sum(fitnesses)
    #print("sum of fitnesses", sum_of_fitnesses)

    id = 0 
    #print("probabilities:", probabilities)
    for ind in population: 
        
        #print("ind",id, ":" ,ind)
        """
        probabilities[ind] = sum_probabilities + fitnesses[cpt]/sum_of_fitnesses
        sum_probabilities += probabilities[ind]
        cpt+=1
        """

        probabilities[id] = sum_probabilities + fitnesses[id] / sum_of_fitnesses
        sum_probabilities += probabilities[id]
        id+=1
    
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda x:x[1]))
    random_number = random.random()
    
    id = 0
    for ind in population: 
        if random_number > sorted_probabilities[id] : 
            selected_ind.append(ind)
            #selected_ids.append(id)
        id += 1
    return selected_ind# ,selected_ids

    
    
# 2 - TOURNAMENT SELECTION 

def tournament_selection(population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, tournament_size): 

    """
    @param population, a pool of individuals/solutions 
    @param fitness, the numeric function that evaluates the "goodness" of a solution 
    @param graph, the instance 
    @param tournamen_size, the nb of individuals to participate in each tournament 
    return a list of selected solutions/individuals, according to the simple tournament selection process based on fitness 
    """

    selected_ind = []
    population_size = len(population)

    for _ in range(population_size):
        tournament = random.sample(population, tournament_size) # random select of a sample individuals in the pop to participate in the tournament 
        
        tournament_fitnesses = [Fitness(ind, graph, transfer_LUT, passenger_LUT, w_f, w_t) for ind in tournament]
        
        best_individual = tournament[tournament_fitnesses.index(max(tournament_fitnesses))]
        
        if best_individual not in selected_ind:
            selected_ind.append(best_individual)

    return selected_ind


# GENERATE NEXT GENERATION (CHILDREN OF SELECTED PARENTS)
def generate_next_generation(parent_population, Fitness, nb_individuals, selection_process, proba_mutation, graph, transfer_LUT, passenger_LUT, w_f, w_t): 
    """
    Combining all processes, it generates a new population from a parent one (through crossover, mutation, selection)
    It will be called at each iteration of the algorithm
    """
    children_pop = []

    for i in range(nb_individuals): 
        #print("i", i)

        # 1 : SELECTION 

        if selection_process == "roulette" : 
            selected_individuals = roulette_wheel_selection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t)

            while(len(selected_individuals) < 2) : 
                selected_individuals = roulette_wheel_selection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t)

        elif selection_process == "tournament" : 
            selected_individuals = tournament_selection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, 3)

            while(len(selected_individuals) < 2) : 
                selected_individuals = tournament_selection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, 3)

        random_picked_parents = random.sample(selected_individuals, 2)
        parent1 = random_picked_parents[0]
        parent2 = random_picked_parents[1]
        
        # 2 : CROSSOVER
        child = CrossoverCustomDARPT(parent1, parent2) 

        # 3 : MUTATION
        child_mutated = MutationCustomDARPT(child, proba_mutation)

        children_pop.append(child_mutated)
        
    # add the best solution from the past generation 
    sorted_parent_population = sorted(parent_population, key=lambda ind: Fitness(ind, graph, transfer_LUT, passenger_LUT, w_f, w_t))
    best_parent = sorted_parent_population[0]
    children_pop[nb_individuals-1] = best_parent

    return children_pop