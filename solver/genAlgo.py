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


        break

        # Compute the overall cost of the transfer
        #chromosome_cost = 

        # Sum up to the individual fitness
        #individual_fitness += chromosome_cost

    return transfer_path


def fitness(individual, graph) : 
    """
    @param individual := a solution, id est a list of lists of integers (list of transfers)

    return an integer value : the better the solution, the lower the value 
    the integer value considers the consumption of the vehicle (fuel) and the bad quality of service (delays)

    """
    res = 0
    overall_consumption = 0
    overall_delay = 0

    for transfer in individual : 

        transfer = [value for value in transfer if value != 0] #delete the zeros bc they don't correspond to any passenger

        consumption = 0
        delay = 0

        # each transfer is an ordered list of integers : the passengers fetched by the vehicle 
        weights = []

        for stop in range(len(transfer)-1) : 
            passenger1 = transfer[stop]
            passenger2 = transfer[stop+1]

            edge_data = graph.get_edge_data(passenger1, passenger2) # fetch the weights from the graph for the edge (passenger1, passenger2)
            weights.append(edge_data)

        #print("weights =", weights)
        for w in weights : 
             if w != None : 
                #print("w=", w)
                #print("fuel cost", w['fuel_cost'])

           
                consumption += w['fuel_cost'] # up the consumption accordingly to the weight 
                delay += w['time_cost'] # up the delay accordingly to the weight 
        
        overall_consumption += consumption
        overall_delay += delay 
    
        res = overall_consumption + overall_delay
    return res 


def compute_mean_fitness(individuals, fitness, graph):
    """
    Computes the mean fitness of a list of individuals.
    
    @param individuals: a list of individuals/solutions
    @param fitness: the numeric function that evaluates the "goodness" of a solution
    @param graph: the instance
    
    @return: the mean fitness of the individuals
    """

    fitnesses = [fitness(ind, graph) for ind in individuals]
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

def roulette_wheel_selection(population, fitness, graph): 
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
        fitnesses.append(fitness(ind, graph))
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

def tournament_selection(population, fitness, graph, tournament_size): 

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
        
        tournament_fitnesses = [fitness(ind, graph) for ind in tournament]
        
        best_individual = tournament[tournament_fitnesses.index(max(tournament_fitnesses))]
        
        if best_individual not in selected_ind:
            selected_ind.append(best_individual)

    return selected_ind


# GENERATE NEXT GENERATION (CHILDREN OF SELECTED PARENTS)
def generate_next_generation(parent_population, fitness, nb_individuals, selection_process, proba_mutation, graph): 
    """
    Combining all processes, it generates a new population from a parent one (through crossover, mutation, selection)
    It will be called at each iteration of the algorithm
    """
    children_pop = []

    for i in range(nb_individuals): 
        #print("i", i)

        # 1 : SELECTION 

        if selection_process == "roulette" : 
            selected_individuals = roulette_wheel_selection(parent_population, fitness, graph)

            while(len(selected_individuals) < 2) : 
                selected_individuals = roulette_wheel_selection(parent_population, fitness, graph)

        elif selection_process == "tournament" : 
            selected_individuals = tournament_selection(parent_population, fitness, graph, 3)

            while(len(selected_individuals) < 2) : 
                selected_individuals = tournament_selection(parent_population, fitness, graph)

        random_picked_parents = random.sample(selected_individuals, 2)
        parent1 = random_picked_parents[0]
        parent2 = random_picked_parents[1]
        
        # 2 : CROSSOVER
        child = CrossoverCustomDARPT(parent1, parent2) 

        # 3 : MUTATION
        child_mutated = MutationCustomDARPT(child, proba_mutation)

        children_pop.append(child_mutated)
        
    # add the best solution from the past generation 
    sorted_parent_population = sorted(parent_population, key=lambda ind: fitness(ind, graph))
    best_parent = sorted_parent_population[0]
    #print("len:", len(children_pop))
    #print(nb_individuals-1)
    children_pop[nb_individuals-1] = best_parent

    return children_pop