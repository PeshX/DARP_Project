import random
import numpy as np
from .routing import RoutingAlgorithm 

def Fitness(individual, graph, transfer_LUT, passenger_LUT, w_f, w_t):

    """
    Computes the fitness of an individual.
    
    @param individuals: a list of lists representing a solution
    @param graph: the instance of the graph built with NetworkX
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges
    
    @return: the fitness of the individual
    """

    # Define fitness of the individual 
    individual_fitness = 0

    n_transfer = 1 # Transfer ID

    # Iterate over the chromosomes (transfers) in the individual
    for chromosome in individual:

        # Check if the transfer is not empty, if so skip to the next one
        if all(v == 0 for v in chromosome): 
            # Increase transfer number according to index of chromosome + 1
            n_transfer += 1
        else:
            # Retrieve the overall path of the transfer
            overall_path = RoutingAlgorithm(chromosome, graph, n_transfer, transfer_LUT, passenger_LUT, w_f, w_t)
            
            # Obtain transfer path and passengers' paths
            transfer_path, passenger_paths = PathTransferPassenger(overall_path)

            # Compute cost for the overall trasnfer path
            chromosome_cost = ComputeCostTransfer(transfer_path, graph, w_f, w_t)

            # Add a penalty for every passenger if he has arrived later that its request (stored in the dictionary)
            penalty_cost = ComputePenaltyTransfer(passenger_paths, graph, chromosome, passenger_LUT)

            # Sum up to the individual fitness
            individual_fitness += chromosome_cost + penalty_cost

            # Increase transfer number according to index of chromosome + 1
            n_transfer += 1

    return individual_fitness


def PathTransferPassenger(overall_path):

    """
    Computes the transfer path and the passengers' paths for the fitness computation
    
    @param overall_path: general path given by the RoutingAlgorithm method
    
    @return: one list fro the transfer's path and a list of lists fro the partial paths of the passengers
    """    
    
    # Find the indices of 'SOP' and 'EOP' in the list
    sop_indices = [index for index, value in enumerate(overall_path) if value == 'SOP']
    eop_indices = [index for index, value in enumerate(overall_path) if value == 'EOP']

    # Segments the list based on 'SOP' and 'EOP' indices
    segmented_list = [overall_path[sop_indices[i]+1:eop_indices[i]] for i in range(len(eop_indices))]

    # Passenger path below

    filtered_segmentation = []

    for su_list in (segmented_list):
        filtered_segmentation.extend([[elem for elem in (su_list) if (elem!='SOP' and elem!='EOP')]])
        

    true_filtered_segmentation=[]
    for su_list in (filtered_segmentation):
        
        result=[su_list[0]]
        for i in range(1, len(su_list)):
                if su_list[i] != su_list[i - 1]:
                    result.append(su_list[i])
                    
        true_filtered_segmentation.extend([result])
        
    # Transfer path below

    # Indexes of EOP
    k = [value for value in  range(len(overall_path)) if overall_path[value] =='EOP']

    k.extend([value2-1 for value2 in k])

    # Indexes of SOP
    j = [value for value in  range(len(overall_path)) if overall_path[value] =='SOP']
    k.extend(j)
    transfer_path = [elem for i, elem in enumerate(overall_path) if i not in k]


    return transfer_path, true_filtered_segmentation


def ComputeCostTransfer(transfer_path, graph, w_f, w_t): 

    """
    Computes the cost of the transfer path, composed of the combination of fuel and time.
    N.B. A penalty is added by the passengers later
    
    @param transfer_path: the list of the overall path of a single transfer
    @param graph: the instance of the graph built with NetworkX
    
    @return: the total cost of the transfer
    """

    total_cost = 0

    for s in range(len(transfer_path)-1) : 
        stop1 = transfer_path[s]
        stop2 = transfer_path[s+1]

        edge_data = graph.get_edge_data(stop1, stop2) # fetch the weights from the graph for the edge (stop1, stop2)

        fuel, time = edge_data['fuel_cost'], edge_data['time_cost']
        
        total_cost += w_f * fuel + w_t * time 

    return total_cost


def ComputePenaltyTransfer(transfer_path, graph, chromosome, passengers_dict): 

    """
    Computes the penalty associated to the transfer coming from the time_request condition of each passenger
    
    @param transfer_path: the list of the overall path of a single transfer
    @param graph: the instance of the graph built with NetworkX
    @param chromosome: the list of the passengers within the current transfer
    @param passengers_dict: dictionary of the passengers
    
    @return: the total time penalty of the transfer
    """

    total_delay = 0

    passengers_in_chromosome = [i for i in chromosome if i != 0] #cleaning the zeros 
    cpt = 0

    for passenger_path in transfer_path : 
        penalty_cost = 0
        service_time = 0
        passenger_index = passengers_in_chromosome[cpt]

        # Compute total service time for each passenger
        for s in range(len(passenger_path)-1) : 
            stop1 = passenger_path[s]
            stop2 = passenger_path[s+1]

            edge_data = graph.get_edge_data(stop1, stop2) # fetch the weights from the graph for the edge (stop1, stop2)
            time = edge_data['time_cost']
            service_time += time     

        # Fetch passenger time request    
        time_request = passengers_dict[passenger_index][2]

        # Compute total delay of service for all passengers
        penalty_cost = service_time - time_request
        penalty_cost = max(penalty_cost,0) # no penalty for on-time service
        total_delay += penalty_cost

        cpt+=1

    # Generate a penalty to be summed up to the transfer cost
    delta = 0.3
    pure_penalty = delta * (total_delay/len(passengers_in_chromosome))

    return pure_penalty


def ComputeMeanFitness(individuals, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t):

    """
    Computes the mean fitness of a list of individuals.
    
    @param individuals: a list of individuals/solutions
    @param fitness: the numeric function that evaluates the "goodness" of a solution
    @param graph: the instance of the graph built with NetworkX
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges
    
    @return: the mean fitness of the individuals
    """

    fitnesses = [Fitness(ind, graph, transfer_LUT, passenger_LUT, w_f, w_t) for ind in individuals]
    mean_fitness = np.mean(fitnesses)
    
    return mean_fitness


def MutationCustomDARPT(child, p):
        
    """
    Performs a mutation operator on a newly generated child, acting on its genes (passengers)
    
    @param child: newly generated individual (complete solution)
    @param p: probability of having a mutation
    
    @return: the individual passed where one of its chromosome (transfer) has its genes permuted
    """

    # Overcoming probability of mutation
    if p > random.uniform(0.0,1.0):
        # Randomly choose a transfer to be mutated
        idMutatedTransfer = random.randint(0,len(child)-1)
        # Apply custom mutation to the selected transfer -> shuffle the passengers in it
        random.shuffle(child[idMutatedTransfer])

    return child


def CrossoverCustomDARPT(parent1, parent2):

    """
    Generates one child from the recombination operator acting on two parent
    
    @param parent1: one individual
    @param parent2: another individual
    
    @return: the child generated by cross-mixing a list of genes from parent1 into parent2
    N.B. Feasibility is preserved with this custom implementation
    """

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

# To compare different strategies, we will implement a roulette wheel selection and a tournament selection 
# Then, we'll get to decide which one is the most suitable for our DARPT  

def RouletteWheelSelection(population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t): 

    """
    @param population, a pool of individuals/solutions 
    @param fitness, the numeric function that evaluates the "goodness" of a solution 
    @param graph: the instance of the graph built with NetworkX
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges

    @return a list of selected solutions/individuals, according to the roulette wheel selection
            and a list of the corresponding ids 
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


def TournamentSelection(population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, tournament_size): 

    """
    @param population: a pool of individuals/solutions 
    @param Fitness: the numeric function that evaluates the "goodness" of a solution 
    @param graph: the instance of the graph built with NetworkX
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges
    @param tournamen_size, the nb of individuals to participate in each tournament 

    @return a list of selected solutions/individuals, according to the simple tournament selection process based on fitness 
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


def GenerateNextGeneration(parent_population, Fitness, nb_individuals, selection_process, proba_mutation, graph, transfer_LUT, passenger_LUT, w_f, w_t): 

    """
    Combining all processes, it generates a new population from a parent one (through crossover, mutation, selection)
    It will be called at each iteration of the algorithm

    @param parent_population: the whole parent population of individuals
    @param Fitness: the numeric function that evaluates the "goodness" of a solution
    @param nb_individuals: the number of individuals in the current population
    @param selection_process: the type of selection process to be used
    @param proba_mutation: probability of having a mutation
    @param graph: the instance of the graph built with NetworkX
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges  

    @return: the children populationg generated according to this process  
    """
    children_pop = []

    for i in range(nb_individuals): 

        # 1 : SELECTION 
        if selection_process == "roulette" : 
            selected_individuals = RouletteWheelSelection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t)

            while(len(selected_individuals) < 2) : 
                selected_individuals = RouletteWheelSelection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t)

        elif selection_process == "tournament" : 
            selected_individuals = TournamentSelection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, 3)

            while(len(selected_individuals) < 2) : 
                selected_individuals = TournamentSelection(parent_population, Fitness, graph, transfer_LUT, passenger_LUT, w_f, w_t, 3)

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