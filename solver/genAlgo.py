import random
import numpy as np

# FITNESS FUNCTION 

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

def IntegerRandMutation(chromosome, individual, p, passengers_list):
    # Filter passenger list to avoid generating the id of a passenger already present in the chromosome
    filtered_list = [passenger_id for passenger_id in passengers_list if passenger_id not in chromosome]
    # Append '0' means that a specific passenger can be excluded from that specific chromosome
    filtered_list.append(0)
    print(filtered_list)
    # Iterate over all chromosome to randomly mutate its entries
    for i in range(0,len(chromosome)-1):
        if p > random.uniform(0.0,1.0):
            chromosome[i] = random.choice(filtered_list)

    return chromosome

def MutationCustomDARPT(child, p, num_transfers):
        # Overcoming probability of mutation
        if p > random.uniform(0.0,1.0):
            # Randomly choose a transfer to be mutated
            idMutatedTransfer = random.randint(0,num_transfers-1)
            # Apply custom mutation to the selected transfer -> shuffle the passengers
            random.shuffle(child[idMutatedTransfer])

        return child

# RECOMBINATION

def RecombIntegers(individuals, max_window_size):

    # Check on the maximum sieze of the window
    if max_window_size > int(len(individuals[0])/2):
            max_window_size = int(len(individuals[0])/2)

    # Find minimum length chromosome and split it in windows
    min_len = len(min(individuals,key=len))
    if min_len == len(individuals[0]):
        min_ch = 0
        max_ch = 1           
        split_index = np.random.randint(2,max_window_size, dtype=int)
        win1 = individuals[0][0:split_index]
        win2 = individuals[0][split_index:]
    else:
        min_ch = 1
        max_ch = 0
        split_index = np.random.randint(2,max_window_size, dtype=int)
        win1 = individuals[1][0:split_index]
        win2 = individuals[1][split_index:]

    # Computing indexes for the crossover of each window
    crx_index1 = random.randint(0,len(individuals[min_ch])-len(max([win1,win2],key=len)))
    crx_index2 = random.randint(crx_index1, len(individuals[max_ch])-len(win2))

    # Swap content of the windows with the second chromosome
    individuals[min_ch][0:split_index],  individuals[max_ch][crx_index1:(len(win1)+crx_index1)] = individuals[max_ch][crx_index1:(len(win1)+crx_index1)], individuals[min_ch][0:split_index]
    individuals[min_ch][split_index:],  individuals[max_ch][crx_index2:(len(win2)+crx_index2)] = individuals[max_ch][crx_index2:(len(win2)+crx_index2)], individuals[min_ch][split_index:] 

    return individuals

def RecombIntegers(parent1, parent2, window_size, nb_passengers):

    # TODO: see if it does make sense to do a check on the window's size

    

    # Pick from parent1 a random set of passenger_id
    passengers_cx = random.sample(list(range(1, nb_passengers+1)),window_size)

    # Iterate over the transfer in the individual
    # for transfer in range (1,len(parent2)):
    #     # TODO: search for second solution in each transfer if the choices are present
    #     parent2[transfer].index()
    transfer_dims = [len(parent2[i]) for i in range(len(parent2))]
    flattened_list = [x for xs in parent2 for x in xs]

    # Check on the zero value


    idx_passenger_in_transfer = [flattened_list.index(i) for i in passengers_cx] 
    



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
def generate_next_generation(parent_population, fitness, nb_individuals, selection_process, graph): 
    """
    Combining all processes, it generates a new population from a parent one (through crossover, mutation, selection)
    It will be called at each iteration of the algorithm
    """
    children_pop = []

    for i in range(nb_individuals-1): 

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

        # 3 : MUTATION

        
    # add the best solution from the past generation 
    sorted_parent_population = sorted(parent_population, key=lambda ind: fitness(ind, graph))
    best_parent = sorted_parent_population[0]
    children_pop[nb_individuals-1] = best_parent

    return children_pop