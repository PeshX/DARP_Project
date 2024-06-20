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

        print("weights =", weights)
        for w in weights : 
             if w != None : 
                print("w=", w)
                print("fuel cost", w['fuel_cost'])

           
                consumption += w['fuel_cost'] # up the consumption accordingly to the weight 
                delay += w['time_cost'] # up the delay accordingly to the weight 
        
        overall_consumption += consumption
        overall_delay += delay 
    
        res = overall_consumption + overall_delay
    return res 

# MUTATION

def IntegerRandMutation(chromosome, p, passengers_list):
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

# RECOMBINATION

def RecombIntegers(individuals, p, passengers_list):

    # Find minimum length chromosome and split it
    min_len = len(min(individuals,key=len))
    if min_len == len(individuals[0]):
        min_ch = 0
        max_ch = 1
        split_index = np.random.randint(2,int(len(individuals[0])/2), dtype=int)
        win1 = individuals[0][0:split_index]
        win2 = individuals[0][split_index:]
    else:
        min_ch = 1
        max_ch = 0
        split_index = np.random.randint(2,int(len(individuals[0])/2), dtype=int)
        win1 = individuals[1][0:split_index]
        win2 = individuals[1][split_index:]

    print(win1,win2)

    print(min_ch)

    crx_index1 = random.randint(0,len(individuals[min_ch])-len(max([win1,win2],key=len)))
    crx_index2 = random.randint(crx_index1, len(individuals[max_ch])-len(win2))

    print(crx_index1,crx_index2)

    # Swap content of the windows with the second chromosome
    individuals[min_ch][0:split_index],  individuals[max_ch][crx_index1:(len(win1)+crx_index1)] = individuals[max_ch][crx_index1:(len(win1)+crx_index1)], individuals[min_ch][0:split_index]
    print(individuals)
    individuals[min_ch][split_index:],  individuals[max_ch][crx_index2:(len(win2)+crx_index2)] = individuals[max_ch][crx_index2:(len(win2)+crx_index2)], individuals[min_ch][split_index:] 

    return individuals

chromosome1 = [1,2,3,4,5,6,7,8]
chromosome2 = [10,11,12,13,14,15,16,17,18,19]
individual = [chromosome1, chromosome2]
passenger_list = [1,2,3,4,5,6,7,8,9]
prob_mutation = 0.5

print(RecombIntegers(individual,prob_mutation,passenger_list))

# SELECTION 

# To compare different strategies, we will implement a roulette wheel selection and a tournament selection 
# Then, we'll get to decide which one is the most suitable for our DARP problem 

# 1 - ROULETTE WHEEL SELECTION 

def roulette_wheel_selection(population, fitness, graph): 
    """
    @param population, a pool of individuals/solutions 
    @param fitness, the numeric function that evaluates the "goodness" of a solution 
    return a list of selected solutions/individuals, according to the roulette wheel selection
    """

    selected_ind = []
    #sum_of_fitnesses = 0 
    fitnesses = []
    for ind in population: 
        fitnesses.append(fitness(ind, graph))
        print("fitnesses list", fitnesses)

    sum_probabilities = 0
    probabilities = {}

    sum_of_fitnesses = sum(fitnesses)
    print("sum of fitnesses", sum_of_fitnesses)

    cpt = 0 
    for ind in population: 
        
        probabilities[ind] = sum_probabilities + fitnesses[cpt]/sum_of_fitnesses
        sum_probabilities += probabilities[ind]
        cpt+=1
    
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda x:x[1]))
    random_number = random.random()
    
    for ind in population: 
        if random_number > sorted_probabilities[ind] : 
            selected_ind.append(ind)
    return selected_ind

    
    
# 2 - TOURNAMENT SELECTION 


