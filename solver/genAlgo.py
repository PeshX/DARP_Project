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

def LineRecombIntegers(individuals, p, passengers_list):

    alpha = random.uniform(-p, 1+p)
    beta = random.uniform(-p, 1+p)

    len_short_chr = len(min(individuals,key=len))

    sx_half_long_chr = int(np.floor(len(min(individuals,key=len))/2))
    low_index = random.randint(0, sx_half_long_chr-1)
    print(low_index)

    while (low_index+len_short_chr < len(max(individuals,key=len))-1):
                    
           low_index = random.randint(0, sx_half_long_chr)

    if len(individuals[0]) > len(individuals[1]):
        long_index = 0
        short_index = 1

    else:
        long_index = 1
        short_index = 0

    for i in range(0,len(min(individuals,key=len))):
        print(i)

        t = alpha * individuals[long_index][i+low_index] +  beta * individuals[short_index][i]
        s = alpha * individuals[short_index][i] +  beta * individuals[long_index][i+low_index]

        while((int(np.floor(t+0.5)) in passengers_list) and (int(np.floor(s+0.5)) in passengers_list)):
            individuals[long_index][i] = int(np.floor(t+0.5))
            individuals[short_index][i] = int(np.floor(s+0.5))
            t = alpha * individuals[long_index][i+low_index] +  beta * individuals[short_index][i]
            s = alpha * individuals[short_index][i] +  beta * individuals[long_index][i+low_index]
            

    return individuals

chromosome1 = [1,2,3,4]
chromosome2 = [5,6,7,8,9]
individual = [chromosome1, chromosome2]
passenger_list = [1,2,3,4,5,6,7,8,9]
prob_mutation = 0.5

#mutated_ind = LineRecombIntegers(individual,prob_mutation,passenger_list)
#print(mutated_ind)

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


