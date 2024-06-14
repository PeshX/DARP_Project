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

            weights = [0] # to modify : fetch the weights from the graph for the edge (passenger1, passenger2)

        for w in weights : 
            consumption += w # to modify : up the consumption accordingly to the weight 
            delay += w # to modify : up the delay accordingly to the weight 
        
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


chromosome1 = [1,2,3,4]
passenger_list = [1,2,3,4,5,6,7,8,9]
prob_mutation = 0.4

mutated_ch = IntegerRandMutation(chromosome1,prob_mutation,passenger_list)
print(mutated_ch)

# RECOMBINATION

def LineRecombIntegers(individuals, p, passengers_list):

    alpha = random.uniform(-p, 1+p)
    beta = random.uniform(-p, 1+p)

    len_short_chr = len(min(individuals,key=len))

    sx_half_long_chr = np.floor(len(min(individuals,key=len))/2)
    low_index = random.randint(0, sx_half_long_chr)

    while (low_index+len_short_chr < len(max(individuals,key=len)-1)):
                    
           low_index = random.randint(0, sx_half_long_chr)

    if len(individuals[1]) > len(individuals[2]):
        long_index = 1
        short_index = 2

    else:
        long_index = 2
        short_index = 1


    for i in range(0,len(min(individuals,key=len))-1):


        t = alpha * individuals[long_index][i+low_index] +  beta * individuals[short_index][i]
        s = alpha * individuals[short_index][i] +  beta * individuals[long_index][i+low_index]

        while((np.floor(t+0.5) in passengers_list) and (np.floor(s+0.5) in passengers_list)):
            individuals[1][i] = np.floor(t+0.5)
            individuals[2][i] = np.floor(s+0.5)

    return individuals

