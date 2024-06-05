# IMPORTS 
#from instance_gen import *

# ALGORITHM PARAMETERS 
N = 50 #nb of individuals in the initial population
nb_generations = 100
proba_mutation = 0.2 #proba for an ind to be muted 
proba_mutation_gene = 0.3 #proba for a gene to be muted 
proba_crossing = 0.4

# FITNESS FUNCTION 
def fitness(individual): #, graph) : 
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

# TEST 
T1 = [1, 3, 5]
T2 = [4, 2]
individual = [T1, T2]
res = fitness(individual)
print(res)


