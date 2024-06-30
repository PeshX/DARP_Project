# IMPORTS 
#from genAlgo import *
import random 

"""
 GENERATION OF THE INITIAL POPULATION (POOL OF SOLUTIONS)
 
"""

def generate_individual_at_random(nb_passengers, capacities) : 
    # generates an individual, id est a solution for the problem, a list of lists (transfers)
    """
    @param nb_transfers := nb of vehicles 
    @param nb_passengers := nb of customers to drive 
    
    """
    #ind = generate_random_lists(nb_passengers, capacities)
    ind = distribute_passengers(nb_passengers, capacities)

    return ind


def distribute_passengers(num_passengers, capacities):
    
    passengers = list(range(1, num_passengers + 1))
    random.shuffle(passengers)
    
    result = [[] for _ in capacities]

    for i, capacity in enumerate(capacities):
        while len(result[i]) < capacity and passengers:
            result[i].append(passengers.pop())
    
    while passengers:
        for sublist in result:
            if passengers:
                sublist.append(passengers.pop())
    
    # Filling with zeros if some seats are free 
    for sublist, capacity in zip(result, capacities):
        while len(sublist) < capacity:
            sublist.append(0)
    
    return result


def generate_initial_pop(nb_individuals, nb_passengers, capacities): 
    # generates a pool of feasible solutions, the initial population to feed the algorithm with 
    """
        @param nb_individuals := the number of solutions in the initial population
    """
    init_pop = []
    for i in range(nb_individuals): 
        ind = generate_individual_at_random(nb_passengers, capacities)
        init_pop.append(ind)

    return init_pop


"""
 FOR PRETTY OUTPUTS 
"""
def pretty_solution(solution): 
    # to display a readable solution 
    L = ""
    transfer_id = 1
    for transfer in solution : 
        subL = ""
        if transfer == [] : 
            subL += "Vehicle " + str(transfer_id) + " doesn't drive any passenger."
        else: 
            subL += "Vehicle " + str(transfer_id) + " fetches: "
            for passenger_id in transfer : 
                subL +=  "passenger " + str(passenger_id) + ", then "
            subL = subL[:-7]
        L+=subL + "\n"
        transfer_id += 1

    return L

def pretty_population(pop): 
    # to display a readable population 
    L = ""
    sol_id = 1
    for sol in pop : 
        subL = ""
        
        subL += "SOLUTION " + str(sol_id) + "\n"
        subL += pretty_solution(sol)
        L+=subL + "\n"
        sol_id += 1

    return L

# TEST
nb_passengers = 1 # total nb of customers 
#capacities = [3, 2, 5]  # capacities of all vehicles 
capacities = [3, 2]


test_one_sol = generate_individual_at_random(nb_passengers, capacities) 
#print(test_one_sol)
print(pretty_solution(test_one_sol))

test_one_pop = generate_initial_pop(5, nb_passengers, capacities)
print(pretty_population(test_one_pop))
print(test_one_pop)