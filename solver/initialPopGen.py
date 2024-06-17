# IMPORTS 
#from genAlgo import *
import random 

"""
 TO GENERATE THE INITIAL POPULATION (POOL OF SOLUTIONS)
 Strategy : first, we'll generate individuals at random 
            then, we check if each individuals is a valid solution 
            if not, we make small changes until it is valid 
    All we want is a good enough population to start with, the genetic algorithm will do the rest improving them 
"""

def generate_individual_at_random(nb_passengers, capacities) : 
    # generates an individual, id est a solution for the problem, a list of lists (transfers)
    """
    @param nb_transfers := nb of vehicles 
    @param nb_passengers := nb of customers to drive 
    
    """
    ind = generate_random_lists(nb_passengers, capacities)

    return ind

def is_valid(solution): 
    # checks if a solution is valid with respect to the constraints 

    # C1 : capacity of each vehicle respected 

    return True

def generate_initial_pop(nb_individuals, nb_passengers, capacities): 
    # generates a pool of feasible solutions, the initial population to feed the algorithm with 
    """
        @param nb_individuals := the number of solutions in the initial population
    """
    init_pop = []
    for i in range(nb_individuals): 
        ind = generate_individual_at_random(nb_passengers, capacities)

        while (not is_valid(ind)) : 
            # FIX IT 
            i = 0
        init_pop.append(ind)

    return init_pop


"""
    ANNEX FUNCTIONS
"""

def generate_random_lists(n, sizes):
    # Check if the sum of sizes equals n
    """if sum(sizes) != n:
        raise ValueError("The sum of the sizes must equal n.")"""
    
    # Create a list of numbers from 1 to N
    numbers = list(range(1, n + 1))
    # Shuffle the list to randomize the order
    random.shuffle(numbers)
    
    # Initialize an empty list to hold the sublists
    main_list = []
    
    # Generate the sublists based on the sizes provided
    for size in sizes:
        # Extract the sublist from the shuffled numbers
        sublist = numbers[:size]
        # Append the sublist to the main list
        main_list.append(sublist)
        # Remove the used numbers from the list
        numbers = numbers[size:]
    
    return main_list


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
nb_passengers = 10 # total nb of customers 
capacities = [3, 2, 5]  # capacities of all vehicles 

test_one_sol = generate_individual_at_random(nb_passengers, capacities) 
#print(test_one_sol)
print(pretty_solution(test_one_sol))

test_one_pop = generate_initial_pop(5, nb_passengers, capacities)
print(pretty_population(test_one_pop))
