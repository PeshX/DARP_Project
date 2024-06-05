# IMPORTS 
from genAlgo import *
import random as rd

"""
 TO GENERATE THE INITIAL POPULATION (POOL OF SOLUTIONS)
 Strategy : first, we'll generate individuals at random 
            then, we check if each individuals is a valid solution 
            if not, we make small changes until it is valid 
    All we want is a good enough population to start with, the genetic algorithm will do the rest improving them 
"""

def generate_individual_at_random(nb_transfers, nb_passengers) : 
    # generates an individual, id est a solution for the problem, a list of lists (transfers)
    """
    @param nb_transfers := nb of vehicles 
    @param nb_passengers := nb of customers to drive 
    
    """
    ind = generate_random_lists(nb_passengers, nb_transfers)

    return ind

def is_valid(): 
    # checks if a solution is valid with respect to the constraints 
    return 0

def generate_initial_pop(): 
    # generates a pool of feasible solutions, the initial population to feed the algorithm with 
    return 0


"""
    ANNEX FUNCTIONS
"""

import random

def generate_random_lists(n, s):
    # Create a list of numbers from 1 to N
    numbers = list(range(1, n + 1))
    # Shuffle the list to randomize the order
    random.shuffle(numbers)
    
    # Initialize an empty list to hold the sublists
    main_list = []
    
    # Ensure we have exactly s sublists
    for _ in range(s - 1):
        # Randomly determine the size of the current sublist
        size = random.randint(1, n - len(main_list))
        # Extract the sublist from the shuffled numbers
        sublist = numbers[:size]
        # Append the sublist to the main list
        main_list.append(sublist)
        # Remove the used numbers from the list
        numbers = numbers[size:]
    
    # Append the remaining numbers as the last sublist
    main_list.append(numbers)
    
    return main_list

# TEST
nb_transfers = 2 # The range from 1 to N
nb_passengers = 5   # Number of sublists
test = generate_individual_at_random(nb_transfers, nb_passengers) 
print(test)
