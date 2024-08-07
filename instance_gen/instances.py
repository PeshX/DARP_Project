# Imports 
import networkx as nx
import random

def createPassengersTransfersBatch(nb_passengers, nb_transfer):

    """
    Create the two dictionaries characterizing both passengers and transfers and one node list for the graph
    
    @param nb_passengers: total number of passengers to be moved
    @param nb_transfer: total number of available transfers
    
    @return: two dictionaries with key-value pairs represented with integers and one list of integers
    """

    # Create empty dictionaries for transfers and passengers
    looUpTablePassengers = {}    
    looUpTableTransfers = {}

    # Compute worst case scenario for number of nodes | Maybe we should think if we want to pass the list of nodes from outside?
    worst_case_nodes = nb_passengers*2 + nb_transfer*2

    # Create start/stop positions for transfers (need to be unique and not usable by passengers)
    transfer_points = random.sample(range(worst_case_nodes), nb_transfer*2)

    # List of actual nodes for the graph
    Nodes_List = []
    Nodes_List.extend(transfer_points)

    # Fill each passenger with the characterizing parameters
    for i in range(nb_passengers):
        key = i+1
        start_position, stop_position = random.sample([x for x in range(worst_case_nodes) if x not in transfer_points], 2)

        # Time_request in minutes
        time_request = random.randint(30,90) # TODO: this has to go according to the time weights on the graph 
        looUpTablePassengers[key] = (start_position, stop_position, time_request)
        # Append new nodes
        Nodes_List.extend([start_position, stop_position])

    # Assign positions to each transfer
    for i in range(nb_transfer):
        key = i+1
        start_position = transfer_points.pop(0)
        stop_position = transfer_points.pop(0) 
        capacity = random.randint(5,10) # TODO: this can be changed according to the number of passengers or viceversa
        looUpTableTransfers[key] = (start_position, stop_position, capacity)
        
    # Cut out duplicates
    Nodes = list(set(Nodes_List))
    
    # Check for feasibility of input parameters and created instance
    sum_capacities = 0
    try:
        # Define your condition here
     for i in range(nb_transfer):
        sum_capacities += looUpTableTransfers[i+1][2] 

     if nb_passengers > sum_capacities:
         my_condition = False 
     else:
         my_condition = True      

     # Check the condition
     check_condition(my_condition)
    except ValueError as e:
        print(e)
        exit(1)

    return looUpTableTransfers, looUpTablePassengers, Nodes

def check_condition(condition):

    """
    Raise the exception if condition is not met
    
    @param condition: boolean value according to previous check
    
    """

    if not condition:
        raise ValueError("Instance of passengers and transfers NOT feasible!")


def createGraphInstance(nodes_list, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree):

    """
    Computes a graph object with Networkx library
    
    @param nodes_list: a list of the nodes to be used for the graph
    @param min_weight_fuel: minimum fuel weight to be assigned to a graph's edge
    @param max_weight_fuel: maximum fuel weight to be assigned to a graph's edge
    @param min_weight_time: minimum time weight to be assigned to a graph's edge
    @param max_weight_time: maximum time weight to be assigned to a graph's edge
    @param min_degree: minimum number of edges departing from a node
    
    @return: a NetworkX instance of a undirected weighted graph
    """

    # Informative print
    print(f"The current number of nodes is {len(nodes_list)}")

    # Create a graph object
    G = nx.Graph()
    
    # Add starting set: 3 nodes connected by 3 edges
    node1, node2, node3 = random.sample(nodes_list,3)
    G.add_node(node1)
    G.add_node(node2)
    G.add_node(node3)
    G.add_edge(node1, node2, fuel_cost=3, time_cost=45)
    G.add_edge(node1, node3, fuel_cost=2, time_cost=30)
    G.add_edge(node2, node3, fuel_cost=4, time_cost=54)

    # Add remaining nodes with following construction
    for i in range(len(nodes_list)):
        G.add_node(nodes_list[i])
        # Add a number of edges starting from new node equal to min_degree
        for edge in range(min_degree):
            fuel_cost = random.randint(min_weight_fuel, max_weight_fuel)
            time_cost = random.randint(min_weight_time, max_weight_time)
            # Connect the edge to the two previous nodes in the list 
            G.add_edge(nodes_list[i], nodes_list[i-(min_degree-edge)], fuel_cost=fuel_cost, time_cost=time_cost)  

    return G
