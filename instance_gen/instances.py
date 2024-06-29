# Imports 
import networkx as nx
import random

def createPassengersTransfersBatch(nb_passengers, nb_transfer ):

    # Create empty dictionary for transfers and passengers
    looUpTableTransfers = {}
    looUpTablePassengers = {}

    # Compute worst case scenario for number of nodes
    worst_case_nodes = nb_passengers*2 + nb_transfer*2

    # Create start/stop positions for transfers (need to be unique and not usable by passengers)
    transfer_points = random.sample(range(worst_case_nodes), nb_transfer*2)

    # Assign positions to each transfer
    for i in range(nb_transfer):
        key = i+1
        start_position = transfer_points.pop(0)
        stop_position = transfer_points.pop(0)  
        looUpTableTransfers[key] = (start_position, stop_position)

    # Fill each passenger with the characterizing parameters

    for i in range(nb_passengers):
        key = i+1
        start_position = random.randint(0, 100)  
        stop_position = random.randint(start_position, 200)  
        time_request = random.uniform(0.1, 10.0)  
        passengers[key] = (start_position, stop_position, time_request)

    return passengers

def createGraphInstance(num_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree):

    print(f"The current number of nodes is {num_nodes}")

    # Create a graph object
    G = nx.Graph()

    # Add starting set: 3 nodes connected by 3 edges
    G.add_nodes_from(range(2))
    G.add_edge(0, 1, fuel_cost=3, time_cost=45)
    G.add_edge(0, 2, fuel_cost=2, time_cost=30)
    G.add_edge(1, 2, fuel_cost=4, time_cost=54)

    # Add remaining nodes with following construction
    for i in range(3,num_nodes):
        G.add_node(i)
        # Add a number of edges starting from new node equal to min_degree
        for edge in range(min_degree):
            fuel_cost = random.randint(min_weight_fuel, max_weight_fuel)
            time_cost = random.randint(min_weight_time, max_weight_time)
            # Connect the edge to the two previous nodes (CAN BE CHANGED)
            G.add_edge(i, i-(min_degree-edge), fuel_cost=fuel_cost, time_cost=time_cost)
    

    return G
