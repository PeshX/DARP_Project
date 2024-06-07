# Imports 
import networkx as nx
import random


def createGraphInstance(min_nodes, max_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, connect_prob):

    # Generate a random number of nodes
    num_nodes = random.randint(min_nodes, max_nodes)
    # ...
    connect_prob = 1 - connect_prob

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Add a random number of edges with random weights
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if random.random() > connect_prob:  # Randomly decide whether to add an edge
                    fuel_cost = random.randint(min_weight_fuel, max_weight_fuel)
                    time_cost = random.randint(min_weight_time, max_weight_time)
                    G.add_edge(i, j, fuel_cost=fuel_cost, time_cost=time_cost)
                    G.add_edge(j, i, fuel_cost=fuel_cost, time_cost=time_cost)
                else:
                    fuel_cost = random.randint(min_weight_fuel, max_weight_fuel)
                    time_cost = random.randint(min_weight_time, max_weight_time)
                    G.add_edge(j, i, fuel_cost=fuel_cost, time_cost=time_cost)

    return G