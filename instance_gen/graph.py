# Imports 
import networkx as nx
import random

def createGraphInstance(min_nodes, max_nodes, min_weight_fuel, max_weight_fuel, min_weight_time, max_weight_time, min_degree):

    # Generate a random number of nodes
    num_nodes = random.randint(min_nodes, max_nodes)
    print(f"The current number of nodes is {num_nodes}")

    # Create a directed graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(2))
    G.add_edge(0, 1, fuel_cost=3, time_cost=45)
    G.add_edge(0, 2, fuel_cost=2, time_cost=30)
    G.add_edge(1, 2, fuel_cost=4, time_cost=54)

    for i in range(3,num_nodes):
        G.add_node(i)
        fuel_cost = random.randint(min_weight_fuel, max_weight_fuel)
        time_cost = random.randint(min_weight_time, max_weight_time)
        G.add_edge(i, i-1, fuel_cost=fuel_cost, time_cost=time_cost)
        G.add_edge(i, i-2, fuel_cost=fuel_cost, time_cost=time_cost)

    return G
