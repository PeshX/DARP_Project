# Imports 
import networkx as nx
import random

# Create a directed graph
G = nx.DiGraph()

# # Add edges with fuel cost and time cost
# G.add_edge('A', 'B', fuel_cost=5, time_cost=6)
# G.add_edge('A', 'D', fuel_cost=10, time_cost=17)
# G.add_edge('B', 'C', fuel_cost=3, time_cost=2)
# G.add_edge('C', 'A', fuel_cost=2, time_cost=2)
# G.add_edge('C', 'D', fuel_cost=1, time_cost=5)

# Define a function that retrieves the list of nodes which are involved per each transfer

def TransferNodesSequence(transfer_LUT, passenger_LUT, transfer_id, passengers_in_transfer):
         
      Nodes_List = []
      Nodes_List.append(transfer_LUT[transfer_id])
      Nodes_List.append(passenger_LUT[i][0:2] for i in passengers_in_transfer) # passenger_LUT[i][0:2] Takes only the nodes for each passenger in the transfer

      return Nodes_List


# Define a list of edges with their fuel cost and time cost
n_nodes = 10 # To be substitute with the length of the list of nodes generated before
max_fuel_cost = 7
max_time_cost = 70
edges = []

for i in range(n_nodes):
    edges.append((random.randint(0,n_nodes-1), random.randint(0,n_nodes-1), random.randint(1,max_fuel_cost), random.randint(1,max_time_cost)))  
      
print(edges)

# Add edges iteratively
for edge in edges:
    u, v, fuel_cost, time_cost = edge
    G.add_edge(u, v, fuel_cost=fuel_cost, time_cost=time_cost)

# Print edges with attributes to verify
for u, v, attrs in G.edges(data=True):
    print(f"Edge ({u}, {v}): {attrs}")



# Define the combined weight function
def combined_weight(u, v, d, w_f=0.5, w_t=0.5):
    return w_f * d['fuel_cost'] + w_t * d['time_cost']

# Set weights for fuel cost and time cost
w_f = 0.4  # Weight for fuel cost
w_t = 0.6  # Weight for time cost

# Find the shortest path using the combined weight function
source = 'A'
target = 'D'
path = nx.dijkstra_path(G, source=source, target=target, weight=lambda u, v, d: combined_weight(u, v, d, w_f, w_t))
path_length = nx.dijkstra_path_length(G, source=source, target=target, weight=lambda u, v, d: combined_weight(u, v, d, w_f, w_t))

# print(f"Shortest path: {path}")
# print(f"Path length: {path_length}")

# transfer_points = random.sample(range(34), 4)
# print(transfer_points)
# print(transfer_points.pop(0))
# print(transfer_points)

