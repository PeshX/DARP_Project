# Imports 
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges with fuel cost and time cost
G.add_edge('A', 'B', fuel_cost=5, time_cost=6)
G.add_edge('A', 'D', fuel_cost=10, time_cost=17)
G.add_edge('B', 'C', fuel_cost=3, time_cost=2)
G.add_edge('C', 'A', fuel_cost=2, time_cost=2)
G.add_edge('C', 'D', fuel_cost=1, time_cost=5)


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

print(f"Shortest path: {path}")
print(f"Path length: {path_length}")