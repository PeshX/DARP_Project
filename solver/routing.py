# Imports 
import networkx as nx

def TransferNodesSequence(transfer_LUT, passenger_LUT, transfer_id, passengers_in_transfer):
    Nodes_List = []
    # Append the transfer node to the list
    Nodes_List.append(transfer_LUT[transfer_id])
    
    # Filter out zero values from passengers_in_transfer
    passengers = [value for value in passengers_in_transfer if value != 0]
    print(passengers)
    
    # Append the nodes for each passenger in the transfer
    for i in passengers:
        Nodes_List.append(passenger_LUT[i][0:2])  # Takes only the nodes for each passenger in the transfer
    
    return Nodes_List


# Define the combined weight function
def CombinedWeight(u, v, d, w_f=0.5, w_t=0.5):
    return w_f * d['fuel_cost'] + w_t * d['time_cost']

def RoutingAlgorithm(chromosome, graph, transfer_LUT, passenger_LUT, w_f, w_t):
     
    #  source = Nodes[1]
    #  target = Nodes[4]
    #  path = nx.dijkstra_path(graph, source=source, target=target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))
    #  path_length = nx.dijkstra_path_length(graph, source=source, target=target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)) 
        
     return
