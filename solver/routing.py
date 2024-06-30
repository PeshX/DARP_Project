# Imports 
import networkx as nx

def TransferNodesSequence(transfer_LUT, passenger_LUT, transfer_id, passengers_in_transfer):
    Nodes_List = []
    # Append the transfer node to the list
    Nodes_List.append(transfer_LUT[transfer_id][0:2])
    
    # Filter out zero values from passengers_in_transfer
    passengers = [value for value in passengers_in_transfer if value != 0]
    
    # Append the nodes for each passenger in the transfer
    for i in passengers:
        Nodes_List.append(passenger_LUT[i][0:2])  # Takes only the nodes for each passenger in the transfer
    
    return Nodes_List


# Define the combined weight function
def CombinedWeight(u, v, d, w_f=0.5, w_t=0.5):
    return w_f * d['fuel_cost'] + w_t * d['time_cost']

def RoutingAlgorithm(chromosome, graph, n_transfer, transfer_LUT, passenger_LUT, w_f, w_t):
     
     nodes_list = TransferNodesSequence(transfer_LUT, passenger_LUT, n_transfer, chromosome)

     starting_node = nodes_list[0][0]
     stopping_node = nodes_list[0][1]
     
     transfer_path = []

     I_cn, I_p = 1,1

     transfer_path.extend(nx.dijkstra_path(graph, source = starting_node, target= nodes_list[I_cn][0], weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
     transfer_path.pop()

     current_node = nodes_list[I_cn][0]
     current_dest = nodes_list[I_p][1]
     next_p = nodes_list[I_cn+1][0]
   
     # Using a lambda function within a for loop
     while current_node != nodes_list[-1][1]:
            
            if (current_dest != -1) and ((I_cn + 1) < len(nodes_list)):

                path_D = nx.dijkstra_path_length(graph, source = current_node, target= current_dest, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))
                path_NextP = nx.dijkstra_path_length(graph, source = current_node, target= next_p, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))

                if path_D < path_NextP:
                    
                    transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= current_dest, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
                    #transfer_path.pop()
                    transfer_path.append('EOP') # one passenger has been dropped
                    current_node = current_dest
                    if I_p<I_cn:
                        I_p += 1
                        current_dest = nodes_list[I_p][1]
                    else:
                        current_dest = -1
                        
                else:
                                                       
                    transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= next_p, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
                    transfer_path.pop()
                    current_node = next_p
                    I_cn += 1
                    if I_cn+1 < len(nodes_list):
                        next_p = nodes_list[I_cn+1][0]
                        
            elif (I_cn + 1) < len(nodes_list):
                
                transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= next_p, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
                transfer_path.pop()
                current_node = next_p
                current_dest = nodes_list[I_cn+1][1]
                I_cn += 1
                if I_cn+1 < len(nodes_list):
                    next_p = nodes_list[I_cn+1][0]

            else:
                
                transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= current_dest, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
                #transfer_path.pop()
                transfer_path.append('EOP') # one passenger has been dropped
                current_node = current_dest
                if I_p < I_cn:
                    I_p += 1
                    current_dest = nodes_list[I_p][1]

    
     transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= stopping_node, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))

        
     return transfer_path