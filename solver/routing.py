# Imports 
import networkx as nx

def TransferNodesSequence(transfer_LUT, passenger_LUT, transfer_id, passengers_in_transfer):

    """
    Find the start/stop of transfer and of all passengers within the chromosome
    
    @param transfer_LUT: dictionary of the transfers
    @param passenger_LUT: dictionary of the passengers
    @param transfer_id: the number of the transfer we are analyzing
    @param passengers_in_transfer: list of the passengers on-board of the transfer
    
    @return: a list of tuples, the first one regards the transfer while the other are for the passengers
    """

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

    """
    Linearly combines the two attributes of an edge (to be used inside a lambda function)
    
    @param u: starting node
    @param v: stopping node
    @param d: edge between 'u' and 'v'
    @param w_f: weight for the fuel attribute of the graph's edges
    @param w_t: weight for the time attribute of the graph's edges
    
    @return: a list of tuples, the first one regards the transfer while the other are for the passengers
    """

    return w_f * d['fuel_cost'] + w_t * d['time_cost']

def RoutingAlgorithm(chromosome, graph, n_transfer, transfer_LUT, passenger_LUT, w_f, w_t):
     
     """
     Compute the overall route of a transfer in the pick-up/drop of all its passengers
    
     @param chromosome: a list of the passenger within a transfer
     @param graph: the instance of the graph built with NetworkX
     @param n_transfer: the number of the transfer we are analyzing 
     @param transfer_LUT: dictionary of the transfers
     @param passenger_LUT: dictionary of the passengers
     @param w_f: weight for the fuel attribute of the graph's edges
     @param w_t: weight for the time attribute of the graph's edges
    
     @return: list of the path of a transfer
     """
     
     nodes_list = TransferNodesSequence(transfer_LUT, passenger_LUT, n_transfer, chromosome)

     starting_node = nodes_list[0][0]
     stopping_node = nodes_list[0][1]
     
     transfer_path = []

     I_cn, I_p = 1,1

     transfer_path.extend(nx.dijkstra_path(graph, source = starting_node, target= nodes_list[I_cn][0], weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
     transfer_path.pop()

     current_node = nodes_list[I_cn][0]
     current_dest = nodes_list[I_p][1]
     if I_cn+1 < len(nodes_list):
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
                I_p += 1
                current_node = next_p
                current_dest = nodes_list[I_p][1]
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