# Imports 
import networkx as nx

def TransferNodesSequence(transfer_LUT, passenger_LUT, transfer_id, passengers_in_transfer):
    Nodes_List = []
    # Append the transfer node to the list
    Nodes_List.append(transfer_LUT[transfer_id])
    
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
                    transfer_path.pop()
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
                transfer_path.pop()
                current_node = current_dest
                if I_p < I_cn:
                    I_p += 1
                    current_dest = nodes_list[I_p][1]

    
     transfer_path.extend(nx.dijkstra_path(graph, source = current_node, target= stopping_node, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))

        
     return transfer_path

# def RoutingAlgorithm(chromosome, graph, n_transfer, transfer_LUT, passenger_LUT, w_f, w_t):

#     stop_condition = 0
#     case = 'start'
#     nodes_list = TransferNodesSequence(transfer_LUT, passenger_LUT, n_transfer, chromosome)
#     transfer_path = []         
#     ip_path = (0,0) # tuple on the left is i-th drop point, on the right we have (i+1)-th pick-up point
#     pending_drop = 1 # used to retrieve the drop in case of all pick-ups, to be upd in case of mixed scenario
#     #analysed_passenger
     
#     while stop_condition==0:    

#         run_case(case, graph, nodes_list, transfer_path, ip_path, pending_drop, w_f, w_t)          
        
#     return

# def run_case(case, graph, nodes_list, transfer_path, ip_path, pending_drop, w_f, w_t):     
                
#     if case == 'start':
#         case, analysed_passenger, transfer_path, ip_path[0] = state_1(graph, nodes_list, transfer_path, w_f, w_t)
#     elif case == 'pick&drop':
#         analysed_passenger, pending_drop, transfer_path, ip_path = state_2(graph, nodes_list, analysed_passenger, pending_drop, transfer_path, ip_path, w_f, w_t)
#     elif case == 'stop':
#         case_3()
#     else:
#         default()


# def state_1(graph, nodes_list, transfer_path, w_f, w_t):

#     # Pick starting and ending nodes
#     starting_node = nodes_list[0][0]
#     #stopping_node = nodes_list[0][1]
#     # Pick first passengers nodes
#     get_positions = lambda route: (route[0], route[1])
#     source, target = get_positions(nodes_list[1])
#     # Go to first passenger and add to path
#     transfer_path.extend(nx.dijkstra_path(graph, source = starting_node, target = source, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
#     # Compute next hypothetical path to drop P1
#     ip_path = nx.dijkstra_path_length(graph, source = source, target = target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))
#     # P1 is analysed
#     analysed_passenger = 1
#     # Change state for next iteration
#     case = 'pick&drop'

#     return case, analysed_passenger, transfer_path, ip_path

# def state_2(graph, nodes_list, analysed_passenger, pending_drop, transfer_path, ip_path, w_f, w_t):
#     # Pick i-th passengers nodes wrt the previous analysed
#     get_positions = lambda route: (route[0], route[1])
#     source, target = get_positions(nodes_list[analysed_passenger])
#     source_next, target_next = get_positions(nodes_list[analysed_passenger+1])
#     # Compute path to P-ith pick-up and compare to ip_path
#     ip_path[1] = nx.dijkstra_path_length(graph, source = source, target = source_next, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t))
#     best_path = ip_path.index(min(ip_path))
#     if best_path == 0:
#         # Drop and go to next passenger's source -> item on the right becomes target of the i-th
#         transfer_path.extend(nx.dijkstra_path(graph, source = source, target = target, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
#         # Go to next passenger to be dropped
#         pending_drop += 1
        
#     else:
#         # Pick-up another passenger and cycle, item on the left remains the same and item on the right varies 
#         transfer_path.extend(nx.dijkstra_path(graph, source = source, target = source_next, weight=lambda u, v, d: CombinedWeight(u, v, d, w_f, w_t)))
#         # Increase passenger for next iteration
#         analysed_passenger += 1        

#     # TODO: last check on the number of passengers or on the ip_path ot go to 'stop' state

#     return analysed_passenger, pending_drop, transfer_path, ip_path
