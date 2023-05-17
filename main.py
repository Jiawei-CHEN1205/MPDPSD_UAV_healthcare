"""PDP with soft deadlines"""

import numpy as np
# import tensorflow as tf
# from tensorflow.python import keras
# from tensorflow.keras import layers
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
from torch.utils.data import Dataset

# parameters setting:
vehicle_num_uav = 5
vehicle_num_human = 10
vehicle_num = vehicle_num_uav + vehicle_num_human

w_uav = 1  # unitcost_uav per distance
w_human = 10  # unitcost_human per distance
H_d = 100  # penalty cost for exceeding the ddl in set D, otherwise 0

capacity_uav = 30
capacity_human = 100
speed_uav = 0.2
speed_human = 0.1

size_order = 20
demand_p = random.sample(range(20, 41), size_order)  # demand range 20,40
demand_d = [-x for x in demand_p]  # negative demand in D set

# np.random.randint(20, 41)
deadline_d = random.sample(range(40, 101), size_order)  # deadlines in D set, otherwise infinity
# print(deadline_d)  # 只有末尾时间约束l_i 因此改成时间窗约束的时候 每个e_i设成0
# 时间窗设置 标准模型里面 是每个节点（包括p, d, depots）都有的：
# data['time_matrix']: An array of travel times between locations.
# data['time_windows']: An array of time windows for the locations

time_window1 = np.zeros((size_order*2 + vehicle_num, 2), dtype=int) # sequence: p_set, d_set, depots
for i in range(size_order):
    time_window1[i][0] = 0
    time_window1[i][1] = 99999
for i in range(size_order, size_order*2):
    time_window1[i][0] = 0
    time_window1[i][1] = np.random.randint(40, 101)  # deadline_d
for i in range(size_order*2, (size_order*2 + vehicle_num)):
    time_window1[i][0] = 0
    time_window1[i][1] = 99999

time_windows = []
for row in time_window1:
    time_windows.append((int(row[0]), int(row[1])))

# print(time_windows)
print(len(time_windows)) # 55 = 20+20+15
# print(time_windows.shape)

# distance calculate func:
def dist_pd(loc_p, loc_d):
    diff = loc_p - loc_d
    square_dist = torch.sum(diff ** 2, dim=-1)
    dist = torch.sqrt(square_dist)
    return dist


loc_p_coor = torch.FloatTensor(size_order, 2).uniform_(0, 1)  # P locations coordinates
loc_d_coor = torch.FloatTensor(size_order, 2).uniform_(0, 1)  # D locations coordinates

dist_pds = dist_pd(loc_p_coor, loc_d_coor)  # dist from p to d one-by-one
depots_coor = loc_p_coor[torch.randint(0, size_order, (vehicle_num,))]  # depots_coordinate sample from locations_p

# change into array:
loc_p_array = loc_p_coor.numpy()
loc_d_array = loc_d_coor.numpy()
loc_depots = depots_coor.numpy()

# all locations, note the index sequence: Pickup - Delivery - Depots
loc_all = np.concatenate((loc_p_array, loc_d_array, loc_depots), axis=0)

# 计算点之间的距离matrix pairwise:
distance_matrix = np.zeros((len(loc_all), len(loc_all)), dtype=int)
for i in range(len(loc_all)):
    for j in range(len(loc_all)):
        distance_matrix[i][j] = np.sqrt(np.sum((np.array(loc_all[i]) - np.array(loc_all[j])) ** 2))


# For each pair, the first entry is index of the pickup location, and the second is the index of the delivery location.
pdp_index = np.zeros((size_order, 2), dtype=int)
for i in range(size_order):
    pdp_index[i][0] = int(i)
    pdp_index[i][1] = int(i + size_order)
# print(pdp_index)  # (size_order, 2) size


# 1 # data_model.py
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    # index pairs p-d:
    data['pickups_deliveries'] = pdp_index
    data['num_vehicles'] = vehicle_num
    # Travel time matrix：
    vehicle_speed = 0.2 # 怎么改成不同的speed还没研究
    time_matrix = {}
    for from_node in range(len(data['distance_matrix'])):
        time_matrix[from_node] = {}
        for to_node in range(len(data['distance_matrix'])):
            time_matrix[from_node][to_node] = int(data['distance_matrix'][from_node][to_node] / vehicle_speed)
    data['time_matrix'] = time_matrix
    data['time_windows'] = time_windows

    # data['depot'] = 0 # index of depot
    # vehicle_num = vehicle_num_uav + vehicle_num_human
    data['num_vehicles'] = vehicle_num

    # Setting vehicle capacities
    data['vehicle_capacities'] = [None] * (data['num_vehicles'])
    for vehicle_id in range(vehicle_num_uav):
        data['vehicle_capacities'][vehicle_id] = capacity_uav  # 30
    for vehicle_id in range(vehicle_num_uav, vehicle_num):
        data['vehicle_capacities'][vehicle_id] = capacity_human  # 100
    # Denoting Depot node by index 0
    data['depot'] = 40

    # Setting the demands for each pickup and drop node as +1 and -1 respectively
    # 这里的demand，长度只有订单数*2 加一，说明也就是所有的node的个数 我们的问题里面 depot不止有一个 所以应该把demand的维度填满补零
    # data['demands'] = [None] * int(2 * len(data['pickups_deliveries']) + 1)
    data['demands'] = [None] * int(len(time_windows))  # 55个点 总长度
    # data['demands'][40] = 0
    for node in (data['pickups_deliveries']):
        data['demands'][node[0]] = np.random.randint(20, 41)
        data['demands'][node[1]] = -1 * data['demands'][node[0]]
    for depot in range(len(data['pickups_deliveries']) * 2, len(time_windows)):
        data['demands'][depot] = int(0)

    # print("demand set:", data['demands'])  # 正负匹配

    # add multiple depots:
    data['start'] = list(range(size_order*2, len(time_windows))) # 剩余的节点数对应depots
    # print("data['start'] = ", data['start'])  # [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    data['end'] = list(range(size_order*2, len(time_windows)))
    # print(f"data['end'] = {data['end']}")  # [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]


    return data


# 2 # print_solution.py
def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)

            time_var = time_dimension.CumulVar(index)
            # plan_output += ' {} -> '.format(manager.IndexToNode(index))
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        time_var = time_dimension.CumulVar(index)
        # plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += '{0} time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        if route_distance > 0:
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


# 3 # main.py
def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    # manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           # data['num_vehicles'], data['depot'])
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['start'], data['end'])
    # multi-depots 加了start和end的节点index

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Define cost of each arc.
    def distance_callback(from_index, to_index):
        """Returns the manhattan distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add Time Windows constraint.
    dimension_name = 'Time'
    routing.AddDimension(
        transit_time_callback_index,
        5,  # allow waiting time, a variable used to represent waiting times at the locations (service time?)
        3000,  # maximum time per vehicle
        True,  # Don't force start cumul to zero-Flase.
        dimension_name)
    time_dimension = routing.GetDimensionOrDie(dimension_name)

    time_dimension.SetGlobalSpanCostCoefficient(100)  # 加系数 似乎可加可不加

    # 1. Add time window constraints for each location except depot.
    # for location_idx, time_window in enumerate(data['time_windows']):
    #     # if location_idx == data['depot']:
    #     #     continue
    #     if location_idx in data['start']:
    #         continue
    #     index = manager.NodeToIndex(location_idx)
    #     time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # 2. Add time window constraints for each location except depot. 因为每个点的TW其实前面已经设置过了 包括0-9999 所以其实每个索引可以全设置一遍的
    for location_idx, time_window in enumerate(data['time_windows']):
        # if location_idx == data['depot']:
        #     continue
        # if location_idx in data['start']:
        #     continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle start node.

    # depot_idx = data['depot']
    for depot_idx in range(size_order * 2, (size_order * 2) + len(data['start'])):  # depot的序号是从2倍订单开始 到 加上depot的数量（也就是车的数量）
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data['time_windows'][depot_idx][0],
                data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')


    # Define Transportation Requests.
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)

    # Improve the initial solution by a meta-heuristic algorithm

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.FromSeconds(10)
    search_parameters.time_limit.seconds = 2
    search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)



if __name__ == '__main__':
    main()



