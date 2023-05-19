# 0518：尝试修改多个depot 成功，cost已经加上vehicle的不同，capacity满足最大限制；
# 发现受到capacity的影响 而且由于cost和距离相关 因此最优解倾向于最少的车辆完成所有任务，且order可以大于20
# 在参数（车辆5-10，w_cost 1-100, capacity both 40的情况下 一般可以利用到3辆车 因为demand的范围20-40
# 0519：尝试给每个订单加上时间窗
"""Capacited Vehicles Routing Problem (CVRP)."""
import numpy as np
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
from torch.utils.data import Dataset

# parameters setting:
vehicle_num_uav = 5  # 5
vehicle_num_human = 10  # 10
vehicle_num = vehicle_num_uav + vehicle_num_human

w_uav = 1  # unitcost_uav per distance # 1
w_human = 100  # unitcost_human per distance # 10
H_d = 100  # penalty cost for exceeding the ddl in set D, otherwise 0

capacity_uav = 40  # 30
capacity_human = 40  # 100
speed_uav = 0.2
speed_human = 0.1

# size_order = 20
# demand_p = random.sample(range(20, 41), size_order)  # demand range 20,40
size_order = 20
demand_p = random.choices(range(20, 41), k=size_order)  # 有放回的choices 否则订单数大于20会报错
demand_d = [-x for x in demand_p]  # negative demand in D set

# np.random.randint(20, 41)
deadline_d = random.sample(range(40, 101), size_order)  # deadlines in D set, otherwise infinity
# print(deadline_d)  # 只有末尾时间约束l_i 因此改成时间窗约束的时候 每个e_i设成0
# 时间窗设置 标准模型里面 是每个节点（包括p, d, depots）都有的：
# data['time_matrix']: An array of travel times between locations.
# data['time_windows']: An array of time windows for the locations

time_window1 = np.zeros((size_order*2 + vehicle_num, 2), dtype=int) # sequence: p_set, d_set, depots
for i in range(vehicle_num):  # depots window 99999 # size_order
    time_window1[i][0] = 0
    time_window1[i][1] = 99999
for i in range(vehicle_num, vehicle_num + size_order):
    time_window1[i][0] = 0
    time_window1[i][1] = 99999  # deadline_p 99999
for i in range(vehicle_num + size_order, (size_order*2 + vehicle_num)):
    time_window1[i][0] = 0
    time_window1[i][1] = np.random.randint(40, 101) + 30  # 加一些slack 30min 防止没有可行解 # deadline_d

time_windows = []
for row in time_window1:
    time_windows.append((int(row[0]), int(row[1])))

print(time_windows)
# print(len(time_windows)) # 55 = 20+20+15
# print(time_windows.shape)

# distance calculate func: 后面又写了一个 没用上这个
def dist_pd(loc_p, loc_d):
    diff = loc_p - loc_d
    square_dist = torch.sum(diff ** 2, dim=-1)
    dist = torch.sqrt(square_dist)
    return dist


loc_p_coor = torch.FloatTensor(size_order, 2).uniform_(0, 10)  # P locations coordinates
loc_d_coor = torch.FloatTensor(size_order, 2).uniform_(0, 10)  # D locations coordinates

dist_pds = dist_pd(loc_p_coor, loc_d_coor)  # dist from p to d one-by-one
depots_coor = loc_p_coor[torch.randint(0, size_order, (vehicle_num,))]  # depots_coordinate sample from locations_p
# torch.randint(),是有放回的采样 因此车辆的起始点可能是同一个地方（坐标）
# change into array:
loc_p_array = loc_p_coor.numpy()
loc_d_array = loc_d_coor.numpy()
loc_depots = depots_coor.numpy()

# all locations, note the index sequence: (new) Depots - Pickup - Delivery
loc_all = np.concatenate((loc_depots, loc_p_array, loc_d_array), axis=0)

# 计算点之间的距离matrix pairwise:
distance_matrix = np.zeros((len(loc_all), len(loc_all)))  # 不要dtype=int 不然uniform 0-1的时候距离函数全是0了 时间窗函数必须是int
for i in range(len(loc_all)):
    for j in range(len(loc_all)):
        distance_matrix[i][j] = np.sqrt(np.sum((np.array(loc_all[i]) - np.array(loc_all[j])) ** 2))
print("distance_matrix:", distance_matrix)

# For each pair, the first entry is index of the pickup location, and the second is the index of the delivery location.
pdp_index = np.zeros((size_order, 2), dtype=int)
for i in range(size_order):
    pdp_index[i][0] = int(vehicle_num + i)
    pdp_index[i][1] = int(vehicle_num + i + size_order)
print(pdp_index)  # (size_order, 2) size

# Travel time matrix：each pairwise node to node time
vehicle_speed = 0.2  # 怎么改成不同的speed还没研究
# time_matrix = {}  #
# for from_node in range(len(data['distance_matrix'])):
#     time_matrix[from_node] = {}
#     for to_node in range(len(data['distance_matrix'])):
#         time_matrix[from_node][to_node] = int(data['distance_matrix'][from_node][to_node] / vehicle_speed)
time_matrix = np.zeros((len(loc_all), len(loc_all)), dtype=int)
for i in range(len(loc_all)):
    for j in range(len(loc_all)):
        time_matrix[i][j] = (np.sqrt(np.sum((np.array(loc_all[i]) - np.array(loc_all[j])) ** 2))) / vehicle_speed


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distance_matrix
    data['pickups_deliveries'] = pdp_index
    data['demands'] = [None] * int(len(time_windows))  # 55个点 总长度
    # data['demands'][40] = 0
    for depot in range(vehicle_num):
        data['demands'][depot] = int(0)
    for node in (data['pickups_deliveries']):
        data['demands'][node[0]] = np.random.randint(20, 41)
        data['demands'][node[1]] = -1 * data['demands'][node[0]]

    data['num_vehicles'] = vehicle_num
    # [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    data['vehicle_capacities'] = [None] * (data['num_vehicles'])
    for vehicle_id in range(vehicle_num_uav):
        data['vehicle_capacities'][vehicle_id] = capacity_uav  # 30
    for vehicle_id in range(vehicle_num_uav, vehicle_num):
        data['vehicle_capacities'][vehicle_id] = capacity_human  # 100
    print(data['vehicle_capacities'])

#     data['depot'] = 0
    data['starts'] = list(range(vehicle_num))  # [0, 3, 4, 5]
    data['ends'] = list(range(vehicle_num))
        # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,0]
        # [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] #list(range(vehicle_num))  # [0, 3, 4, 5]  # [2, 10, 7, 6]
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
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
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'], data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # (new) Define cost of each arc:
    # all SetArcCostEvaluatorOfAllVehicles()替换成下面的函数routing.SetArcCostEvaluatorOfVehicle()
    vehicle_cost_map = [None] * vehicle_num
    for vehicle_id in range(vehicle_num_uav):
        vehicle_cost_map[vehicle_id] = int(w_uav)  # 1
    for vehicle_id in range(vehicle_num_uav, vehicle_num):
        vehicle_cost_map[vehicle_id] = int(w_human)  # 10
    print("vehicle_cost_map:", vehicle_cost_map)

    def distance_callback(from_index, to_index):  # cost_callback
        # Define travel cost of each arc.
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        vehicle_cost = vehicle_cost_map[vehicle_i][0]  # vehicle_i 是调用下面SetArcCostEvaluatorOfVehicle函数时传入的参数arg2
        return data['distance_matrix'][from_node][to_node] * vehicle_cost

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    for vehicle_i in range(len(vehicle_cost_map)):
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle_i)  # 区别all的arcCost函数

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


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



    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        False,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)


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
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()