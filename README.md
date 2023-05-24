# MPDPSD_UAV_healthcare
OR-Tools for Multi-depot Pickup and Delivery Problem with Soft Deadlines
## 简单记录一下现阶段的实验结果-20230523：
1. 参数设置：
（车辆参数）
vehicle_num_uav = 5  # 5
vehicle_num_human = 10  # 10
vehicle_num = vehicle_num_uav + vehicle_num_human
w_uav = 1  # unitcost_uav per distance # 1
w_human = 50  # unitcost_human per distance # 10-400都可以
capacity_uav = 40  # 30
capacity_human = 100  # 100
speed_uav = 0.2 # speed 的不同影响的是distance的cost结果不同 倍数关系
speed_human = 0.1
（location参数）均匀分布 0-10
# if uniform(0,10) 前面的cost就不能设置的太大 我感觉可能是因为计算限制的问题 反正超过200就跑不出来了
loc_p_coor = torch.FloatTensor(size_order, 2).uniform_(0, 10)  # P locations coordinates
loc_d_coor = torch.FloatTensor(size_order, 2).uniform_(0, 10)  # D locations coordinates

（订单参数）
size_order = 30 # 因此len(TW)就是 15(车)+30(P)+30(D)=75
demand_p = random.choices(range(20, 41), k=size_order)  # 有放回的choices 否则订单数大于20会报错
demand_d = [-x for x in demand_p]  # negative demand in D set
deadline_d = random.sample(range(40, 101), size_order)  # deadlines in D set, otherwise infinity(99999)

3. 实验结果摘录（硬约束）：
C:\CJW_2022\DRL_PDPTW_UAV\venu\Scripts\python.exe C:\CJW_2022\DRL_PDPTW_UAV\simpleVRP.py 
time_windows: 75
distance_matrix: [[0.         5.95191526 0.         ... 3.96181941 2.15440035 0.97736061]
 [5.95191526 0.         5.95191526 ... 2.04630065 6.34482765 6.90714455]
 [0.         5.95191526 0.         ... 3.96181941 2.15440035 0.97736061]
 ...
 [3.96181941 2.04630065 3.96181941 ... 0.         4.70486212 4.89719105]
 [2.15440035 6.34482765 2.15440035 ... 4.70486212 0.         2.5546174 ]
 [0.97736061 6.90714455 0.97736061 ... 4.89719105 2.5546174  0.        ]]
 # P_D index表格：
[[15 45]
 [16 46]
 [17 47]
 [18 48]
 [19 49]
 [20 50]
 [21 51]
 [22 52]
 [23 53]
 [24 54]
 [25 55]
 [26 56]
 [27 57]
 [28 58]
 [29 59]
 [30 60]
 [31 61]
 [32 62]
 [33 63]
 [34 64]
 [35 65]
 [36 66]
 [37 67]
 [38 68]
 [39 69]
 [40 70]
 [41 71]
 [42 72]
 [43 73]
 [44 74]]
time_matrix: [[ 0 29  0 ... 19 10  4]
 [29  0 29 ... 10 31 34]
 [ 0 29  0 ... 19 10  4]
 ...
 [19 10 19 ...  0 23 24]
 [10 31 10 ... 23  0 12]
 [ 4 34  4 ... 24 12  0]]
time_windows: [(0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 99999), (0, 83), (0, 90), (0, 69), (0, 83), (0, 52), (0, 88), (0, 78), (0, 84), (0, 109), (0, 82), (0, 58), (0, 105), (0, 62), (0, 81), (0, 59), (0, 72), (0, 107), (0, 78), (0, 62), (0, 84), (0, 76), (0, 62), (0, 102), (0, 78), (0, 87), (0, 62), (0, 107), (0, 91), (0, 99), (0, 91)]
vehicle_cost_map: [1, 1, 1, 1, 1, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
[40, 40, 40, 40, 40, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
Objective: 7397
Route for vehicle 0:
 0 Load(0) ->  29 Load(29) ->  59 Load(0) ->  34 Load(30) ->  64 Load(0) ->  0 Load(0)
Distance of the route: 759m
Load of the route: 0

Route for vehicle 1:
 1 Load(0) ->  1 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 2:
 2 Load(0) ->  2 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 3:
 3 Load(0) ->  3 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 4:
 4 Load(0) ->  42 Load(24) ->  72 Load(0) ->  33 Load(32) ->  63 Load(0) ->  4 Load(0)
Distance of the route: 865m
Load of the route: 0

Route for vehicle 5:
 5 Load(0) ->  5 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 6:
 6 Load(0) ->  41 Load(23) ->  17 Load(61) ->  71 Load(38) ->  27 Load(78) ->  47 Load(40) ->  57 Load(0) ->  26 Load(36) ->  56 Load(0) ->  6 Load(0)
Distance of the route: 853m
Load of the route: 0

Route for vehicle 7:
 7 Load(0) ->  15 Load(25) ->  21 Load(54) ->  39 Load(82) ->  51 Load(53) ->  69 Load(25) ->  45 Load(0) ->  7 Load(0)
Distance of the route: 769m
Load of the route: 0

Route for vehicle 8:
 8 Load(0) ->  8 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 9:
 9 Load(0) ->  9 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 10:
 10 Load(0) ->  10 Load(0)
Distance of the route: 0m
Load of the route: 0

Route for vehicle 11:
 11 Load(0) ->  44 Load(33) ->  40 Load(72) ->  25 Load(99) ->  55 Load(72) ->  70 Load(33) ->  74 Load(0) ->  11 Load(0)
Distance of the route: 1177m
Load of the route: 0

Route for vehicle 12:
 12 Load(0) ->  16 Load(29) ->  37 Load(68) ->  19 Load(100) ->  67 Load(61) ->  38 Load(81) ->  46 Load(52) ->  49 Load(20) ->  28 Load(53) ->  68 Load(33) ->  58 Load(0) ->  12 Load(0)
Distance of the route: 837m
Load of the route: 0

Route for vehicle 13:
 13 Load(0) ->  23 Load(38) ->  22 Load(59) ->  35 Load(98) ->  53 Load(60) ->  52 Load(39) ->  20 Load(69) ->  36 Load(95) ->  65 Load(56) ->  43 Load(87) ->  66 Load(61) ->  50 Load(31) ->  73 Load(0) ->  13 Load(0)
Distance of the route: 1000m
Load of the route: 0

Route for vehicle 14:
 14 Load(0) ->  24 Load(24) ->  32 Load(61) ->  30 Load(93) ->  62 Load(56) ->  31 Load(88) ->  60 Load(56) ->  18 Load(82) ->  54 Load(58) ->  48 Load(32) ->  61 Load(0) ->  14 Load(0)
Distance of the route: 1137m
Load of the route: 0

Total distance of all routes: 7397m
Total load of all routes: 0
Route for vehicle 0:
0 Time(0,0) -> 29 Time(7,7) -> 59 Time(37,37) -> 34 Time(40,40) -> 64 Time(52,52) -> 0 Time(74,74)
Time of the route: 74min

Route for vehicle 1:
1 Time(0,0) -> 1 Time(0,0)
Time of the route: 0min

Route for vehicle 2:
2 Time(0,0) -> 2 Time(0,0)
Time of the route: 0min

Route for vehicle 3:
3 Time(0,0) -> 3 Time(0,0)
Time of the route: 0min

Route for vehicle 4:
4 Time(0,0) -> 42 Time(0,0) -> 72 Time(40,40) -> 33 Time(46,46) -> 63 Time(60,60) -> 4 Time(85,85)
Time of the route: 85min

Route for vehicle 5:
5 Time(0,0) -> 5 Time(0,0)
Time of the route: 0min

Route for vehicle 6:
6 Time(0,0) -> 41 Time(2,2) -> 17 Time(8,8) -> 71 Time(31,31) -> 27 Time(41,41) -> 47 Time(44,44) -> 57 Time(58,58) -> 26 Time(62,62) -> 56 Time(76,76) -> 6 Time(82,82)
Time of the route: 82min

Route for vehicle 7:
7 Time(0,0) -> 15 Time(8,8) -> 21 Time(16,16) -> 39 Time(32,32) -> 51 Time(34,34) -> 69 Time(49,49) -> 45 Time(64,64) -> 7 Time(72,72)
Time of the route: 72min

Route for vehicle 8:
8 Time(0,0) -> 8 Time(0,0)
Time of the route: 0min

Route for vehicle 9:
9 Time(0,0) -> 9 Time(0,0)
Time of the route: 0min

Route for vehicle 10:
10 Time(0,0) -> 10 Time(0,0)
Time of the route: 0min

Route for vehicle 11:
11 Time(0,0) -> 44 Time(0,0) -> 40 Time(7,7) -> 25 Time(28,28) -> 55 Time(48,48) -> 70 Time(60,60) -> 74 Time(76,76) -> 11 Time(116,116)
Time of the route: 116min

Route for vehicle 12:
12 Time(0,0) -> 16 Time(0,0) -> 37 Time(5,5) -> 19 Time(9,9) -> 67 Time(29,29) -> 38 Time(31,31) -> 46 Time(33,33) -> 49 Time(49,49) -> 28 Time(54,54) -> 68 Time(63,63) -> 58 Time(71,71) -> 12 Time(79,79)
Time of the route: 79min

Route for vehicle 13:
13 Time(0,0) -> 23 Time(0,0) -> 22 Time(5,5) -> 35 Time(6,6) -> 53 Time(11,11) -> 52 Time(25,25) -> 20 Time(38,38) -> 36 Time(49,49) -> 65 Time(53,53) -> 43 Time(56,56) -> 66 Time(59,59) -> 50 Time(75,75) -> 73 Time(85,85) -> 13 Time(95,95)
Time of the route: 95min

Route for vehicle 14:
14 Time(0,0) -> 24 Time(0,0) -> 32 Time(12,12) -> 30 Time(12,12) -> 62 Time(28,28) -> 31 Time(29,29) -> 60 Time(36,36) -> 18 Time(52,52) -> 54 Time(72,72) -> 48 Time(80,80) -> 61 Time(86,86) -> 14 Time(108,108)
Time of the route: 108min

Total time of all routes: 711min

Process finished with exit code 0
