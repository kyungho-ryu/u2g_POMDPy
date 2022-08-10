import csv
import math
import numpy as np
from mobility.semi_lazy import SLModel as SL
from mobility.mobility_config import MConfig  as p
import pandas as pd
import matplotlib.pyplot as plt
from mobility.utils import set_coordinate, get_cellCoordinate
import logging, sys
my_logger = logging.getLogger('POMDPy')

my_format = "%(asctime)s - %(name)s - %(message)s"

sys_handler = logging.StreamHandler(sys.stdout)
sys_handler.setFormatter(logging.Formatter(my_format))
my_logger.setLevel(logging.DEBUG)
sys_handler.setLevel(logging.DEBUG)
my_logger.addHandler(sys_handler)


result = {}
f = open('result(H=100).csv', 'a', encoding='utf-8')
wr = csv.writer(f)

for index in range(51) :
    NumOfMO = 50
    error_path = {}
    id = index
    sl = SL(NumOfMO, 400, 14, 5, id)
    RO = []
    file = "/home/kyungho/project/POMDPy/mobility/trajectory/MO" + str(id) + "_traj.csv"
    traj = pd.read_csv(file)

    sl.test_update_trajectory(id, 50)
    for i in range(200) :
        plt_x_prediction = []
        plt_y_prediction = []
        plt_x_real = []
        plt_y_real = []

        ro = sl.get_reference_objects(id)
        path = sl.prediction_probabilistic_path(ro, p.theta, id)
        real_path = []

        if len(path) == 0 :
            path = [0]

        if len(path) == 1  :
            updated_time = sl.traj[id].updated_time + sl.sampling_interval
            sl.update_trajectory(sl.MOS[id], id, updated_time)
        else :
            init_updated_time = sl.traj[id].updated_time
            for j in range(1, len(path)) :
                updated_time  = init_updated_time + (sl.sampling_interval * j)
                pos, coordinate = sl.get_traj(id, updated_time)
                # print("len : {}, path : {}/{}, time : {}".format(j, pos, coordinate, updated_time))
                if j == 1 :
                    sl.update_trajectory(sl.MOS[id], id, updated_time)

                error_x = coordinate[0] - path[j][0]
                error_y = coordinate[1] - path[j][1]
                error = math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2))

                if not j in error_path :
                    error_path[j] = []

                error_path[j].append(error)

                plt_x_prediction.append(path[j][0])
                plt_y_prediction.append(path[j][1])
                plt_x_real.append(coordinate[0]- 0.1)
                plt_y_real.append(coordinate[1]- 0.1)

        # print("real : ", plt_x_real)
        # print("real : ", plt_y_real)
        # print(plt_x_prediction)
        # print(plt_y_prediction)
        # plt.plot(plt_x_prediction, plt_y_prediction)
        # plt.plot(plt_x_real, plt_y_real, '--')
        # plt.xlim(0,5)
        # plt.ylim(0,5)
        # plt.show()
        # print("-----------------------------")
        # for z in range(len(path), 11) :
        #     if not z in error_path :
        #         error_path[z] = []
        #
        #     error_path[z].append(-1)


    for k, v in error_path.items() :
        # print("{} - {}".format(k, len(v)))
        if k % 2 == 0 :
            continue
        for vv in v :
            wr.writerow([k, vv, 0.])

f.close()
#
# for k, v in result.items() :
#     print("path length : {}=====================================================".format(k))
#     for kv, vv in v.items() :
#         print("{} - {}".format(kv, np.average(vv)))
