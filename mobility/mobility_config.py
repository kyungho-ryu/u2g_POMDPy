import numpy as np
class MConfig :
    # trajectories buffer size
    # H = np.inf
    H = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    initialTrip = 2
    # initialUnObservedLength = [i for i in range(5)]
    initialUnObservedLength = [5]
    Batch = 30
    # BatchInitialTrip = 280
    BatchInitialTrip = 400
    # 300 * 400 = 1200 20m
    # MaxTrajectory =  299
    # backward trajectory size
    BH = 2

    # interval
    interval = 60
    velocity = 1.4

    # coordinate
    x0 = 477
    y0 = 634
    xE = 2422
    yE = 2326

    # threshold
    theta = 0.
    MaxPath = 20

    #density coefficient
    C_density = 1

    # probability of specific RO
    RO_prob = 1

    # minimun size to provide information for trajectory prediction in lookup process
    Min_remaining_trajectory = MaxPath
