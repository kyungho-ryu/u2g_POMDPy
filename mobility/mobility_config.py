import numpy as np
class MConfig :
    # trajectories buffer size
    # H = np.inf
    H = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    initialTrip = 2
    Batch = 70
    BatchInitialTrip = 280

    MaxTrajectory =  299
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
    MaxPath = 10

    #density coefficient
    C_density = 1/10

    # probability of specific RO
    RO_prob = 1

    # minimun size to provide information for trajectory prediction in lookup process
    Min_remaining_trajectory = MaxPath
