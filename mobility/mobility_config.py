class MConfig :
    # trajectories buffer size
    H = 300
    initialTrip = H

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