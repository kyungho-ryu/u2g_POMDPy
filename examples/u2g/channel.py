import math
import numpy as np
from examples.u2g import util

lightSpeed = 3* pow(10,8)
c1 = 10
c2 = 0.6
a2g_carrier_freq = 5 * pow(10,9) # 5GHz
a2g_wavelen = lightSpeed/a2g_carrier_freq

a2a_carrier_freq = 5 * pow(10,9) # 5GHz vs 900 * pow(10, 6) 900 MHz
a2a_wavelen = lightSpeed/a2a_carrier_freq

RxGain = 5 #isotropic radiator gain, dbi
TxGain = 5
IMP_LOSS = 15
NOISE_FIGURE = 9
THERM_NOISE = -173.83 #dbm
BAND_A2G = 20 * pow(10,6) # 20 MHz
BAND_A2A = 20 * pow(10,6) # 20 MHz
TxPower = 23 #dbm
NLOS_K = 0.2

def getTxPInWatt():
    global TxPower
    return util.dbm2Watt(TxPower)

#a2g link functions ----------------------
def calA2GAngle(dist, height): #uav height, a2g dist are meter
    return 180/math.pi * np.arcsin(height/dist)

def calLOSProb(a2gAngle): #angle is degree, not radian
    return 1/(1+c1 * math.exp(-c2 * a2gAngle + c1 * c2))

def calA2GLinkRate(uavLoc, gmuLoc, channel_bw):
    dist = util.getA2GDist(uavLoc.x, uavLoc.y, uavLoc.h, gmuLoc.x, gmuLoc.y)
    a2gAngle = calA2GAngle(dist, uavLoc.h)
    prob_los = calLOSProb(a2gAngle)
    noise = THERM_NOISE + util.watt2Db(BAND_A2G)
    #print('noise: ', noise, util.watt2Db(BAND_A2G))
    pathloss_los = 20 * math.log10(4 * math.pi * dist / a2g_wavelen)
    pathloss_nlos = 20 * math.log10(4 * math.pi * dist / (NLOS_K*a2g_wavelen))
    #print('pathloss ', pathloss_los, pathloss_nlos)
    RxP = TxPower - (prob_los * pathloss_los + (1-prob_los)* pathloss_nlos) + RxGain + TxGain - IMP_LOSS - NOISE_FIGURE
    #print('rxp ', RxP)
    snr = RxP - noise
    #print('snr:', snr, util.db2Watt(snr))
    r = channel_bw * math.log2(1+ util.db2Watt(snr))
    #print(r, math.log2(1+ util.db2Watt(snr)))
    return r

def setA2GDefaultRadioResource(uav, lGmu):
    subChBand = BAND_A2G / len(lGmu)  # even allocation for each GMU
    for g in lGmu:
        linkCapa = calA2GLinkRate(uav, g, subChBand)
        g.dnRate = g.demand if g.demand <= linkCapa else linkCapa

#a2a link functions -------------------
def calRicianK(height):
    delta = 212.3 * pow(height, -2.221) + 1.289
    return 6.469**2/(2* pow(delta,2))

def calA2ALinkRate(uavLoc1, uavLoc2):
    K = calRicianK(uavLoc1.h)
    dist = util.getA2ADist(uavLoc1.x, uavLoc1.y, uavLoc2.x, uavLoc2.y)
    noise = THERM_NOISE + util.watt2Db(BAND_A2A)
    pathloss_los = 20 * math.log10(4 * math.pi * dist / a2a_wavelen)
    RxP = TxPower - math.sqrt(K/(1+K)) * pathloss_los + RxGain + TxGain - IMP_LOSS - NOISE_FIGURE
    #print('rxp ', RxP)
    snr = RxP - noise
    #print('snr:', snr, util.db2Watt(snr))
    r = BAND_A2G * math.log2(1+ util.db2Watt(snr))
    #print(r, math.log2(1+ util.db2Watt(snr)))
    return r

# for i in range(10, 100, 10):
#     K = calRicianK(i)
#     print('K', K)
    calA2AChannel(500, K)