# this module calculates power for communication and propulsion. need to derive energy by duration for comm and moving
from examples.u2g import channel
import math

a2a_static_power = 3.5 #Watt
a2g_static_power = 6.8
a2g_load_co = 4

drag_co = 0.012
airDense = 1.225
rotorSol = 0.05
rotorDisc = 0.503
bladeSpeed = 300
rotorRadius = 0.4
uavWeight = 20
fuselage = 0.6
hoverSpeed = 4.03
corrFactor = 0.1

# communication energy
def calA2ACommEnergy(uav1, uav2, sumFlowRate): #user flows between uav1 and uav2
    linkCapa = channel.calA2ALinkRate(uav1, uav2)
    return a2a_static_power + channel.getTxPInWatt() * sumFlowRate/linkCapa

def calA2AMaxCommEnergy(): #user flows between uav1 and uav2
    return a2a_static_power + channel.getTxPInWatt()

def calA2GCommEnergy(uav, lGmu): #list of GMUs for a serving uav
    dataLoad = 0
    subChBand = channel.BAND_A2G/len(lGmu) #even allocation for each GMU
    for g in lGmu:
        linkCapa = channel.calA2GLinkRate(uav, g, subChBand)
        dataLoad += min(g.dnRate / linkCapa, 1) #need to change for actual dnRate instead of demand

    return a2g_static_power + a2g_load_co * channel.getTxPInWatt() * dataLoad / len(lGmu)


def calA2GMaxCommEnergy(): #list of GMUs for a serving uav
    return a2g_static_power + a2g_load_co * channel.getTxPInWatt()


def calUavHoverEnergy():
    _p0 = drag_co/8 * airDense * rotorSol * rotorDisc * pow(bladeSpeed*rotorRadius, 3)
    _pi = (1+corrFactor) * pow(uavWeight, 0.5)/math.sqrt(2*airDense*rotorDisc)
    return _p0, _pi # need to sum all

def calUavFowardEnergy(_p0, _pi, _flightSpeed):
    _bladeProf = _p0 * (1+ 3* pow(_flightSpeed,2)/pow(bladeSpeed*rotorRadius,2))
    _induced = _pi * math.sqrt(math.sqrt(1+ 0.25* pow(_flightSpeed/hoverSpeed, 4))- 0.5*pow(_flightSpeed/hoverSpeed, 2))
    _parasite = 0.5 * fuselage * airDense * rotorSol * rotorDisc * pow(_flightSpeed, 3)
    #print(_bladeProf , _induced , _parasite)
    return _bladeProf + _induced + _parasite

# p_0, p_i = calUavHoverEnergy()
# print(p_0, p_i)
# pv = calUavFowardEnergy(p_0, p_i, 20) #uav speed 20 m/s

#print(pv)