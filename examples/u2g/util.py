import math
def dbm2Watt(_dbm):
    return math.pow(10, _dbm/10 - 3)

def db2Watt(_db):
    return math.pow(10, _db/10)

def watt2Dbm(watt):
    return 10*math.log10(watt) + 30

def watt2Db(watt):
    return 10*math.log10(watt)

def mWatt2Dbm(mwatt):
    return 10*math.log10(mwatt)

def getA2GDist(_ux, _uy, _uh, _mx, _my):
    dx = abs(_ux - _mx); dy = abs(_uy - _my)
    d2dist = math.sqrt(dx**2 + dy**2)
    return math.sqrt(d2dist**2 + _uh**2)

def getA2ADist(_ux1, _uy1, _ux2, _uy2):
    dx = abs(_ux1 - _ux2)
    dy = abs(_uy1 - _uy2)
    return math.sqrt(dx ** 2 + dy ** 2)

def path2Edges(_lpath):
    edges = []
    for i in range(len(_lpath)):
        if i+1 < len(_lpath):
            edges.append((_lpath[i],_lpath[i+1]))
    return edges
