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

def uavTopoUpdate(_graph, _lUav, _locStat, MAX_XGRID_N, MAX_GRID_INDEX):
    _graph.clear()
    _uavlist = [u.id for u in _lUav]
    _graph.add_nodes_from(_uavlist)
    for u in _lUav:
        _ngbs = getNgbUAVs(u.cell, _locStat, MAX_XGRID_N, MAX_GRID_INDEX)
        for n in _ngbs:
            if n in _lUav and n != u:
                _graph.add_edge(u.id, n.id)


def getNgbUAVs(_gIdx, _locStat, MAX_XGRID_N, MAX_GRID_INDEX):
    adjUavs = []
    l, r, u, b = getNgbCellAvail(_gIdx, MAX_XGRID_N, MAX_GRID_INDEX)
    if l:
        adjUavs += _locStat[_gIdx -1]
    if r:
        adjUavs += _locStat[_gIdx + 1]
    if u:
        adjUavs += _locStat[_gIdx + MAX_XGRID_N]
    if b:
        adjUavs += _locStat[_gIdx - MAX_XGRID_N]
    if l and u:
        adjUavs += _locStat[_gIdx + MAX_XGRID_N - 1]
    if l and b:
        adjUavs += _locStat[_gIdx - MAX_XGRID_N - 1]
    if r and u:
        adjUavs += _locStat[_gIdx + MAX_XGRID_N + 1]
    if r and b:
        adjUavs += _locStat[_gIdx - MAX_XGRID_N + 1]
    return adjUavs

def getNgbCellAvail(_gIdx, MAX_XGRID_N, MAX_GRID_INDEX):
    _left = True if _gIdx % MAX_XGRID_N - 1 >= 0 else False
    _right = True if _gIdx % MAX_XGRID_N + 1 <= MAX_XGRID_N-1 else False
    _upper = True if _gIdx + MAX_XGRID_N <= MAX_GRID_INDEX else False
    _lower = True if _gIdx - MAX_XGRID_N >= 0 else False
    return _left, _right, _upper, _lower

def getLocStat(_lnodes, MAX_GRID_INDEX):
    locStat = {}
    for i in range(MAX_GRID_INDEX + 1):
        locStat[i] = []
    for m in _lnodes:
        locStat[m.cell].append(m)
    return locStat

