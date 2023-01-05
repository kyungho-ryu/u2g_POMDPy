
# _list = [0.388595166163142, 0.5103222557905337, 0.573262839879154, 0.5780463242698892, 0.5849697885196374]
# hovering = 11920
# active_UAV= [6,9,12.8,13,13.5]
# result = []
# for i in range(len(_list)) :
#     _temp = _list[i]*11896.536
#     _temp = _temp + hovering
#     _temp = _temp * active_UAV[i]
#
#     result.append(_temp)
#
# print(result)
# [99257.61830211482, 161919.60378851963, 239869.97775468277, 244357.73578247736, 254868.04099848942]

# case : 12,30 UAV transition
_list = [0.42220543806646527, 0.4509063444108761, 0.45996978851963743, 0.5382678751258811, 0.552870090634441]
hovering = 11920
active_UAV= [5.15,7.2,7.5,9,10.05]
result = []
for i in range(len(_list)) :
    _temp = _list[i]*11896.536
    _temp = _temp + hovering
    _temp = _temp * active_UAV[i]

    result.append(_temp)

print(result)
# [87255.32829577039, 124446.4096241692, 130440.35361027191, 164911.70838670695, 185897.25131238674]
