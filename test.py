# total = 1.32e+5
# hovering = 7946
# Flying = 0 # 1 -> 7944, 2-> 8368, 3-> 9711 4-> 12338 4\root2 -> 20548
# N = 11.5
#
# temp = (total - (hovering+Flying) * N)
#
# print(temp/N)

#energy.calUavFowardEnergy(self.p0, self.p1, _vel)) 11896.536221403663
# hoveringEnergy 11920.07757945932
# totalPropEnergy 23816.613800862982
# Max Power Consumption : 312421.2430439399  # 12
# Max Power Consumption : 393326.29814645014 # 20

# energy.calUavFowardEnergy(self.p0, self.p1, _vel)) 7944.379488484893
# hoveringEnergy 7946.718386306213
# totalPropEnergy 15891.097874791107
# Max Power Consumption : 253440.96070263797
total = 8.3e+4
UAV = 7.2
hovering = 7946

y = total / UAV
# print(y-hovering)

# case : 20,20 UAV transition
_list = [3087,3582,4554,4592,4647]
for i in _list :
    print(i/7944)
# 0.388595166163142
# 0.5103222557905337
# 0.573262839879154
# 0.5780463242698892
# 0.5849697885196374

# case : 12,30 UAV transition
# _list = [3354,3859,3654,4276,4392]
# for i in _list :
#     print(i/7944)

# 0.42220543806646527
# 0.4509063444108761
# 0.45996978851963743
# 0.5382678751258811
# 0.552870090634441
