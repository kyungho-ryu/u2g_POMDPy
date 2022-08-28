a = {1:2, 5:4, 3:2}

b = list(a.keys())[0]
a.pop(b)
print(b)
print(a)
print(len(a.keys()))