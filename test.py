import collections

a = collections.deque()
b = collections.deque(maxlen=4)
a.append(0)
a.append(1)
a.append(2)
b.append(0)
b.append(1)
b.append(2)
b.append(3)
b.append(4)

index = 2
print(b)
start = len(b)-(index+2)
end = len(b)-index
c = list(b)
print(c[start:end])
