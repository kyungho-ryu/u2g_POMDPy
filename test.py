from enum import IntEnum, unique
@unique
class message_type(IntEnum) :
    t1 = 0
    t2=1
    t3=2

a = message_type(message_type.t1)
print(a)
print(a == message_type.t1.value)
