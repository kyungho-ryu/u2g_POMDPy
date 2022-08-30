import gc, os, psutil

class test :
    def __init__(self, y):
        self.x = self
        self.y = y
pid = os.getpid()
current_process = psutil.Process(pid)
current_process_memory_usage_as_MB = current_process.memory_info()[0] / 2. ** 20
print("Current memory MB: : {} MB".format(current_process_memory_usage_as_MB))

a = ["asdasdasd" for _ in range(1000000)]
b = test(a)
# b = None
del b
# c = gc.collect()
# gc.enable()
# print(c)
pid = os.getpid()
current_process = psutil.Process(pid)
current_process_memory_usage_as_MB = current_process.memory_info()[0] / 2. ** 20
print("Current memory MB: : {} MB".format(current_process_memory_usage_as_MB))
