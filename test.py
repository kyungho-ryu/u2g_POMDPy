def check_init_depth(initMaxDepth, MaxDepth):
    if initMaxDepth == 0:
        return True
    else:
        return initMaxDepth == MaxDepth

print(check_init_depth(0, 6))