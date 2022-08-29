import os, psutil, gc

def check_momory(logger) :
    percent = psutil.virtual_memory()[2]
    logger.info("memory_usage_percent: : {}".format(percent))
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
    logger.info("Current memory KB: : {} KB".format(current_process_memory_usage_as_KB))

    return percent
def clean_memory(logger) :
    logger.info("Distroy simulation object : {}".format(gc.collect()))