import logging
import time
from tqdm import tqdm
import io

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level
    def write(self,buf):
        self.buf = buf.strip('\r ')
    def flush(self):
        self.logger.log(self.level, self.buf)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tqdm_out = TqdmToLogger(logger,level=logging.INFO)
    for x in tqdm(range(100),file=tqdm_out,mininterval=5):
        time.sleep(.5)