
class Sample :
    def __init__(self):
        self.s_list = []
        self.a_list = []
        self.logprob_list = []
        self.r_list = []
        self.termial_list = []

        self.s_batch = []
        self.a_batch = []
        self.logprob_batch = []
        self.r_batch = []
        self.termial_batch = []

        self.NumSample = 0
        self.MaxDepth = 0


    def check_depth(self, tree_depth):
        if self.MaxDepth < tree_depth :
            self.MaxDepth = tree_depth
            self.reset_sample()
            self.reset_batch()
            return True
        elif self.MaxDepth == tree_depth :
            return True
        else :
            return False

    def add_batch_sample(self, s, a, logprob, r, termial):
        self.s_batch.append(s)
        self.a_batch.append(a)
        self.logprob_batch.append(logprob)
        self.r_batch.append(r)
        self.termial_batch.append(termial)

    def add_sample(self):
        if len(self.s_batch) != 0 :
            self.s_list.append(self.s_batch)
            self.a_list.append(self.a_batch)
            self.logprob_list.append(self.logprob_batch)
            self.r_list.append(self.r_batch)
            self.termial_list.append(self.termial_batch)
            self.NumSample +=1

    def reset_sample(self):
        self.s_list = []
        self.a_list = []
        self.logprob_list = []
        self.r_list = []
        self.termial_list = []
        self.NumSample = 0

    def reset_batch(self):
        self.s_batch = []
        self.a_batch = []
        self.logprob_batch = []
        self.r_batch = []
        self.termial_batch = []


    def get_batch_len(self):
        return len(self.s_batch)