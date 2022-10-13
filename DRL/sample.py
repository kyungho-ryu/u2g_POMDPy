
class Sample :
    def __init__(self, _MaxNumIterationForBeliefNode, initMaxDepth=0):
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

        self.NumIterationForBeliefNode = {}
        self.MaxNumIterationForBeliefNode = _MaxNumIterationForBeliefNode

        self.NumSample = 0
        self.MaxDepth = 0
        self.initMaxDepth = initMaxDepth


    def check_depth(self, tree_depth):
        # if tree_depth >=5 : return False
        if self.MaxDepth < tree_depth :
            self.MaxDepth = tree_depth
            self.NumIterationForBeliefNode = {}
            self.reset_sample()
            self.reset_batch()
            return True
        elif self.MaxDepth == tree_depth :
            return True
        else :
            return False

    def check_init_depth(self):
        if self.initMaxDepth == 0 :
            return True
        else :
            return self.initMaxDepth == self.MaxDepth

    def add_batch_sample(self, id, s, a, logprob, r, termial):
        if id not in self.NumIterationForBeliefNode :
            self.NumIterationForBeliefNode[id] = 0

        if self.NumIterationForBeliefNode[id] < self.MaxNumIterationForBeliefNode :
            self.s_batch.append(s)
            self.a_batch.append(a)
            self.logprob_batch.append(logprob)
            self.r_batch.append(r)
            self.termial_batch.append(termial)

            self.NumIterationForBeliefNode[id]+=1
            return True

        else :
            self.reset_batch()
            return False

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

    def set_init_depth(self):
        if self.initMaxDepth == 0 :
            self.initMaxDepth = self.MaxDepth

    def get_batch_len(self):
        return len(self.s_batch)