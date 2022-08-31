class Sample :
    def __init__(self):
        self.s_list = []
        self.a_list = []
        self.td_target = []
        self.NumSample = 0

    def add_sample(self, s, a, td_target):
        self.s_list.append(s)
        self.a_list.append(a)
        self.td_target.append(td_target)
        self.NumSample +=1

        return self.NumSample

    def reset_sample(self):
        self.s_list = []
        self.a_list = []
        self.td_target = []
        self.NumSample = 0