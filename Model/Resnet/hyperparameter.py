class Hyperpara:
    def __init__(self):

        self.OUTPUT_FEATURES_NUM = 6
        self.batch_size = 95
        self.prev_loss = 1000
        self.max_epochs = 300
        self.learning_rate = 1e-3
        self.l2 =1e-9               #weight_decay
        self.sequence_len = 100
        self.momentum=0.9
        self.epoch_change={             #change parameters of optimizers by epoch
            20:(1e-4,0),
            40:(1e-5,0), #epoch:(lr,weight_decay)
            35000:(1e-7,1e-9)
            }
        self.train='/jhcnas1/zhoutaichang/enhanced/train/'
        self.test='/jhcnas1/zhoutaichang/enhanced/test/'
        self.file_path = '/jhcnas1/zhoutaichang/original/'

        self.cuda=6
        self.eps=12
        self.worker=40
        self.width=7
        # self.device=0,1,2,3,4,5,6,7