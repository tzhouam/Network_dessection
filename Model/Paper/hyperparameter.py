class Hyperpara:
    def __init__(self):

        self.OUTPUT_FEATURES_NUM = 6
        self.batch_size = 100
        self.prev_loss = 1000
        self.max_epochs = 300
        self.learning_rate = 1e-4
        self.l2 =1e-4               #weight_decay
        self.sequence_len = 100
        self.momentum=0.9
        self.epoch_change={             #change parameters of optimizers by epoch
            80:(1e-5,1e-8),          #epoch:(lr,weight_decay)
            35000:(1e-7,1e-9)
            }
        self.file_path = '/jhcnas1/zhoutaichang/original/'
        self.enhance = '/jhcnas1/zhoutaichang/enhanced/'

        self.cuda = 6
        self.test = self.enhance + 'test/'
        self.train = self.enhance + 'train/'
        self.worker = 10
        self.eps = 12
        # self.device=0,1,2,3,4,5,6,7