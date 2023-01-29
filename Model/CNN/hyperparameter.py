class Hyperpara:
    def __init__(self):

        self.OUTPUT_FEATURES_NUM = 6
        self.batch_size = 90
        self.prev_loss = 1000
        self.max_epochs = 200
        self.learning_rate = 1e-3
        self.l2 =1e-8              #weight_decay
        self.sequence_len = 100
        self.momentum=0.9
        self.epoch_change={             #change parameters of optimizers by epoch
            100:(1e-4,1e-9),
            150:(1e-5,1e-9),
            250:(1e-6,1e-10),#epoch:(lr,weight_decay)
            350: (1e-7, 1e-11),  # epoch:(lr,weight_decay)

            35000:(1e-7,1e-9)
            }
        self.file_path='/jhcnas1/zhoutaichang/original/'
        self.cuda=0
        self.enhance = '/jhcnas1/zhoutaichang/enhanced/'

        self.test = self.enhance + 'test/'
        self.train = self.enhance + 'train/'
        self.worker=10
        self.eps=0