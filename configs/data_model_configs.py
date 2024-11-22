"""
@author: Mingyang Liu
@contact: mingyang1024@gmail.com
"""

def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class UCIHAR():
    """
    UCIHAR: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ
    """
    def __init__(self):
        super(UCIHAR, self)
        self.scenarios = [("2", "11"),   ("6", "23"),  ("7", "13"), ("9", "18"),  ("12", "16"),
                          ("13", "19"),  ("18", "21"), ("20", "6"), ("23", "13"), ("24", "12")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 64
        self.avg_mode = 10
       
       

class WISDM(object):
    """
    WISDM: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B
    """
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.scenarios = [("2", "32"), ("4", "15"), ("7", "30"), ('12','7'),   ('12','19'),
                          ('18','20'), ('20','30'), ("21", "31"),("25", "29"), ('26','2')]
        self.sequence_len = 128
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 64
        self.avg_mode = 16

class HHAR_P(object):
    """
    HHAR-P: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO
    """
    def __init__(self):
        super(HHAR_P, self).__init__()
        self.scenarios =  [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),
                           ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

    
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.mid_channels = 64 * 2
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 64
        self.avg_mode = 10
    


class FD(object):
    """
    FD: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN
    """
    def __init__(self):
        super(FD, self).__init__()
        self.scenarios = [
                          ("0", "1"), ("0","2"), ("0", "3"), ("1", "0"), ("1", "2"),
                          ("2", "0"), ("2", "1"),("2", "3"), ("3", "0"), ("3", "2"),
                          ]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.sequence_len = 5120
        self.num_classes = 3
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 300
        self.avg_mode = 30
       


class CAP(object):
    """
    CAP: https://woods-benchmarks.github.io/cap.html
    """
    def __init__(self):
        super(CAP, self).__init__()
        self.scenarios = [ 
                           ("0", "1"), ("0", "3"), ("0", "4"), ("1", "0"), ("1", "4"), 
                           ("2", "3"), ("3", "0"), ("3", "1"), ("4", "1"), ("4", "3"),
                         ]
        self.class_names = ['Awake', 'NREM1', 'NREM2', 'NREM3', 'NREM4', 'REM']
        self.sequence_len = 3000
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # Model configs
        self.input_channels = 19
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 300
        self.avg_mode = 40
        

class PCL(object):
    """
    PCL: https://woods-benchmarks.github.io/pcl.html
    """
    def __init__(self):
        super(PCL, self).__init__()
        self.scenarios = [
                          ("0", "1"), ("0", "2"),
                          ("1", "0"), ("1", "2"),
                          ("2", "0"), ("2", "1"),
                          ]
        self.class_names = ['LeftHand', 'RightHand']
        self.sequence_len = 750
        self.num_classes = 2
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = True

        # Model configs
        self.input_channels = 48
        self.kernel_size = 15
        self.stride = 3
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 400
        self.avg_mode = 30
       

class EMG(object):
    """
    EMG: https://github.com/microsoft/robustlearn/tree/main/diversify
    """
    def __init__(self):
        super(EMG, self).__init__()
        self.scenarios = [
                          ("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("1", "3"),
                          ("2", "0"), ("2", "1"), ("2", "3"), ("3", "1"), ("3", "2"),
                          ]
        self.class_names = ['0', '1', '2', '3', '4', '5']
        self.sequence_len = 200
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # Model configs
        self.input_channels = 8
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 100
        self.avg_mode = 25


class HHAR_D(object):
    """
    HHAR-D: https://woods-benchmarks.github.io/hhar.html
    """
    def __init__(self):
        super(HHAR_D, self).__init__()
        self.scenarios = [
                          ("0", "1"), ("0", "2"), ("0", "3"), ("0", "4"), ("1", "0"),  
                          ("1", "3"), ("1", "4"), ("2", "1"), ("3", "4"), ("4", "1"),
                         ]
        self.class_names = ['0', '1', '2', '3', '4', '5']
        self.sequence_len = 500
        self.num_classes = 6
        self.shuffle = True
        self.normalize = True
        self.fft_normalize = False

        # Model configs
        self.input_channels = 6
        self.kernel_size = 10
        self.stride = 2
        self.dropout = 0.5
        self.mid_channels = 64
        self.t_feat_dim = 128
        self.features_len = 1
        self.period = 250
        self.avg_mode = 40