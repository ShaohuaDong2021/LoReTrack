from lib.utils import TensorDict


class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    # def __init__(self, net_backbone, net_head, net_teacher_backbone, net_teacher_head, objective):
    def __init__(self, net, net_teacher, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        # self.net_backbone = net_backbone
        # self.net_head = net_head
        # self.net_teacher_backbone = net_teacher_backbone
        # self.net_teacher_head = net_teacher_head
        self.net = net
        self.net_teacher = net_teacher
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        # self.net_backbone.to(device)
        # self.net_head.to(device)
        # self.net_teacher_backbone.to(device)
        # self.net_teacher_head.to(device)
        self.net.to(device)
        self.net_teacher.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        # self.net_backbone.train(mode)
        # self.net_head.train(mode)
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)