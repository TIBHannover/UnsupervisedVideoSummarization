# -*- coding: utf-8 -*-
from tensorboardX import SummaryWriter
import pickle


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        """
        Extended SummaryWriter Class from tensorboard-pytorch (tensorbaordX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer
        """
        super(TensorboardWriter, self).__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module, step_i):
        """
        module: nn.Module
        """
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        self.add_histogram(name, values, step_i)

def init_scores_dict(keys):
    all_scores = dict()
    all_scores = {el: [] for el in keys}  # init dict
    return all_scores

def drop_file_extension(file_name):
    if file_name is None:
        raise ValueError
    file_name = file_name.split('.')[:-1]
    return '.'.join(file_name)

def open_pickle_file(filename):
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                return pickle.load(openfile)
            except EOFError:
                print(EOFError)
                break
