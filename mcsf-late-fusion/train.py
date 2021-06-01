from configs import get_config
from solver import Solver
from data_loader import get_loader,get_difference_attention

if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')
    print(config)
    train_loader = get_loader(config.mode, config.split_index, config.video_type,config.motion_feaures)
    test_loader = get_loader(test_config.mode, test_config.split_index, config.video_type,config.motion_feaures)
    difference_attention,motion_attention = get_difference_attention(config.video_type)
    solver = Solver(config, train_loader, test_loader,difference_attention,motion_attention)

    solver.build()
    #solver.evaluate(-1)
    solver.train()
