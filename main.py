import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
import argparse
from torch.backends import cudnn
from models.dncnn import build_net
from models.DeepGyro import DeepBlind, DeepGyro
from train import _train
from eval import _eval
from torchsummary import summary


def main(config):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(config.model_save_dir)
    if not os.path.exists('results/' + config.model_name + '/'):
        os.makedirs('results/' + config.model_name + '/')
    if not os.path.exists('results/' + config.model_name + '/valid/'):
        os.makedirs('results/' + config.model_name + '/valid/')
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # model = DeepBlind()
    model = DeepGyro()
    summary(model.cuda(), (1, 512, 512))

    if torch.cuda.is_available():
        model.cuda()
    if config.mode == 'train':
        _train(model, config)
    elif config.mode == 'test':
        _eval(model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', type=str, default='DeepGyro')
    parser.add_argument('--data_dir', type=str, default='')

    # Train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--valid_save', type=bool, default=True)
    parser.add_argument('--restore', type=bool, default=True)

    # Test
    parser.add_argument('--test_model', type=str, default='results/DeepGyro/weights/model.pkl')
    parser.add_argument('--mode', type=str, default='train')

    config = parser.parse_args()
    config.model_save_dir = os.path.join('results/', config.model_name, 'weights/')
    config.result_dir = os.path.join('results/', config.model_name, 'eval/')
    print(config)
    main(config)
