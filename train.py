import os
import torch
from data.data_load_deepgyro import train_dataloader
from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
from tqdm import tqdm


def _train(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    dataloader = train_dataloader(config.data_dir, config.batch_size, config.num_worker)

    if config.restore:
        model_load = os.path.join(config.model_save_dir, "model.pkl")
        state_dict = torch.load(model_load)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict['epoch']
        print("\n Model Restored, epoch = %4d\n" % (state_dict['epoch']))
    else:
        start_epoch = 1

    writer = SummaryWriter()
    epoch_adder = Adder()
    iter_adder = Adder()
    epoch_timer = Timer('h')
    iter_timer = Timer('m')
    max_iter = len(dataloader)
    for epoch_idx in range(start_epoch, config.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(tqdm(dataloader)):
            input_img, label_img = batch_data
            input_img = input_img.to(device) - 0.5
            label_img = label_img.to(device) - 0.5

            optimizer.zero_grad()
            pred_img = model(input_img)

            loss = criterion(pred_img, label_img)
            loss.backward()
            optimizer.step()

            iter_adder(loss.item())
            epoch_adder(loss.item())

            if (iter_idx + 1) % config.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d Loss: %7.4f" % (iter_timer.toc(), epoch_idx,
                                                                             iter_idx + 1, max_iter,
                                                                             iter_adder.average()))
                writer.add_scalar('Loss', iter_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                iter_timer.tic()
                iter_adder.reset()

        if epoch_idx==1 or epoch_idx % config.save_freq == 0:
            # save_name = os.path.join(config.model_save_dir, 'model_%d.pkl' % epoch_idx)
            save_name = os.path.join(config.model_save_dir, 'model.pkl')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Loss: %7.4f" % (
        epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        epoch_adder.reset()

        if epoch_idx % config.valid_freq == 0:
            val = _valid(model, config)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)

    # save_name = os.path.join(config.model_save_dir, 'Model.pkl')
    # torch.save({'model': model.state_dict()}, save_name)
