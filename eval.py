import os
import torch
import torchvision
from torchvision.transforms import functional as F
from skimage.measure import compare_psnr, compare_ssim
from utils import Adder, calculate_psnr
from data.data_load_deepblind import test_dataloader


def _eval(model, config):
    state_dict = torch.load(config.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)

    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img = data

            save_name = os.path.join(config.result_dir, '%d' % (iter_idx) + '.png')
            input_img = input_img.to(device) - 0.5

            pred = model(input_img)

            if config.valid_save ==True:
                torchvision.utils.save_image(pred.data + 0.5, save_name)

            psnr = compare_psnr(label_img[0].cpu().numpy(),pred[0].cpu().numpy()+0.5)
            psnr_adder(psnr)

            print('%d iter PSNR: %.2f' % (iter_idx + 1, psnr))

        print('The average PSNR is %.2f dB' % (psnr_adder.average()))