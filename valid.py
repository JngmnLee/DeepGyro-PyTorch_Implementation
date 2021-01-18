import torch
from torchvision.transforms import functional as F
from data.data_load_deepgyro import valid_dataloader
from utils import Adder, calculate_psnr
from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm
import torchvision
def _valid(model, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(config.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            input_img, label_img = data
            input_img = input_img.to(device) - 0.5

            pred_img = model(input_img)

            if config.valid_save ==True:
                torchvision.utils.save_image(pred_img.data + 0.5, 'results/' + config.model_name + '/valid/%03d_result.png' % idx)
                torchvision.utils.save_image(input_img.data + 0.5, 'results/' + config.model_name + '/valid/%03d.png' % idx)

            psnr = compare_psnr(label_img[0].cpu().numpy(), pred_img[0].cpu().numpy()+0.5)
            psnr_adder(psnr)

    model.train()
    return psnr_adder.average()

