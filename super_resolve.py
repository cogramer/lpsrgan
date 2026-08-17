import yaml  
import torch  
import argparse  
import models  
import torchvision.transforms as T  
from pathlib import Path  
from train import make_dataloader  
  
def resize_fn(img, size):  
    return T.ToTensor()(  
        T.Resize(size, T.InterpolationMode.BICUBIC)(T.ToPILImage()(img))  
    )  
  
def main(config, save_path):  
    # Build test dataloader (same call used in prepare_testing)  
    test_loader = make_dataloader(config['test_dataset'], tag='test')  
  
    # Load SR model only (no OCR)  
    sv_file = torch.load(config['model']['load'])  
    model_sr, _ = models.make(sv_file['model'], load_model=True)  
    model_sr.cuda()  
    model_sr.eval()  
  
    save_path.mkdir(parents=True, exist_ok=True)  
  
    with torch.no_grad():  
        for idx, batch in enumerate(test_loader):  
            if idx >= 5:  
                break  
  
            imgs_lr = batch['lr'].view(-1, 3, 32, 48).cuda()  
            imgs_sr = model_sr(imgs_lr)  
  
            for i, img_sr in enumerate(imgs_sr):  
                img_sr = T.ToPILImage()(img_sr.cpu())  
                img_sr.save(save_path / f"sr_{idx:03}_{i:03}.png")  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config')  
    parser.add_argument('--save', default='./sr_results')  
    args = parser.parse_args()  
  
    with open(args.config, 'r') as f:  
        config = yaml.load(f, Loader=yaml.FullLoader)  
  
    main(config, Path(args.save))