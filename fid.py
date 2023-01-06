import torch
import argparse
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from PIL import Image
import csv

def make_tensor(dir_name, img_name):
    # load image in RGB mode (png files contains additional alpha channel)
    img = Image.open(dir_name+"/"+img_name).convert('RGB')

    # set up transformation to resize and tensor the image
    transform = transforms.Compose([transforms.Resize([229, 229]),
		transforms.PILToTensor()
	])
    
    # apply transformation and convert to Pytorch tensor
    tensor = transform(img)
    tensor = tensor.to(torch.uint8)
    # torch.Size([3, 229, 229])

    return tensor

def calculate_fid(fake_imgs_path, real_imgs_path, output_path):   
    _ = torch.manual_seed(123)
    fid = FrechetInceptionDistance(feature=2048)

    # generate two slightly overlapping image intensity distributions
    imgs_fake = os.listdir(fake_imgs_path)
    imgs_fake_tensors = torch.stack([make_tensor(fake_imgs_path, i) for i in imgs_fake])
    
    imgs_real = os.listdir(real_imgs_path)
    imgs_real_tensors = torch.stack([make_tensor(real_imgs_path, i) for i in imgs_real])

    fid.update(imgs_fake_tensors,real=True)
    fid.update(imgs_real_tensors, real=False)
    fid_val = fid.compute()

    with open(output_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(fid_val)

    return fid_val

def main(args):
    print(args)
    fid_val = calculate_fid(args.path_to_fake, args.path_to_real)
    print("The FID score is:",fid_val)
    return fid_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_fake', type=str,
                        help='Directory that contains the fake images')
    parser.add_argument('--path_to_real', type=str,
                        help='Directory that contains the real images')
    parser.add_argument('--outpath', type=str,
                        help='csv filename that stores the output')
    args = parser.parse_args()
    main(args)

