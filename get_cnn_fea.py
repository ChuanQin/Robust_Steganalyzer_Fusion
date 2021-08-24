import torch
import tensorflow as tf
import torchvision
import os
import argparse
from scipy import io
import data
from ext_fea_lib import ext_cnn_fea

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, required=True,)
    parser.add_argument('--cover_dir', type=str, required=True,)
    parser.add_argument('--stego_dir', type=str, required=True,)
    parser.add_argument('--adv_dir', type=str, required=True,)
    parser.add_argument('--cover_feature_path', type=str, required=True,)
    parser.add_argument('--stego_feature_path', type=str, required=True,)
    parser.add_argument('--adv_feature_path', type=str, required=True,)
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

def set_dataloader(args):
    # num_image = len(os.listdir(args.cover_dir))
    # random_images = np.arange(0, num_image)
    # np.random.seed(3)
    # np.random.shuffle(random_images)
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
    ])
    cover_data = data.ImageWithNameDataset(
                    img_dir = args.cover_dir, 
                    transform = transform)
    stego_data = data.ImageWithNameDataset(
                    img_dir = args.stego_dir, 
                    transform = transform)
    adv_data = data.ImageWithNameDataset(
                    img_dir = args.adv_dir, 
                    ref_dir = args.cover_dir, 
                    transform = transform)
    cover_loader = torch.utils.data.DataLoader(
                    cover_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    stego_loader = torch.utils.data.DataLoader(
                    stego_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    adv_loader = torch.utils.data.DataLoader(
                    adv_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)

    return cover_loader, stego_loader, adv_loader

args = parse_args()
cover_loader, stego_loader, adv_loader = set_dataloader(args)
# get cover features
cover_feature, names = ext_cnn_fea(args.model_type, cover_loader, args.load_path, args.batch_size)
# get stego features
stego_feature, names = ext_cnn_fea(args.model_type, stego_loader, args.load_path, args.batch_size)
# get adv features
adv_feature, names = ext_cnn_fea(args.model_type, adv_loader, args.load_path, args.batch_size)

# save features
if not os.path.exists('/'.join(args.cover_feature_path.split('/')[:-1])):
    os.makedirs('/'.join(args.cover_feature_path.split('/')[:-1]))
io.savemat(args.cover_feature_path, {'F':cover_feature, 'names': names})
io.savemat(args.stego_feature_path, {'F':stego_feature, 'names': names})
io.savemat(args.adv_feature_path, {'F':adv_feature, 'names': names})