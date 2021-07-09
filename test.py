import os
import time
import copy
import tqdm
import logging
import argparse
import collections

from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from utils import load_state_dict, create_dir
from datasets.sg2 import StyleGAN2_Data
import models.senet as SENet


def save_image(input, outdir, pred):
    plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(input)

    for pred_type in pred_types.values():
        ax.plot(pred[pred_type.slice, 0],
                pred[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    plt.savefig(outdir)


def test_model(args, model, data, data_loader, device, save_fig=True):
    since = time.time()

    model.eval()

    # Iterate over data.
    for i, batch in enumerate(data_loader):

        inputs = batch['image'].to(device)
        indices = batch['meta']['index']
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()

        for (index, output) in zip(indices, outputs):

            output = output.reshape(1, -1)
            pred = data.inv_scale_label(output)
            pred = pred.reshape(-1, 2)

            input = data.get_image(index, os.path.join(args.root, 'test'))
            save_image(input, os.path.join(args.val_save_dir, f'out_{index}.png'), pred)

            if i == 0:
                answ = data.labels_original[index]
                answ = answ.reshape(-1, 2)
                save_image(input, os.path.join(args.val_save_dir, f'out_{index}_ans.png'), answ)


def initialize_model(args):
    model = SENet.senet50(num_classes=args.n_identity, include_top=True)  # forward output w/ FC layer (dim: 138=NUM_OUT_FT)
    num_in_ft = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_in_ft, args.out_features)
    model.load_state_dict(torch.load(args.model_path))

    return model


def main():
    parser = argparse.ArgumentParser("PyTorch SE-Net Fine-tuning Code")

    parser.add_argument('--image_size', type=int, default=256, help='training data size == (x.size[0])')
    parser.add_argument('--out_features', type=int, default=136, help='number of classes == y')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--root', type=str, default='./datasets/generated', help='training data dir')
    parser.add_argument('--model_path', type=str, default='./ckpt/senet_ckpt_0.0005_1e-05_0.pth', help='pretrained SENet model path')

    parser.add_argument('--save_epoch', type=int, default=1, help='epoch to save images for validation')
    parser.add_argument('--val_save_dir', type=str, default='./assets/out0.0005_1e-05_0', help='save directory for validation file')
    parser.add_argument('--log_dir', type=str, default='log', help='save directory for log file')

    parser.add_argument('--n_identity', type=int, default=8631, help='number of classes pretrained in SENet')
    parser.add_argument('--pretrained_size', type=int, default=224, help='image size pretrained in SENet')

    args = parser.parse_args()

    create_dir(args.val_save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f'senet_{datetime.now().time()}.log')),
            logging.StreamHandler()
        ]
    )

    logging.info('PyTorch Version: ', torch.__version__)
    logging.info('Torchvision Version: ', torchvision.__version__)

    logging.info(f'args: {args}')

    model = initialize_model(args)

    logging.info(model)
    logging.info('Initializing Datasets and Data_loader...')

    data_transforms = transforms.Compose([
                        transforms.Resize(args.pretrained_size),
                        transforms.CenterCrop(args.pretrained_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    data = StyleGAN2_Data(root=args.root, split='test', transform=data_transforms, scale_size=args.image_size)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    logging.info('Test starts!')
    test_model(args, model, data, data_loader, device, save_fig=True)


if __name__ == '__main__':
    main()