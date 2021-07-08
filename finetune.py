import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import time
import copy
from datasets.sg2 import StyleGAN2_Data
import models.senet as SENet
from utils import load_state_dict
import collections
import matplotlib.pyplot as plt
import os
import tqdm
import logging
from datetime import datetime


N_IDENTITY = 8631
NUM_OUT_FT = 68 * 2
MODEL_PATH = os.path.join('pretrained', 'senet50_scratch_weight.pkl')
SAVE_EPOCH = 1
LOG_DIR = 'log'
CKPT_DIR = 'ckpt'
SAVE_DIR = 'out2'
LEARNING_RATE = 0.005

data_dir = os.path.join('datasets', 'generated')
num_classes = 2
batch_size = 128
num_epochs = 500
input_size = 224
feature_extract = True

data_transforms = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


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


def train_model(model, data, data_loader, criterion, optimizer, device, num_epochs, save_fig=False):
    since = time.time()

    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):

        logging.info('-' * 10)
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                if epoch % 1 == 0:
                    model.eval()
                else:
                    continue

            running_loss = 0.0

            # Iterate over data.
            for batch in tqdm.tqdm(data_loader[phase]):

                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                indices = batch['meta']['index']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                val_loss_history.append(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(CKPT_DIR, f'senet_ckpt_{LEARNING_RATE}.pth'))

                if save_fig and epoch % SAVE_EPOCH == 0:
                    outputs = outputs.cpu().numpy()          

                    for i, (index, output) in enumerate(zip(indices, outputs)):
                        if i > 2:
                            break
                        output = output.reshape(1, -1)
                        pred = data[phase].inv_scale_label(output)
                        pred = pred.reshape(-1, 2)

                        input = data[phase].get_image(index, os.path.join(data_dir, phase))
                        save_image(input, os.path.join('assets', SAVE_DIR, f'out_{index}_{epoch}.png'), pred)

                        if epoch == 0:
                            answ = data[phase].labels_original[index]
                            answ = answ.reshape(-1, 2)
                            save_image(input, os.path.join('assets', SAVE_DIR, f'out_{index}_ans.png'), answ)

    time_elapsed = time.time() - since

    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(feature_extract):
    model = SENet.senet50(num_classes=N_IDENTITY, include_top=True)  # output w/ FC layer (dim: 138=NUM_OUT_FT)
    # model = SENet.senet50(num_classes=N_IDENTITY, include_top=False)  # output w/o FC layer (dim: 2048)
    load_state_dict(model, MODEL_PATH)
    set_parameter_requires_grad(model, feature_extract)
    num_in_ft = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_in_ft, NUM_OUT_FT)

    return model


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, f'senet_{datetime.now().time()}.log')),
            logging.StreamHandler()
        ]
    )

    logging.info('PyTorch Version: ',torch.__version__)
    logging.info('Torchvision Version: ',torchvision.__version__)

    model = initialize_model(feature_extract)

    logging.info(model)
    logging.info('Initializing Datasets and Data_loader...')

    data = {
        'train': StyleGAN2_Data(root=data_dir, split='train', transform=data_transforms),
        'val': StyleGAN2_Data(root=data_dir, split='val', transform=data_transforms)
        }

    data_loader = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    logging.info('Params to learn:')
    if feature_extract:
        params_to_update = []
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad == True:
                params_to_update.append(param)
                logging.info(f'{i}. param.name: {name}')
                # logging.info(f'{i}. param.shape: {param.shape}')
    else:
        params_to_update = model.parameters()
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad == True:
                logging.info(f'{i}: {name}')

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

    # Setup the loss fxn
    criterion = nn.MSELoss()

    # Train and evaluate
    logging.info('Training starts!')
    model, loss_hist = train_model(model, data, data_loader, criterion, optimizer, device, num_epochs=num_epochs, save_fig=True)