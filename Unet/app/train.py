from PIL.Image import DECODERS
from model import Unet
from utils import get_dataloader
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Desktop/data/mask/train"
TRAIN_TARGET = "Desktop/data/mask/train_masks"
NUM_EPOCH = 2
NUM_CHANNEL_INPUT = 1
NUM_CHANNEL_OUTPUT = 1
KERNEL_SIZE = 3
MODEL = Unet(NUM_CHANNEL_INPUT,NUM_CHANNEL_OUTPUT,KERNEL_SIZE).to(DEVICE)


#file train
def train_fn(train_dir,train_target,num_epoch,model_pred,loss,optimizer,batchsize,transform):
    data_train = get_dataloader(train_dir,train_target,batchsize,transform)
    for i in range(num_epoch):
        for data in data_train:
            img = data['data'].to(DEVICE)
            target = data['target'].to(DEVICE)
            y_pred = model_pred(img)
            loss_pred = loss(y_pred,target)
            loss_pred.backward()
            optimizer.step()
            optimizer.zero_grad()