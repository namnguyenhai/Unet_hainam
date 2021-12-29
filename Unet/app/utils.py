import io
from PIL import Image
from dataset import CustomMask
from torch.utils.data import DataLoader
from model import Unet
import torch
import torchvision
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloader(train_dir,train_target,bathsize,transform):
    data = CustomMask(
        train_dir,
        train_target,
        transform
    )
    train_loader = DataLoader(data,bathsize,shuffle=True,drop_last=True)
    return train_loader

model = Unet(3,1,3).to(DEVICE)
model.load_state_dict(torch.load("app/unet_path.pth",map_location=DEVICE))
def transfrom(image):
    transform = torchvision.transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor()
    ])
    img = Image.open(io.BytesIO(image))
    b = transform(img)
    c = b.unsqueeze(0)
    c = c.to(DEVICE)
    return c
def predict(image):
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(image))
        im = (y_pred>0.5).float()
        tran = transforms.Compose([transforms.ToPILImage()])
        result = tran(im[0])
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        # print(f"img byte la {img_byte_arr}")
    return img_byte_arr