import os
from PIL import Image
# file dataset

class CustomMask:
    def __init__(self,data_dir,target_dir,transform=None):
        self.data = data_dir
        self.target = target_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data))

    def __getitem__(self,index):
        data = Image.open(os.path.join(self.data,os.listdir(self.data)[index])).convert("RGB")
        target = Image.open(os.path.join(self.target,os.listdir(self.target)[index])).convert("L")
        if self.transform != None:
            data = self.transform(data)
            target = self.transform(target)

        return {
            "data": data,
            "target": target
        }