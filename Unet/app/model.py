import torch
import torch.nn as nn
import torch.nn.functional as f

class DoubleConv(nn.Module):
    def __init__(self,input_channel,output_channel,kennerl_size):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kennerl_size,1,1,bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,kennerl_size,1,1,bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        output = self.conv(x)
        return output
        
class Unet(nn.Module):
    def __init__(self,input_channel,output_channel,kennerl_size,
                 num_channel=[64,128,256,512]):
        super(Unet,self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        for channel in num_channel:
            self.down.append(DoubleConv(input_channel,channel,kennerl_size))
            input_channel = channel
        for channel in reversed(num_channel):
            self.up.append(nn.ConvTranspose2d(channel*2,channel,kernel_size=2,stride=2))
            self.up.append(DoubleConv(channel*2,channel,kennerl_size))
        self.bottom = DoubleConv(num_channel[-1],num_channel[-1]*2,kennerl_size)
        self.end  = nn.Conv2d(num_channel[0],output_channel,kernel_size=1)


    def forward(self,x):
        skip_connect =  []
        for down in self.down:
            x = down(x)
            skip_connect.append(x)
            x = self.pool(x)
            
        x = self.bottom(x)

        skip_connect = skip_connect[::-1]
        
        for up in range(0,len(self.up),2):
            x = self.up[up](x)
            skip = skip_connect[up//2]
            if x.shape != skip.shape:
#                 skip = F.resize(skip,size=x.shape[2:])
                column = skip.size(-1) - x.size(-1)
                row = skip.size(-2) - x.size(-2)
                x = f.pad(x,(column-(column//2),column//2,row-(row//2),row//2),"replicate")
                column = 0 
                row = 0
            concat = torch.cat((skip,x),dim=1)
            x = self.up[up+1](concat)

            
        x = self.end(x)
        return x