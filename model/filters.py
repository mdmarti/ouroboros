import torch
import torch.nn as nn 

def calc_padding_size(filter_size):

    return filter_size//2


class Filter(nn.Module):

    def __init__(self,n_filters=4,filter_size=255,device='cuda'):

        try:
            iter(n_filters)

        except:
            
            n_filters = [n_filters]
       
        super().__init__()    
        self.device=device
        filters = [nn.Conv1d(in_channels=1,out_channels=n_filters[0],kernel_size=filter_size,padding=(filter_size-1)//2)]
        filters += [nn.Conv1d(in_channels=i,out_channels=o,kernel_size=filter_size,padding=(filter_size-1)//2) for i,o in zip(n_filters[:-1],n_filters[1:])]
        filters += [nn.Conv1d(in_channels=n_filters[-1],out_channels=1,kernel_size=filter_size,padding=(filter_size-1)//2)]
        self.filter = nn.Sequential(*filters).to(self.device)
        

    def forward(self,x):

        if len(x.shape) == 2:
            x = x[:,None,:]
        #print(x.shape)
        return self.filter(x)
    

