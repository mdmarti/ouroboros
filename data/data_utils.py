from torch.utils.data import Dataset,DataLoader
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from utils import deriv_approx_d2y,deriv_approx_dy,spline_approx_signal
from tqdm import tqdm
from model.model_utils import integrate_batched

class aud_neur_ds(Dataset):

    def __init__(self,data):
        self.x = data
        self.dx = deriv_approx_dy(data)
        self.dx2 = deriv_approx_d2y(data)
        

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        x = self.x[idx]
        dx = self.dx[idx]
        dx2 = self.dx2[idx]
        
        x = torch.from_numpy(x).to(torch.float32)
        dx = torch.from_numpy(dx).to(torch.float32)
        dx2 = torch.from_numpy(dx2).to(torch.float32)

        return x,dx,dx2
    
    def _spline_interp_y(self,dt,lam=5.,upsample_prop=32):

        self.x = spline_approx_signal(self.x,dt,lam,to_torch=False,upsample_prop=upsample_prop)
        self.dx = deriv_approx_dy(self.x)
        self.dx2 = deriv_approx_d2y(self.x)

class int_real_ds(Dataset):

    def __init__(self,integrated,real):
        self.int = integrated
        self.real = real
        assert len(self.int) == len(self.real), print(f"Error: integrated data only has {len(self.int)} samples, while real data has {len(self.real)}!!")

    def __len__(self):

        return self.int.shape[0]

    def __getitem__(self, index):
        
        integrated,real = self.int[index],self.real[index]

        integrated = torch.from_numpy(integrated).to(torch.float32)
        real = torch.from_numpy(real).to(torch.float32)

        return integrated,real

def time_stretch(data,true_dt,fake_dt):

    L = len(data)
    T = fake_dt * L
    currTimes = np.arange(0,T + fake_dt/2,fake_dt)
    newTimes = np.arange(0,T + true_dt/2,true_dt)

    interp = np.interp(newTimes,currTimes,data)

    return interp
    

def euler_integrate(y0,dy,dt):

    return np.cumsum(dy*dt,axis=1) + y0

def adjusted_euler_integrate(y0,dy,d2y,dt=1):

    dy_adjusted = dy*dt + 1/2 * d2y * (dt**2)

    return y0 + np.cumsum(dy_adjusted,axis=1)

def get_loaders(data,dt=1/44100,num_workers=4,batch_size=32,\
                train_size=0.8,cv = False,seed=None,\
                interp_y=False,lam=1e-5,upsample_prop=32):

    dls = {}
    
    test_size = 1 - train_size
    val_size= test_size/2
    X_train,X_test = train_test_split(data,test_size=test_size,random_state=seed)


    if cv:
        X_val, X_test = train_test_split(X_test,test_size=0.5,random_state=seed)
        dsVal = aud_neur_ds(X_val)
        if interp_y:
            print('interpolating validation y...')
            dsVal._spline_interp_y(dt,lam=lam,upsample_prop=upsample_prop)
            print('done!!')
        dls['val'] = DataLoader(dsVal,num_workers=num_workers,batch_size=batch_size,shuffle=False)
    dsTrain,dsTest = aud_neur_ds(X_train),aud_neur_ds(X_test)
    if interp_y:
        print('interpolating y for train set...')
        dsTrain._spline_interp_y(dt,lam=lam,upsample_prop=upsample_prop)
        print('interpolating ys for test set....')
        dsTest._spline_interp_y(dt,lam=lam,upsample_prop=upsample_prop)
        print('done!')
    dls['train'] = DataLoader(dsTrain,num_workers=num_workers,batch_size=batch_size,shuffle=True)
    dls['test'] = DataLoader(dsTest,num_workers=num_workers,batch_size=batch_size,shuffle=False)

    return dls


def get_integration_loaders(dataLoaders,model,dt,batched_integration=True,\
                            integration_batch_size=256,
                            num_workers=4,dl_batch_size=32):

    int_loaders = {}

    for key in dataLoaders.keys():

        tmp_data = dataLoaders[key].dataset.x

        int_data = []
        if batched_integration:
            total_data = tmp_data.shape[0]
            for batch_on in tqdm(np.arange(0,total_data,integration_batch_size),desc=f'Integrating samples for {key} set'):
                batch_off = min(batch_on+integration_batch_size,total_data)
                model.trend_filtering=True
                out,*_ = integrate_batched(model,\
                            torch.from_numpy(tmp_data[batch_on:batch_off,:,:]).to(model.device).to(torch.float32),\
                            dt=dt,st=0,method='rk4',int_length=1,options=dict(step_size=dt/4),smooth_len=0.01)
                int_data.append(out.detach().cpu().numpy().squeeze())

        else:
            for sample in tqdm(tmp_data,desc=f'Integrating samples for {key} set',total=len(tmp_data)):

                sample = sample[None,:,:].to(model.device).to(torch.float32)
                integrated_sample,*_ = model.integrate(sample,\
                                                    dt,st=0.)
                int_data.append(integrated_sample[0,:][None,:,None])
        int_data = np.concatenate(int_data,axis=0)

        ds = int_real_ds(int_data[:,:,:1],tmp_data[:,4:-4,:])
        if key == 'train':
            shuffle=True
        else:
            shuffle = False
        int_loaders[key] = DataLoader(ds,num_workers=num_workers,batch_size=dl_batch_size,shuffle=shuffle)

    return int_loaders