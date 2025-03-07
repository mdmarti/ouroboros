from torch.utils.data import Dataset,DataLoader
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from utils import deriv_approx_d2y,deriv_approx_dy,spline_approx_signal

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
        
        x = torch.from_numpy(x).type(torch.float32)
        dx = torch.from_numpy(dx).type(torch.float32)
        dx2 = torch.from_numpy(dx2).type(torch.float32)

        return x,dx,dx2
    
    def _spline_interp_y(self,dt,lam=5.,upsample_prop=32):

        self.x = spline_approx_signal(self.x,dt,lam,to_torch=False,upsample_prop=upsample_prop)
        self.dx = deriv_approx_dy(self.x)
        self.dx2 = deriv_approx_d2y(self.x)

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
                interp_y=False,lam=1e-5):

    dls = {}
    
    test_size = 1 - train_size
    val_size= test_size/2
    X_train,X_test = train_test_split(data,test_size=test_size,random_state=seed)


    if cv:
        X_val, X_test = train_test_split(X_test,test_size=0.5,random_state=seed)
        dsVal = aud_neur_ds(X_val)
        if interp_y:
            print('interpolating validation 2nd derivs...')
            dsVal._spline_interp_y(dt,lam=lam)
            print('done!!')
        dls['val'] = DataLoader(dsVal,num_workers=num_workers,batch_size=batch_size,shuffle=False)
    dsTrain,dsTest = aud_neur_ds(X_train),aud_neur_ds(X_test)
    if interp_y:
        print('interpolating second derivs for train set...')
        dsTrain._spline_interp_y(dt,lam=lam)
        print('interpolating second derivs for test set....')
        dsTest._spline_interp_y(dt,lam=lam)
        print('done!')
    dls['train'] = DataLoader(dsTrain,num_workers=num_workers,batch_size=batch_size,shuffle=True)
    dls['test'] = DataLoader(dsTest,num_workers=num_workers,batch_size=batch_size,shuffle=False)

    return dls