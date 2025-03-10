import torch
from torch.utils.data import Dataset,DataLoader

from utils import deriv_approx_d2y,deriv_approx_dy
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint,odeint_adjoint

class aud_neur_ds(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        sample = self.data[idx]
        x,y = sample[:-1,:],sample[1:,:]

        x,y = torch.from_numpy(x).type(torch.FloatTensor),torch.from_numpy(y).type(torch.FloatTensor)

        return x,y

def save(model,opt,save_loc):
    sd = {'model': model.state_dict(),
      'opt':opt.state_dict()}
    
    torch.save(sd,save_loc)

def load(model,opt,load_loc):

    checkpoint = torch.load(load_loc, weights_only=True)

    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])

    return model,opt

def simulate_step(model,sample,mask_audio=True):

    caches = [(None, torch.zeros((1,model.config.d_model * model.config.expand_factor,model.config.d_conv),device='cuda')) for _ in model.layers]

    sample = sample.squeeze()
    N = sample.shape[0]

    gen_data = []
    
    for ii in range(N):
        s = sample[ii:ii+1]
        
        if mask_audio:
            mask = torch.ones(s.shape,device=s.device)
            mask[:,-1] = 0
            s = s * mask
        dy,caches = model.step(s,caches)
        gen_data.append(s + dy)

    return torch.vstack(gen_data)[None,:,-1]

def generate(model,sample,mask_audio=False):

    caches = [(None, torch.zeros((1,model.config.d_model * model.config.expand_factor,model.config.d_conv),device='cuda')) for _ in model.layers]
    sample = sample.squeeze()

    N = sample.shape[0]

    gen_data = []
    
    for ii in range(N):
        if ii == 0:
            s = sample[ii:ii+1]
        else:
            s = gen_data[ii-1]
        
        if mask_audio:
            mask = torch.ones(s.shape,device=s.device)
            mask[:,-1] = 0
            s = s * mask
        dy,caches = model.step(s,caches)
        gen_data.append(s + dy)
    
    return torch.vstack(gen_data)[None,:,-1]

def smooth(data,smooth_len):

    B,L,D = data.shape
    if smooth_len == 1:
        return data
    pad = torch.zeros((B,smooth_len,D),device='cuda')
    try:
        cumsum = torch.cumsum(torch.cat([pad,data],dim=1),dim=1)
    except:
        print(pad.shape,data.shape)
        assert False
    return (cumsum[:,smooth_len:,:] - cumsum[:,:-smooth_len,:])/float(smooth_len)

def correct_torch(integration,data,center_rounds=2):

    for smooth_round in range(center_rounds):
        smoothed = smooth(integration,smooth_len=5*(smooth_round + 1))
        integration -= smoothed

    model_env = smooth(integration.abs(),smooth_len=20)
    data_env = smooth(data.abs(),smooth_len=20)

    ratio = model_env/data_env 
    return integration / ratio

def integrate_batched(model,x,dt,method='RK45',st=0.05,scaled=True,\
                  correct_signal=True,int_length=0.05,options=dict(),smooth_len=0.):

        if smooth_len==0:
            model.trend_filtering=True
        else:
            model.trend_filtering=False
            model.smooth_len=smooth_len

        smooth_len = int(round(model.smooth_len/dt))
        B,_,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        xddot = deriv_approx_d2y(x)

        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]

        ############## gather functions ##################

        #if L > int(round(1/dt)):
        #    yhat,omega,gamma,weights,weighted_kernels = self.funcs_by_step(z,dt,scaled=scaled)
        #else:
        yhat,*_ = model.forward(x[:1,:,:],dt)

        omega,gamma,weighted_kernels,states = model.get_funcs(x[:1,:,:],dt,scaled=scaled)
        if smooth_len != 0:
            model.trend_filtering=True
        start = int(round(st/dt))
        omega,gamma,weighted_kernels = omega[:,start:],gamma[:,start:],weighted_kernels[:,start:]

        t_steps = torch.arange(0,L*dt+dt/2,dt)[:L][start:]

        z[:,:,-1]/=dt
        z0 = z[:,start,:]

        max_len = (L -1 ) - (start )

        #### define system of ODEs ###############
        def dz(t,z):

            # t: time, should have a timestep of roughly dt. treat as ZOH
            # z: B x 2d
            b_ind = min(int(t/dt),max_len)
            
            omega_step = omega[:,b_ind,:] 
            gamma_step = gamma[:,b_ind,:] 
            weighted_kernels_step = weighted_kernels[:,b_ind,:]
            #print(z.shape)
    
            z1 = z[:,:1]
            
            z2 = z[:,1:] 
    
            dz2 = -(omega_step**2)*z1 - gamma_step * z2 - weighted_kernels_step
            dz1 = z[:,:1]

            return torch.cat([dz1,dz2],dim=-1)

        #### integrate, over small chunk, then correct ############
        int_output = []

        start_times=np.arange(st,L*dt,int_length)
        for t in start_times:
            bounds = (t,min(t+int_length,L*dt))
            bounds_samples = (int(round(bounds[0]/dt)),int(round(bounds[1]/dt)))
            int_length_samples = bounds_samples[1] - bounds_samples[0]
            t_eval = torch.arange(t,t+int_length + dt/2,dt)[:int_length_samples].to(model.device).to(torch.float32)
            yh = odeint_adjoint(dz,z0,t_eval,adjoint_params=(),method=method,options=options).transpose(dim0=0,dim1=1)
            
            if correct_signal:
                corrected_y = correct_torch(yh,z[:,bounds_samples[0]:bounds_samples[1]])
            else:
                corrected_y = yh

            int_output.append(corrected_y)
            z0 = int_output[-1][:,-1,:]
        
        int_output = torch.cat(int_output,dim=1)
        
        return int_output,omega,gamma,weighted_kernels

class NonNegClipper(object):

    def __init__(self,min=0):
        self.min=0

    def __call__(self,module):

        if hasattr(module,'weight'):
            w = module.weight.data
            w.clamp_(min=self.min)
        if hasattr(module,'bias'):
            b = module.bias.data
            b.clamp_(min=self.min)