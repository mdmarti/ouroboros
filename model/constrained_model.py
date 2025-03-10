import torch
from torch import nn
from mambapy.mamba import Mamba, MambaConfig
from model.model_utils import smooth, NonNegClipper
from utils import deriv_approx_dy,deriv_approx_d2y
from model.kernels import *

from scipy.integrate import solve_ivp
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from abc import ABC, abstractmethod


class constrained_ouroboros(nn.Module):



    def __init__(self,d_data,
                 d_out=1,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                tau = 1000,
                noInput = False,
                omega_scale = 10000,
                 smooth_len=0.001,
                smooth_penalty=lambda x: torch.var(x,dim=1),
                good_init=False):
        
        super(constrained_ouroboros,self).__init__()

        omegaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        gammaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        dConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)

        self.omegaMamba = Mamba(omegaConfig).to(device)
        self.gammaMamba = Mamba(gammaConfig).to(device)
        self.dMamba = Mamba(dConfig).to(device)
        self.omega_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #unconstrained
        self.gamma_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) # output unconstrained, but weights nonneg
        self.b = nn.Parameter(torch.zeros(1,device=device),requires_grad=True) #non-neg
        self.d_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #non-neg
        if good_init:
            print('initializing with good values')
            self.omega_net.weight.data = torch.zeros((d_data,4*d_data),device=device)
            self.omega_net.bias.data = omega_scale * torch.ones((1,1),device=device)
            self.d_net.weight.data = torch.zeros((d_data,4*d_data),device=device)
            self.d_net.bias.data = torch.zeros((1,1),device=device)
            self.gamma_net.weight.data = torch.zeros((d_data,4*d_data),device=device)
            self.gamma_net.bias.data = torch.zeros((1,1),device=device)
        #torch.nn.init.uniform_(self.omega_net.weight,a=-omega_scale/tau,b=omega_scale/tau)
        #torch.nn.init.uniform_(self.omega_net.bias,a=-omega_scale/tau,b=omega_scale/tau)
        #torch.nn.init.uniform_(self.gamma_net.weight,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.gamma_net.bias,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.b,a=0,b=2/tau)
        #torch.nn.init.uniform_(self.d_net.weight,a=-1/tau**2,b=1/tau**2)
        #torch.nn.init.uniform_(self.d_net.bias,a=-1/tau**2,b=1/tau**2)
        
        self.device = device
        self.clipper = NonNegClipper(min=0)
        self.smooth_penalty = smooth_penalty
        self.smooth_len=smooth_len

        self.tau = tau
        #self.omega=omega
        if noInput:
            print("running with no input")
            
        self.noInput = noInput
        self.d_data = d_data 
        self.names = [rf"$\omega$",rf"$\gamma$",'nonlinear weight','nonlinear term','d','control signals']

    def _clip_weights(self):

        self.gamma_net.apply(self.clipper)
        #self.b.apply(self.clipper)
        #self.b = self.b.clamp(min=0)
        #self.b_net.apply(self.clipper)
        

    def forward(self,x,dt):
        """
        predicts second derivative at time t.
        all other predictions should be done in the train look (train_utils.py)
        """

        B,L,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        omegaControl = self.omegaMamba(x_in)
        gammaControl = self.gammaMamba(x_in)
        dControl = self.dMamba(x_in)

        #dControl=smooth(dControl.abs(),smooth_len)
        #omegaControl=smooth(omegaControl.abs(),smooth_len)
        #gammaControl=smooth(gammaControl,smooth_len)
        b = nn.ReLU()(self.b)/self.tau #.view(B,L,d,self.poly_dim,self.poly_dim) # B x L x 2d x P x P
        d = nn.ReLU()(self.d_net(dControl))
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        omega = self.omega_net(omegaControl)
        gamma = self.gamma_net(gammaControl)/self.tau
        d=smooth(d.abs(),smooth_len)
        omega=smooth(omega.abs(),smooth_len)
        gamma = smooth(gamma,smooth_len)
        #gammaControl=smooth(gammaControl,smooth_len)
        #/self.tau
        
       
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        power_mat_z1 = z[:,:,:1]**2 
        z2 = z[:,:,1:]

        yhat = -(omega**2)*z1 + gamma * z2 - b*power_mat_z1 * z2 - d

        smooth_penalty = self.smooth_penalty(torch.diff(omega,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(gamma,dim=1)[:,1:,:]).mean() + self.smooth_penalty(torch.diff(d,dim=1)[:,1:,:]).mean()
        size_penalty = 0 # (d).abs().mean()
        return yhat,torch.cat([omegaControl,gammaControl,dControl],dim=-1),size_penalty + smooth_penalty/3

    def get_funcs(self,x,dt):

        B,_,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))
        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
    
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        omegaControl = self.omegaMamba(x_in)
        gammaControl = self.gammaMamba(x_in)
        dControl = self.dMamba(x_in)
        #dControl=smooth(dControl.abs(),smooth_len)
        #omegaControl=smooth(omegaControl.abs(),smooth_len)
        #gammaControl=smooth(gammaControl,smooth_len)
        b = nn.ReLU()(self.b)
        d = self.d_net(dControl)
        if self.noInput:
            d = torch.zeros(d.shape,device=self.device)
        omega = self.omega_net(omegaControl)
        gamma = self.gamma_net(gammaControl)
        omega = smooth(omega.abs(),smooth_len)
        gamma = smooth(gamma,smooth_len)
        d = smooth(d.abs(),smooth_len)
        
        omega *= self.tau
        gamma *= self.tau
        d *= self.tau**2
    
        z[:,:,1] /= dt
        z1 = z[:,:,:1]
        z2 = z[:,:,1:]
        power_mat_z1 = z[:,:,:1].pow(2) #expand(-1,-1,-1,model.poly_dim) # B x 2d -> B x 2d x P

        #b = nn.ReLU()(self.b_net(states)) #self.b * torch.ones(power_mat_z1.shape,device='cuda')
        b *= self.tau
        b_out =b * power_mat_z1 * z2
        b = b * torch.ones((B,L,D),device=self.device)
        return omega,gamma,b,b_out,d,torch.cat([omegaControl,gammaControl,dControl],dim=-1)
    
    def visualize(self,x,dt):

        B,L,D = x.shape
        
        with torch.no_grad():
            terms = self.get_funcs(x[:1,:,:],dt)
            terms = [t.detach().cpu().numpy().squeeze() for t in terms]
            
            torch.cuda.empty_cache()

        on = np.random.choice(L-45)
        t_ax = np.arange(on,on+40,1)*dt
        for ii,(t,n) in enumerate(zip(terms,self.names)):
            if ii not in [2,5]:
                ax = plt.gca()
                ax.plot(t_ax,t[on:on+40]/(2*np.pi))
                ax.set_title(n)
                plt.show()
                plt.close()

        return

    
    def integrate(self,x,dt,method='RK45',st=0.05,with_residual=False):

        B,_,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        xdot= deriv_approx_dy(x)
        xddot = deriv_approx_d2y
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        
        z0 = z[:,0,:]
        z0[:,-1] /= dt
        
        omega,gamma,b,_,d,_ = self.get_funcs(x,dt)
        yhat,*_ = self.forward(x,dt)
        residual = xddot - yhat 

        
        start = int(round(st/dt))
        omega,gamma,b,d = omega.detach().cpu().numpy().squeeze()[start:], gamma.detach().cpu().numpy().squeeze()[start:],\
                        b.detach().cpu().numpy().squeeze()[start:],d.detach().cpu().numpy().squeeze()[start:]
        
        t_steps = np.arange(0,L*dt + dt/2,dt)[:L][start:]
        
        omegaTerp = lambda t: np.interp(t,t_steps,omega)
        gammaTerp = lambda t: np.interp(t,t_steps,gamma)
        dTerp = lambda t: np.interp(t,t_steps,d)
        bTerp = lambda t: np.interp(t,t_steps,b)
        def dz(t,z):

            # t: time, timestep ~ dt. treat as variables as ZOH
            # z: B x 2d
            #b_ind = int(t/dt)
            b_step = bTerp(t) #* self.tau**2 #b[0,b_ind,:,:,:]
            omega_step = omegaTerp(t)
            gamma_step = gammaTerp(t)
            d_step = dTerp(t)

            z1 = z[:1]
            power_mat_z1 = z[:1]**2 #
            z2 = z[1:]
            dz2 = -(omega_step**2)*z1 + gamma_step * z2 - b_step*power_mat_z1 * z2 - d_step 
            
            dz1 = z[1]
            
            return np.hstack([dz1,dz2])
        
        s = time.time()
        obj = solve_ivp(dz,(st,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method=method,atol=1e-5)
        #print(f'integrated in {np.round(time.time() - s,2)} s')

        return obj.y,omega,gamma,b,d,obj.status
    
    
class arneodobouros(nn.Module):


    def __init__(self,d_data,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                 tau = 1000,
                 noInput = False,
                 omega_scale = 10000,
                 smooth_len=0.001,
                smooth_penalty=lambda x: torch.var(x,dim=1)):
        
        super(arneodobouros,self).__init__()

        self.device=device
        alphaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        betaConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        dConfig = MambaConfig(d_model=4*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        
        self.alphaMamba = Mamba(alphaConfig).to(device)
        self.betaMamba = Mamba(betaConfig).to(device)
        self.dMamba = Mamba(dConfig).to(device)

        self.alpha_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #unconstrained
        self.beta_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) # output unconstrained, but weights nonneg
        #self.b = nn.Parameter(torch.zeros(1),requires_grad=True).to(device) #non-neg
        self.d_net = nn.Linear(in_features=4*d_data,out_features=d_data,device=device) #non-neg

        tau = nn.Parameter(torch.tensor([tau],dtype=torch.float64,device=device), requires_grad=True)
        self.tau = tau
        self.smooth_len = smooth_len
        self.noInput = noInput
        self.names = [rf"$\alpha$",rf"$\beta$",'d','states']

    def forward(self,x,dt):


        B,L,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        
        alphaControl = self.alphaMamba(x_in)
        betaControl = self.betaMamba(x_in)
        dControl = self.dMamba(x_in)

        #dControl,betaControl = smooth(dControl.abs(),smooth_len),smooth(betaControl.abs(),smooth_len)

        d = self.d_net(dControl)
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        alpha = self.alpha_net(alphaControl)
        beta = self.beta_net(betaControl)
        d, beta = smooth(d.abs(),smooth_len),smooth(beta.abs(),smooth_len)
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        z1_2 = z[:,:,:1]**2 
        z1_3 = z[:,:,:1]**3 
        z2 = z[:,:,1:]

        yhat = alpha + beta.abs()**2 *z1 + z1_2 - z1_3 - z1 * z2/self.tau - z1_2 * z2/self.tau - d 
        return yhat, torch.cat([alphaControl,betaControl,dControl]),torch.zeros((1,),device=self.device)
    
    def get_funcs(self,x,dt):

        B,_,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        x_in = torch.cat([z, torch.flip(z,[1])],dim=-1)
        alphaControl = self.alphaMamba(x_in)
        betaControl = self.betaMamba(x_in)
        dControl = self.dMamba(x_in)
        
        
        #dControl=smooth(dControl.abs(),smooth_len)
        #betaControl=smooth(betaControl.abs(),smooth_len)
        d = self.d_net(dControl)
        if self.noInput:
            d = torch.zeros(d.shape,device='cuda')
        alpha = self.alpha_net(alphaControl)
        beta = self.beta_net(betaControl)
        d, beta = smooth(d.abs(),smooth_len),smooth(beta.abs(),smooth_len)
        alpha *= self.tau**2
        beta *= self.tau
        d *= self.tau**2

        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        z1_2 = z[:,:,:1]**2 
        z1_3 = z[:,:,:1]**3 
        z2 = z[:,:,1:]
        
        return alpha,beta,d,torch.cat([alphaControl,betaControl,dControl],dim=-1)
    
    def visualize(self,x,dt):

        B,L,D = x.shape
        
        with torch.no_grad():
            terms = self.get_funcs(x[:1,:,:],dt)
            terms = [t.detach().cpu().numpy().squeeze() for t in terms]
            #omega,gamma,b,b_out,d= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
            #                    b.detach().cpu().numpy().squeeze(), b_out.detach().cpu().numpy().squeeze(),\
            #                    d.detach().cpu().numpy().squeeze()
            torch.cuda.empty_cache()

        on = np.random.choice(L-45)
        t_ax = np.arange(on,on+40,1)*dt
        for ii,(t,n) in enumerate(zip(terms,self.names)):
            if ii != 3:
                ax = plt.gca()
                ax.plot(t_ax,t[on:on+40]/(2*np.pi))
                ax.set_title(n)
                plt.show()
                plt.close()

        return
    
    def integrate(self,x,dt,method='RK45',st=0.05):

        B,_,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        xdot= deriv_approx_dy(x)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(l-4)dt
        
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]

        alpha,beta,d,states = self.get_funcs(x[:1,:,:],dt)
        alpha,beta,d= alpha.detach().cpu().numpy().squeeze(),beta.detach().cpu().numpy().squeeze(),\
                            d.detach().cpu().numpy().squeeze()
        
        start = int(round(st/dt))
        alpha,beta,d = alpha[start:],beta[start:],d[start:]

        t_steps = np.arange(0,L*dt+dt/2,dt)[:L][start:]
        alphaTerp = lambda t: np.interp(t,t_steps,alpha)
        betaTerp = lambda t: np.interp(t,t_steps,beta)
        dTerp = lambda t: np.interp(t,t_steps,d)
        z0 = z[0,start,:]
        z0[-1] /= dt

        tau = self.tau.detach().cpu().numpy()

        def dz(t,z):

            # t: time, should have a timestep of roughly dt. treat as ZOH
            # z: B x 2d
            b_ind = int(t/dt)
            
            alpha_step = alphaTerp(t) #alpha[b_ind]
            beta_step = betaTerp(t) # beta[b_ind] 
            d_step = dTerp(t) #d[b_ind]
            
            z1 = z[:1]
            
            z1_2 = z[:1]**2  
            z1_3 = z[:1]**3  
            z2 = z[1:] 
    
            dz2 = alpha_step + np.abs(beta_step)**2 *z1 + z1_2 *tau**2 - z1_3*tau**2 - z1 * z2*tau - z1_2 * z2*tau - d_step 

            dz1 = z[1]
            
            return np.hstack([dz1,dz2])
        
        s = time.time()
        obj = solve_ivp(dz,(st,L*dt),z0.squeeze().detach().cpu().numpy(),t_eval=t_steps,method=method,atol=1e-5)
        print(f'integrated in {np.round(time.time() - s,2)} s')
        return obj.y,alpha,beta,d,obj.status

class residual_fit(nn.Module):

    def __init__(self,d_data,
                 kernel,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                 tau = 1000,
                 smooth_len=0.001,
                 trend_filtering=True):
        
        super().__init__()

        self.device=device
        ## if stacking on data dimension, d should be 4* d_data (y,dy,rev y, rev dy)
        ## if stacking on time dimension, d should be 2*d_data (y,dy)
        resid_config = MambaConfig(d_model=2*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        
        self.resid_mamba = Mamba(resid_config).to(device)
        self.resid_net = nn.Linear(in_features=2*d_data,out_features=d_data,device=device)
        self.tau = tau
        self.smooth_len = smooth_len
        self.names = [rf"residual"]
        self.trend_filtering=trend_filtering

    def forward(self,x,dt,use_trend_filtering=False,trend_level=1):
        """
        predicts residual of second derivative at time t.
        all other predictions should be done in the train look (train_utils.py)
        """

        B,L,D = x.shape
        #x = x
        #print(x.dtype)
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        #x_in = torch.cat([z, torch.flip(z,[1])],dim=-1) # stack on data dimension
        x_in = torch.cat([torch.flip(z,[1]),z],dim=1) # stack on time dimension
        
        resid_control = self.resid_mamba_mamba(x_in)[:,L:,:]

        resid = self.resid_net(resid_control)

        if self.trend_filtering:
            resid = resid.abs()

        else:
            resid = smooth(resid.abs(),smooth_len)
           

                #weighted_kernels = weighted_kernels#,smooth_len)/self.tau
        ## should i be modifying z above? before i give it to the kernel?
       

        return -resid,resid_control,torch.tensor([0]).to(self.device)


class rkhs_ouroboros(nn.Module):

    def __init__(self,d_data,
                 kernel,
                 n_layers=2,
                 d_state=16,
                 d_conv=4,
                 expand_factor=1,
                 device='cuda',
                 tau = 1000,
                 smooth_len=0.001,
                 trend_filtering=True):
        
        super().__init__()

        self.device=device
        ## if stacking on data dimension, d should be 4* d_data (y,dy,rev y, rev dy)
        ## if stacking on time dimension, d should be 2*d_data (y,dy)
        omegaConfig = MambaConfig(d_model=2*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        gammaConfig = MambaConfig(d_model=2*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        kernelConfig = MambaConfig(d_model=2*d_data,\
                                    n_layers=n_layers,d_state=d_state,\
                                    d_conv=d_conv,expand_factor=expand_factor)
        
        self.omega_mamba = Mamba(omegaConfig).to(device)
        self.gamma_mamba = Mamba(gammaConfig).to(device)
        self.kernel_mamba = Mamba(kernelConfig).to(device)

        self.omega_net = nn.Linear(in_features=2*d_data,out_features=d_data,device=device) #unconstrained
        self.gamma_net = nn.Linear(in_features=2*d_data,out_features=d_data,device=device) # output unconstrained, but weights nonneg

        ############ should maybe move kernel creation into this?
        ############ maybe make trend level an attribute as well?
        
        self.kernel = kernel
        self.tau = tau
        self.smooth_len = smooth_len
        self.names = [rf"$\omega$",rf"$\gamma$",'weighted kernels','states']
        self.trend_filtering=trend_filtering
        if self.trend_filtering:
            print('we will not be smoothing during training!')

    def forward(self,x,dt,use_trend_filtering=False,trend_level=1):
        """
        predicts second derivative at time t.
        all other predictions should be done in the train look (train_utils.py)
        """

        B,L,D = x.shape
        #x = x
        #print(x.dtype)
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        #x_in = torch.cat([z, torch.flip(z,[1])],dim=-1) # stack on data dimension
        x_in = torch.cat([torch.flip(z,[1]),z],dim=1) # stack on time dimension
        
        omegaControl = self.omega_mamba(x_in)[:,L:,:]
        gammaControl = self.gamma_mamba(x_in)[:,L:,:]
        kernelControl = self.kernel_mamba(x_in)[:,L:,:]

        omega = self.omega_net(omegaControl) # let's try rectifying here new change: un-rectifying
        gamma = self.gamma_net(gammaControl)/self.tau # oops
        weighted_kernels,weights = self.kernel(z,kernelControl,smooth_len)
        weighted_kernels /= self.tau

        if self.trend_filtering:
            omega_diffs,gamma_diffs = torch.diff(omega,dim=1,n=trend_level),torch.diff(gamma,dim=1,n=trend_level)
            #weight_diffs = torch.diff(weights,dim=1,n=trend_level)
            tf = omega_diffs.abs().sum() + gamma_diffs.abs().sum() #+ weight_diffs.abs().sum()
        else:
            omega=smooth(omega.abs(),smooth_len)
            gamma = smooth(gamma,smooth_len)
                #weighted_kernels = weighted_kernels#,smooth_len)/self.tau
        ## should i be modifying z above? before i give it to the kernel?
       
        z[:,:,-1] /= dt
        z1 = z[:,:,:1]
        z2 = z[:,:,1:]

        yhat = -(omega**2)*z1 - gamma * z2 - weighted_kernels
        if self.trend_filtering:
            return yhat,torch.cat([omegaControl,gammaControl,kernelControl]),tf
        else:
            return yhat,torch.cat([omegaControl,gammaControl,kernelControl]),torch.tensor([0]).to(self.device)

    def get_funcs(self,x,dt,scaled = True):

        B,L,D = x.shape
        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len/dt))

        xdot= deriv_approx_dy(x) # this gives: dxdt (currently unitless)
        # dx: dx_4dt,dx_5dt,dx_6dt,..., dx_(L-4)dt
        z = torch.cat([x[:,4:-4,:],xdot],dim=-1)
        L = z.shape[1]
        if L > int(round(1/dt)):
            print('using step by step functions')
            omega,gamma,weights,weighted_kernels = self.funcs_by_step(z,dt,scaled=scaled)
            return omega,gamma,weighted_kernels,weights
        # x_in = torch.cat([z, torch.flip(z,[1])],dim=-1) ## stack on data dimension
        x_in = torch.cat([torch.flip(z,[1]),z],dim=1) ## stack on time dimension
        
        omegaControl = self.omega_mamba(x_in)[:,:,:]
        gammaControl = self.gamma_mamba(x_in)[:,:,:]
        kernelControl = self.kernel_mamba(x_in)[:,:,:]

        omega = self.omega_net(omegaControl).abs()
        gamma = self.gamma_net(gammaControl)
        
        if not self.trend_filtering:
            omega=smooth(omega.abs(),smooth_len)#*self.tau
            gamma = smooth(gamma,smooth_len)#*self.tau

        weighted_kernels,_ = self.kernel(x_in,kernelControl,smooth_len)#*self.tau
        if not self.trend_filtering:
            weighted_kernels = smooth(weighted_kernels,smooth_len)
        if scaled:
            omega,gamma,weighted_kernels = omega*self.tau,gamma*self.tau,weighted_kernels*self.tau
        #weighted_kernels = smooth(weighted_kernels,smooth_len)*self.tau

        return omega[:,L:,:],gamma[:,L:,:],weighted_kernels[:,L:,:],torch.cat([omegaControl[:,L:,:],gammaControl[:,L:,:],kernelControl[:,L:,:]],dim=-1)
    

    def visualize(self,x,dt):

        B,L,D = x.shape
        
        with torch.no_grad():
            terms = self.get_funcs(x[:1,:,:],dt)
            terms = [t.detach().cpu().numpy().squeeze() for t in terms]
            
            torch.cuda.empty_cache()

        on = np.random.choice(L-45)
        t_ax = np.arange(on,on+40,1)*dt
        for ii,(t,n) in enumerate(zip(terms,self.names)):
            if ii not in [3]:
                ax = plt.gca()
                ax.plot(t_ax,t[on:on+40]/(2*np.pi))
                ax.set_title(n)
                #plt.show()
                plt.close()

        return
    
    def integrate(self,x,dt,method='RK45',st=0.05,scaled=True,\
                  with_residual=False,correct=True,int_length=0.05,
                  strategy='interpolate'):

        smooth_len = int(round(self.smooth_len/dt))
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
        yhat,*_ = self.forward(x[:1,:,:],dt)
        omega,gamma,weighted_kernels,states = self.get_funcs(x[:1,:,:],dt,scaled=scaled)
        omega,gamma,weighted_kernels= omega.detach().cpu().numpy().squeeze(),gamma.detach().cpu().numpy().squeeze(),\
                        weighted_kernels.detach().cpu().numpy().squeeze()
    
        residual = xddot - yhat 
        smoothed_residual = smooth(residual,smooth_len).detach().cpu().numpy().squeeze()

        start = int(round(st/dt))
        omega,gamma,weighted_kernels,smoothed_residual = omega[start:],gamma[start:],weighted_kernels[start:],smoothed_residual[start:]

        t_steps = np.arange(0,L*dt+dt/2,dt)[:L][start:]
        omegaTerp = lambda t: np.interp(t,t_steps,omega)
        gammaTerp = lambda t: np.interp(t,t_steps,gamma)
        weighted_kernelsTerp = lambda t: np.interp(t,t_steps,weighted_kernels)
        if with_residual:
            residTerp = lambda t: np.interp(t,t_steps,smoothed_residual)
        z[:,:,-1]/=dt
        z0 = z[0,start,:]

        max_len = (L -1 ) - (start )

        #### define system of ODEs ###############
        def dz(t,z):

            # t: time, should have a timestep of roughly dt. treat as ZOH
            # z: B x 2d
            b_ind = min(int(t/dt),max_len)
            
            if strategy == 'interpolate':
                omega_step = omegaTerp(t) 
                gamma_step = gammaTerp(t) 
                weighted_kernels_step = weighted_kernelsTerp(t) 
            else:
                omega_step = omegaTerp(t) 
                gamma_step = gammaTerp(t) 
                weighted_kernels_step = weighted_kernelsTerp(t) 

            z1 = z[:1]
            
            z2 = z[1:] 
    
            dz2 = -(omega_step**2)*z1 - gamma_step * z2 - weighted_kernels_step
            dz1 = z[1]
            if with_residual:
                resids = residTerp(t)
                dz2 += resids
            
            return np.hstack([dz1,dz2])
        
        z0 = z0.detach().cpu().numpy().squeeze()
        z = z.detach().cpu().numpy().squeeze().T

        #### integrate, over small chunk, then correct ############
        int_output = []
        for t in np.arange(st,L*dt,int_length):
            bounds = (t,min(t+int_length,L*dt))
            bounds_samples = (int(round(bounds[0]/dt)),int(round(bounds[1]/dt)))
            int_length_samples = bounds_samples[1] - bounds_samples[0]
            t_eval = np.arange(t,t+int_length + dt/2,dt)[:int_length_samples]
            obj = solve_ivp(dz,bounds,z0,t_eval=t_eval,method=method,atol=1e-8)
            
            if correct:
                corrected_y = correct(obj.y,z[:,bounds_samples[0]:bounds_samples[1]])
            else:
                corrected_y = obj.y

            int_output.append(corrected_y)
            z0 = int_output[-1][:,-1]
        
        int_output = np.hstack(int_output)
        
        if with_residual:
            return int_output,omega,gamma,weighted_kernels,smoothed_residual,obj.status
        return int_output,omega,gamma,weighted_kernels,obj.status
    
    def funcs_by_step(self,z,dt,scaled=True):

        smooth_len = int(round(self.smooth_len/dt))

        omega_cache = [(None, torch.zeros((1,self.omega_mamba.config.d_model * self.omega_mamba.config.expand_factor,\
                                           self.omega_mamba.config.d_conv),device='cuda')) for _ in self.omega_mamba.layers]
        gamma_cache = [(None, torch.zeros((1,self.gamma_mamba.config.d_model * self.gamma_mamba.config.expand_factor,\
                                           self.gamma_mamba.config.d_conv),device='cuda')) for _ in self.gamma_mamba.layers]
        weights_cache = [(None, torch.zeros((1,self.kernel_mamba.config.d_model * self.kernel_mamba.config.expand_factor,\
                                             self.kernel_mamba.config.d_conv),device='cuda')) for _ in self.kernel_mamba.layers]
        
        x_in = torch.cat([torch.flip(z,[1]),z],dim=1)
        B,L,D = x_in.shape
        omega,gamma,weights,kernel= [],[],[],[]
        
        for ii in range(L):
            s = x_in[:,ii,:]

            omega,omega_cache = self.omega_mamba.step(s,omega_cache)
            gamma,gamma_cache = self.omega_mamba.step(s,gamma_cache)
            weights,weights_cache = self.omega_mamba.step(s,weights_cache)

            omega = self.omega_net(omega).abs()
            gamma = self.gamma_net(gamma)
            
            s = z[:,ii:ii+1,:]
            weights.append(weights.detach().cpu().numpy())
            omega.append(omega.detach().cpu().numpy())
            gamma.append(gamma.detach().cpu().numpy())
            weighted_kernels,_ = self.kernel(s,weights[:,None,:],smooth_len)
            kernel.append(weighted_kernels.detach().cpu().numpy())
            

        z[:,:,1]/=dt 
        omega= np.stack(omega,axis=1)
        gamma = np.stack(gamma,axis=1)
        weights = np.stack(weights,dim=1)
        kernel = np.stack(kernel,dim=1)

        z1 = z[:,:,:1]/dt
        z2 = z[:,:,1:]

        yhat = -(omega**2)*z1 - gamma * z2 - weighted_kernels

        if not self.trend_filtering:
            omega=smooth(omega,smooth_len)#*self.tau
            gamma = smooth(gamma,smooth_len)#*self.tau
        
        if scaled:
            omega *= self.tau
            gamma *= self.tau 
            kernel *= self.tau 
            weights *= self.tau
        return yhat,omega,gamma,weights,kernel

