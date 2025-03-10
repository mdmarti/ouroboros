from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import deriv_approx_d2y,deriv_approx_dy,sst,sse,euler_step_k,spline_approx_signal
import matplotlib.pyplot as plt
import os
from model.constrained_model import rkhs_ouroboros
from model.filters import filter as filt
from model.kernels import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def save_filter(trained_filt,location,n_filters,filter_size):

    sd = {'filter_params':trained_filt.state_dict(),
            'n_filters':n_filters,
            'filter_size':filter_size
            }
    torch.save(sd,location)
    return

def load_filter(location):
    sd = torch.load(location,weights_only=False)
    trained_filt = filt(n_filters=sd['n_filters'],filter_size=sd['filter_size'])
    trained_filt.load_state_dict(sd['filter_params'])

    return trained_filt



def save_model(model,opt,location):
    sd = {'ouroboros':model.state_dict(),
        'opt':opt.state_dict(),
        'tau':model.tau,
        'smooth_len':model.smooth_len,
        'n_kernel':model.kernel.nTerms
    }
        
    torch.save(sd,location)

def load_model(location,kernel_type='gauss'):

    sd = torch.load(location,weights_only=False)
    if kernel_type == 'gauss':
        kernel = simpleGaussModule(nTerms=sd['n_kernel'],device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=1)
    else:
        kernel = polyModule(nTerms=sd['n_kernel'],device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=0.9,trend_filtering=1)
    

    model = rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                d_conv=4,expand_factor=1,tau=1000,\
                            smooth_len=sd['smooth_len'],kernel=kernel,
                            trend_filtering=1)
    opt = Adam(model.parameters(),
                lr=1e-3)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)
    model.load_state_dict(sd['ouroboros'])
    opt.load_state_dict(sd['opt'])
    return model,opt,scheduler

def train_filter(model,optimizer,loss_fn,loaders,scheduler=None,\
                 nEpochs=100,val_freq=25,runDir='.',vis_freq=100):
    
    print('training filter')
    


def train(model,optimizer,loss_fn,loaders,filter=None,scheduler=None,
          nEpochs=100,val_freq=25,mask_prob_aud = 0.1,
          init_len = 200,runDir='.',dt=1/44100,vis_freq=100,
          use_trend_filtering=False,
          trend_level=1,alpha=1):
    
    print(f'training with trend filtering alpha = {alpha}')

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,dy,dy2 = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            dy = dy.to('cuda').to(torch.float32)
            dy2 = dy2.to('cuda').to(torch.float32)/(dt**2)

            #dy = deriv_approx_dy(x)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #change: scaling to "true" d2y
            #dy2 = deriv_approx_d2y(x)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            #dy2 = spline_approx_signal(dy2,dt)/(dt**2)
            y2hat,state_pred,trend_penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
            # change: scaling to "true" d2y
            y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2
            
            #y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3
            
            #yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes y[6:-2]
            
            yhat = y2hat #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
            #print(yhat.shape)
            # y starts as x[1:0]
            y = dy2 #torch.cat([x[:,5:-5],dy[:,2:],dy2[:,2:]],dim=-1)
            L = y.shape[1]
            #print(dy2.shape)
            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    sse_sample = sse(yhat[:1,:,:1],y[:1,:,:1])
                    sst_sample = sst(y[:1,:,:1])
                    r2_sample = (1 - sse_sample/sst_sample).item()
                    model.visualize(x,dt)
                
                    on = np.random.choice(L-45)
                    ax = plt.gca()
                    ax.plot(yhat[0,:,0].detach().cpu().numpy(),label='model')
                    ax.plot(y[0,:,0].detach().cpu().numpy(),label='data')
                    ax.set_title(f"sample r2: {r2_sample: 0.4f}")
                    ax.legend()
                    plt.savefig(os.path.join(runDir,f"y_vs_yhat_batch_{idx}.svg"))
                    plt.close()

            """    
            ### trend filtering penalty
            diff = torch.diff(state_pred,dim=1)
            penalty = torch.abs(diff).sum(dim=-1).mean()
            """
            
            """
            ### sparsity penalty
            alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
            norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
            norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
            penalty = alpha*norm1.mean()
            """

            """
            ### covariance penalty
            cov = state_pred.transpose(-1,-2) @ state_pred /L 
            sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
            denom = sds @ sds.transpose(-1,-2)
            abscorr = (cov/denom).abs()
    
            inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
            penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
            """

            ##################################
            
            loss = loss_fn(y,yhat[:,:L,:]) 
            alpha = max(0,min(alpha,alpha*(idx-10*len(loaders['train']))/5000)) if use_trend_filtering else 0
            l = loss + alpha*trend_penalty
            #print(l)
            l.backward()
            optimizer.step()
            tot = sst(y)
            train_losses.append((1 - loss.item()/tot.item(),trend_penalty.item()))
            writer.add_scalar('Loss/train',1 - loss.item()/tot.item(),idx)
            writer.add_scalar('Penalty/train',trend_penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            vn = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,dy,dy2 = batch
                    x = x.to('cuda').to(torch.float32)
                    dy = dy.to('cuda').to(torch.float32)
                    dy2 = dy2.to('cuda').to(torch.float32)/(dt**2)
                    
                    #dy = deriv_approx_dy(y)
                    # dy_4dt, dy_3dt, ...., dy_(L-4)dt
                    #scaling to "true" d2y
                    #dy2 = deriv_approx_d2y(y)/(dt**2)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
                    
                    y2hat,state_pred,penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
                    ## scaling to "true" d2y
                    y2hat = y2hat * model.tau **2 #(model.tau*dt)**2
                    
                    #y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3
                    
                    #yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes y[6:-2]
                    
                    yhat = y2hat #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
                    #print(yhat.shape)
                    # y starts as x[1:0]
                    y = dy2 #torch.cat([x[:,6:-4],dy[:,2:],dy2[:,2:]],dim=-1)
                    L = y.shape[1]
                    if vis_freq > 0:
                        if idx == epoch*len(loaders['train']):
                            sse_sample = sse(yhat[:1,:,:1],y[:1,:,:1])
                            sst_sample = sst(y[:1,:,:1])
                            r2_sample = (1 - sse_sample/sst_sample).item()
                            model.visualize(x,dt)
                        
                            on = np.random.choice(L-45)
                            ax = plt.gca()
                            ax.plot(yhat[0,on:on+600,0].detach().cpu().numpy(),label='model')
                            ax.plot(y[0,on:on+600,0].detach().cpu().numpy(),label='data')
                            ax.set_title(f"sample r2: {r2_sample: 0.4f}")
                            ax.legend()
                            plt.savefig(os.path.join(runDir,f"y_vs_yhat_batch_{idx}_test.svg"))
                            plt.close()
                    
                    l = loss_fn(y,yhat[:,:L,:])
                    tot = sst(y)

                    vl += 1 - l.item()/tot.item()

                    """
                    ### trend filtering penalty
                    diff = torch.diff(state_pred,dim=1)
                    penalty = torch.abs(diff).sum(dim=-1).mean()
                    """

                    """
                    #### norm penalty
                    
                    alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
                    norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
                    norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
                    penalty = alpha * norm1.mean()
                    #vp += alpha*penalty.item()
                    """ 

                    """
                    #### cov penalty

                    cov = state_pred.transpose(-1,-2) @ state_pred /L 
                    sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
                    denom = sds @ sds.transpose(-1,-2)
                    abscorr = (cov/denom).abs()
            
                    inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
                    penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
                    """
                    
                    vp += penalty.item()
                    
                    
            if scheduler:
                scheduler.step(vl/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val']),vp/len(loaders['val'])))
            writer.add_scalar('Loss/validation',vl/len(loaders['val']),idx)
            writer.add_scalar('Penalty/validation',vp/len(loaders['val']),idx)

    writer.close()
    return train_losses,val_losses,model,optimizer

def train_ksteps(model,filter,optimizer,loaders,scheduler=None,
          nEpochs=100,val_freq=25,runDir='.',dt=1/44100,vis_freq=100,
          use_trend_filtering=False,
          trend_level=1,alpha=1,ksteps=1):
    
    print(f'training with trend filtering alpha = {alpha}')

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]

    loss_fn = lambda y,yhat: sse(yhat,y,reduction='none')
    loss_scale = torch.FloatTensor([1,dt,dt**2])[None,:].to('cuda')
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            y = y.to('cuda').to(torch.float32)

            dy = deriv_approx_dy(x)/dt
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #change: scaling to "true" d2y
            dy2 = deriv_approx_d2y(x)/(dt**2)
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt 
            y = x[:,4:-4,:]           
            
            y2hat,state_pred,trend_penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
            # change: scaling to "true" d2y
            y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2, corresponding to x[4:-4]
            
            (y_target,dy_target),(yhat,dyhat) = euler_step_k(y,dy,y2hat,dt,k=ksteps) #dy + y2hat*dt # makes dy [5:-3], corresponding to x[5:-3]
            #print(yhat.shape)
            yhat = torch.vmap(filter,in_dims=-1,out_dims=-1)(yhat)
            #torch.vmap(filter,in_dims=-1,out_dims=-1)(yhat)
            #yhat = x[:,5:-3] + y1hat *dt # makes corresponding to x[6:-2]
            
            #yhat = torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
            #print(yhat.shape)
            # y starts as x[1:0]
            #y = torch.cat([x[:,6:-4],dy[:,2:],dy2[:,2:]],dim=-1)
            L = y.shape[1]

            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    sse_sample = sse(yhat[:1,:,:1],y_target[:1,:,:1])
                    sst_sample = sst(y_target[:1,:,:1])
                    r2_sample = (1 - sse_sample/sst_sample).item()
                    model.visualize(x,dt)
                
                    on = np.random.choice(L-45)
                    ax = plt.gca()
                    ax.plot(yhat[0,:,0].detach().cpu().numpy(),label='model')
                    ax.plot(y_target[0,:,0].detach().cpu().numpy(),label='data')
                    ax.set_title(f"sample r2: {r2_sample: 0.4f}")
                    ax.legend()
                    plt.savefig(os.path.join(runDir,f"y_vs_yhat_batch_{idx}.svg"))
                    plt.close()

            ##################################
            y_loss = (loss_fn(y_target,yhat).sum(dim=-1)/ksteps).mean()
            dy_loss = (loss_fn(dy_target,dyhat).sum(dim=-1)*dt/ksteps).mean()
            d2y_loss = (loss_fn(dy2,y2hat).sum(dim=-1)*dt**2).mean()
            loss = (y_loss +dy_loss + d2y_loss) 
            alpha = 0# max(0,min(alpha,alpha*(idx-10*len(loaders['train']))/5000)) if use_trend_filtering else 0
            l = loss + alpha*trend_penalty
            #print(l)
            l.backward()
            optimizer.step()
            tot1 = (sst(y_target,reduction='none').sum(dim=-1)/ksteps).mean()
            tot2 = (sst(dy_target,reduction='none').sum(dim=-1)*dt/ksteps).mean()
            tot3 = (sst(dy2,reduction='none').sum(dim=-1)*dt**2).mean()
            full_tot = (tot1 + tot2 + tot3)
            train_losses.append((1 - loss.item()/full_tot.item(),trend_penalty.item()))
            writer.add_scalar('Loss/train',1 - loss.item()/full_tot.item(),idx)
            writer.add_scalar('Loss/train, y',1 - y_loss.item()/tot1.item(),idx)
            writer.add_scalar('Loss/train, dy',1 - dy_loss.item()/tot2.item(),idx)
            writer.add_scalar('Loss/train, d2y',1 - d2y_loss.item()/tot3.item(),idx)
            #writer.add_scalar('Penalty/train',trend_penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl=0.
            vly = 0.
            vldy = 0.
            vld2y = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x = x.to('cuda').to(torch.float32)
                    y = y.to('cuda').to(torch.float32)
                    
                    dy = deriv_approx_dy(y)/dt
                    # dy_4dt, dy_3dt, ...., dy_(L-4)dt
                    #scaling to "true" d2y
                    dy2 = deriv_approx_d2y(y)/(dt**2)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt  
                    y = x[:,4:-4,:]          
                    
                    y2hat,state_pred,penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
                    ## scaling to "true" d2y
                    y2hat = y2hat * model.tau **2 #(model.tau*dt)**2
                    
                    (y_target,dy_target),(yhat,dyhat) = euler_step_k(y,dy,y2hat,dt,k=ksteps)
                    yhat = torch.vmap(filter,in_dims=-1,out_dims=-1)(yhat)

                    #yhat = torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
                    #print(yhat.shape)
                    # y starts as x[1:0]
                    #y = torch.cat([x[:,6:-4],dy[:,2:],dy2[:,2:]],dim=-1)
                    L = y.shape[1]
                    if vis_freq > 0:
                        if idx == epoch*len(loaders['train']):
                            sse_sample = sse(yhat[:1,:,:1],y_target[:1,:,:1])
                            sst_sample = sst(y_target[:1,:,:1])
                            r2_sample = (1 - sse_sample/sst_sample).item()
                            model.visualize(x,dt)
                        
                            on = np.random.choice(L-45)
                            ax = plt.gca()
                            ax.plot(yhat[0,on:on+600,0].detach().cpu().numpy(),label='model')
                            ax.plot(y_target[0,on:on+600,0].detach().cpu().numpy(),label='data')
                            ax.set_title(f"sample r2: {r2_sample: 0.4f}")
                            ax.legend()
                            plt.savefig(os.path.join(runDir,f"y_vs_yhat_batch_{idx}_test.svg"))
                            plt.close()
                    y_loss = (loss_fn(y_target,yhat).sum(dim=-1)/ksteps).mean()
                    dy_loss = (loss_fn(dy_target,dyhat).sum(dim=-1)*dt/ksteps).mean()
                    d2y_loss = (loss_fn(dy2,y2hat).sum(dim=-1)*dt**2).mean()
                    l = (y_loss +dy_loss + d2y_loss) 
                    #l =  (loss_fn(y,yhat[:,:L,:])*loss_scale).sum(dim=-1).mean() 
                    tot1 = (sst(y_target,reduction='none').sum(dim=-1)/ksteps).mean()
                    tot2 = (sst(dy_target,reduction='none').sum(dim=-1)*dt/ksteps).mean()
                    tot3 = (sst(dy2,reduction='none').sum(dim=-1)*dt**2).mean()
                    full_tot = (tot1 + tot2 + tot3)

                    vl += 1 - l.item()/full_tot.item()
                    
                    vly += 1 - y_loss.item()/tot1.item()
                    vldy += 1 - dy_loss.item()/tot2.item()
                    vld2y += 1 - d2y_loss.item()/tot3.item()
                    
                    
            if scheduler:
                scheduler.step(vl/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val']),0))
            writer.add_scalar('Loss/validation',vl/len(loaders['val']),idx)
            writer.add_scalar('Loss/validation, y',1 - y_loss.item()/tot1.item(),idx)
            writer.add_scalar('Loss/validation, dy',1 - dy_loss.item()/tot2.item(),idx)
            writer.add_scalar('Loss/validation, d2y',1 - d2y_loss.item()/tot3.item(),idx)
            #writer.add_scalar('Penalty/validation',vp/len(loaders['val']),idx)

    writer.close()
    return train_losses,val_losses,model,optimizer


def train_filter(model,filter,optimizer,loss_fn,loaders,scheduler=None,
          nEpochs=100,val_freq=25,
          runDir='.',dt=1/44100,vis_freq=100,
          use_trend_filtering=False,
          trend_level=1,alpha=1):
    
    print(f'training with trend filtering alpha = {alpha}')

    writer = SummaryWriter(log_dir=runDir)

    train_losses,val_losses=[],[]
    
    for epoch in tqdm(range(nEpochs),desc='training model'):

        model.train()

        for idx,batch in enumerate(loaders['train'],start=epoch*len(loaders['train'])):

            optimizer.zero_grad()
            x,y = batch # each is bsz x seq len x n neurons + 1
            bsz,_,n = x.shape

            x = x.to('cuda').to(torch.float32)
            y = y.to('cuda').to(torch.float32)

            dy = deriv_approx_dy(x)
            # dy_4dt, dy_3dt, ...., dy_(L-4)dt
            #change: scaling to "true" d2y
            dy2 = deriv_approx_d2y(x)/(dt**2) # makes dy2 from x[4:-4] or y[3:-5]
            # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
            
            y2hat,state_pred,trend_penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
            # change: scaling to "true" d2y
            y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2 #y2hat from x[4:-4] (or y[3:-5])
            
            y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3, y1hat from x[5:-3] or y[4:-4]
            
            yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes x[6:-2] or y[5:-3]
            filtered_yhat = filter(yhat)

            ######### potentially add some shrinkage on filter #################

            ####################################################################
            
            yhat = torch.cat([filtered_yhat[:,:-2,:],y2hat[:,2:,:]],dim=-1) #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
            #print(yhat.shape)
            # y starts as x[1:0]
            y = torch.cat([y[:,5:-5,:],dy2[:,2:,:]],dim=-1) #dy2 #torch.cat([x[:,5:-5],dy[:,2:],dy2[:,2:]],dim=-1)
            assert yhat.shape == y.shape,print(yhat.shape,y.shape)
            L = y.shape[1]

            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    model.visualize(x,dt)
                
                    on = np.random.choice(L-45)
                    ax = plt.gca()
                    ax.plot(yhat[0,on:on+40,0].detach().cpu().numpy())
                    ax.plot(y[0,on:on+40,0].detach().cpu().numpy())
                    ax.set_title("data (orange) vs model (blue)")
                    plt.show()
                    plt.close()

            ##################################
            
            loss = loss_fn(y,yhat[:,:L,:]) 
            sse = ((y - y.mean(dim=1,keepdim=True))**2).mean()
            alpha = max(0,min(alpha,alpha*(idx-10*len(loaders['train']))/5000)) if use_trend_filtering else 0
            l = loss + alpha*trend_penalty
            #print(l)
            l.backward()
            optimizer.step()
            train_losses.append((loss.item()/sse.item(),trend_penalty.item()))
            writer.add_scalar('Loss/train',loss.item(),idx)
            writer.add_scalar('Penalty/train',trend_penalty.item(),idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.
            vp = 0.
            vn = 0.
            for idx,batch in enumerate(loaders['val'],start=epoch*len(loaders['train'])):
                with torch.no_grad():
                    x,y = batch
                    x = x.to('cuda').to(torch.float32)
                    y = y.to('cuda').to(torch.float32)
                    
                    dy = deriv_approx_dy(y)
                    # dy_4dt, dy_3dt, ...., dy_(L-4)dt
                    #scaling to "true" d2y
                    dy2 = deriv_approx_d2y(y)/(dt**2)
                    # d2y_4dt, d2y_5dt, ..., d2y_(L-4)dt            
                    
                    y2hat,state_pred,penalty = model(x,dt,use_trend_filtering=use_trend_filtering,trend_level=trend_level) #state: B x L x SD
            
                    ## scaling to "true" d2y
                    # change: scaling to "true" d2y
                    y2hat = y2hat * model.tau**2 #* (model.tau*dt)**2 #y2hat from x[4:-4] (or y[3:-5])
                    
                    y1hat = dy + y2hat/(dt*model.tau) # makes dy 5:-3, y1hat from x[5:-3] or y[4:-4]
                    
                    yhat = x[:,5:-3] + y1hat * (dt*model.tau) # makes x[6:-2] or y[5:-3]
                    filtered_yhat = filter(yhat)

                    ######### potentially add some shrinkage on filter #################

                    ####################################################################
                    
                    yhat = torch.cat([filtered_yhat[:,:-2,:],y2hat[:,2:,:]],dim=-1) #torch.cat([yhat[:,:-2],y1hat[:,1:-1],y2hat[:,2:]],dim=-1) #points 6:-4
                    #print(yhat.shape)
                    # y starts as x[1:0]
                    y = torch.cat([y[:,5:-5,:],dy2[:,2:,:]],dim=-1) #dy2 #torch.cat([x[:,5:-5],dy[:,2:],dy2[:,2:]],dim=-1
                    
                    L = y.shape[1]
                    l = loss_fn(y,yhat[:,:L,:])
                    sse = ((y - y.mean(dim=1,keepdim=True))**2).mean()
                    vl += l.item()/sse.item()

                    """
                    ### trend filtering penalty
                    diff = torch.diff(state_pred,dim=1)
                    penalty = torch.abs(diff).sum(dim=-1).mean()
                    """

                    """
                    #### norm penalty
                    
                    alpha = 1/(state_pred.shape[-1] * state_pred.shape[-2])
                    norm2 = torch.linalg.vector_norm(state_pred,ord=0.5,dim=-1)
                    norm1 = torch.linalg.vector_norm(norm2,ord=1,dim=-1)
                    penalty = alpha * norm1.mean()
                    #vp += alpha*penalty.item()
                    """ 

                    """
                    #### cov penalty

                    cov = state_pred.transpose(-1,-2) @ state_pred /L 
                    sds = torch.diagonal(cov,dim1=-1,dim2=-2).sqrt()[:,:,None]
                    denom = sds @ sds.transpose(-1,-2)
                    abscorr = (cov/denom).abs()
            
                    inds = torch.triu_indices(abscorr.shape[-1],abscorr.shape[-1],offset=-1,device=abscorr.device)
                    penalty = abscorr[:,inds[0],inds[1]].sum(dim=-1).mean()
                    """
                    
                    vp += penalty.item()
                    
                    
            if scheduler:
                scheduler.step(vl/len(loaders['val']))
            val_losses.append((epoch*len(loaders['train']),vl/len(loaders['val']),vp/len(loaders['val'])))
            writer.add_scalar('Loss/validation',vl/len(loaders['val']),idx)
            writer.add_scalar('Penalty/validation',vp/len(loaders['val']),idx)

    writer.close()
    return train_losses,val_losses,model,optimizer
