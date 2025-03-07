from data.real_data import *
from data.data_utils import get_loaders
from utils import sse
from train.train import train,save_model,load_model
from model.constrained_model import rkhs_ouroboros
from model.kernels import *
import matplotlib.pyplot as plt
import gc
import torch
import fire 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization.model_vis import loss_plot
from train.eval import eval_model_error,eval_model_integration

def run_model(audio_path,seg_path='', model_path= '',\
              seg_filetype='.txt',audio_filetype='.wav',voctype='adultsong',\
                context_len=0.3,max_pairs=1000,trend_level=1,
                nEpochs=100, kernel_type='gauss',n_kernels=10,alpha=1e7,seed=None,\
                    save_loaders=False,smooth_len=0.005,vis_freq=0,batch_size=32,
                    smooth_d2y=False,lam=1e-5,tau=1000):

    
    use_trend = True if trend_level > 0 else False
    if seg_path == '':
        seg_path = audio_path

    model_path += '_' + str(seed)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
 
    print('loading data')
    
    audios,sr = get_segmented_audio(audio_path,seg_path,envelope=False,context_len=context_len,\
                                    audio_type=audio_filetype,seg_type=seg_filetype,max_pairs=max_pairs,seed=seed)
    
    loader_path = model_path + '/loaders.pth'
    if os.path.isfile(loader_path):
        dls = torch.load(loader_path,weights_only=False)
    else:
        print(f'getting dataloaders with seed {seed}')
        smooth_str = '' if smooth_d2y else ' NOT'
        print(f'we are{smooth_str} smoothing with lam = {lam}')
        dls = get_loaders(np.vstack(audios),dt=1/sr,cv = True,train_size=0.6,seed=seed,batch_size=batch_size,interp_y=smooth_y,lam=lam)
        if save_loaders:
            print('saving dataloaders...')
            del audios
            dls['sr'] = sr
            torch.save(dls,loader_path)

    #alpha_xaxis = np.arange(len(alphas))
    #n_kernels = [10] #1,2,3

    ## if data dim = 1 and stacking on data dim, z_dim should be 4
    ## if data dim = 1 and stacking on time dim, z_dim should be 2
    if kernel_type == 'gauss':
        kernel = simpleGaussModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation=lambda x: x,trend_filtering=use_trend)
    else:
        kernel = polyModule(nTerms=n_kernels,device='cuda',x_dim=1,z_dim=2,activation = lambda x: x,lam=0.9,trend_filtering=use_trend)
    
    model = rkhs_ouroboros(d_data=1,n_layers=1,d_state=1,\
                d_conv=4,expand_factor=1,tau=tau,\
                            smooth_len=smooth_len,kernel=kernel,
                            trend_filtering=use_trend)
    opt = Adam(model.parameters(),
                lr=1e-3)
    scheduler = ReduceLROnPlateau(opt,factor=0.75,patience=5,min_lr=1e-10)

    model_path_full = model_path + f'/kernelborous_{voctype}_trendfiltering_{trend_level}_alpha_{alpha}_kernel_{kernel_type}_nkernels_{n_kernels}_lam_{str(lam)}'
    save_loc = model_path_full + '/checkpoint_100.tar'

    if os.path.isfile(save_loc):
        #print(f'loading_checkpoint at {save_loc}')
        model,opt,scheduler = load_model(save_loc,kernel_type=kernel_type)
        state = torch.load(save_loc,weights_only=True)
        model.load_state_dict(state['ouroboros'])
        opt.load_state_dict(state['opt'])
    else:

        loss_fn = lambda y,yhat: sse(yhat,y,reduction='mean')
        tl,vl,model,opt = train(model,opt,loss_fn=loss_fn,loaders=dls,scheduler=scheduler,nEpochs=nEpochs,val_freq=1,\
                        runDir=model_path_full,\
                        dt = 1/sr,use_trend_filtering=use_trend,trend_level=trend_level,vis_freq=vis_freq,\
                            alpha=alpha)
        
        loss_plot(tl,vl,save_loc=model_path_full,show=False)
        sd = {'ouroboros':model.state_dict(),
        'opt':opt.state_dict()}
        
        save_model(model,opt,save_loc)
    #model.trend_filtering=False
    (train_mu,test_mu),(train_sd,test_sd) = eval_model_error(dls,model,dt=1/sr)

    (train_coef,test_coef),(train_coef_sd,test_coef_sd) = eval_model_integration(dls,model,dt=1/sr,n_segs=25,st=0)

    
    tl = np.array(tl)
    ax = plt.gca()
    ax.plot(tl[:,0],color='tab:blue',label='train loss')

    vl = np.array(vl)
    ax.plot(vl[:,0],vl[:,1],color='tab:orange',label='val loss')
    ax.set_title("losses")
    plt.legend()
    ax.set_xlabel("gradient updates")
    ax.set_ylabel("loss (MSE)")
    plt.savefig(model_path_full + '/train_losses.svg')
    plt.close()


    ####### add in train/test error analysis here #######
    ####### maybe save out dataloader so that we can 
    ####### maintain same train/test split after training run -- that's a good idea actually, save in model path

    return model, dls

if __name__ == '__main__':

    fire.Fire(run_model)
    print('all done!')