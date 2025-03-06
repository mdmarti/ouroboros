import torch
import numpy as np

from scipy.signal.windows import hann
from scipy.signal import ShortTimeFFT as STFFT
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

def deriv_approx_dy(y):

    B,L,d = y.shape
    assert L >= 5, print(f"y is not long enough to approximate with 9 points! needs 9 points, has {L}")
    return (3*y[:,:-8,:]-32*y[:,1:-7,:] + 168 * y[:,2:-6,:] - 672*y[:,3:-5,:] +\
             672*y[:,5:-3,:] - 168*y[:,6:-2,:] + 32*y[:,7:-1,:] - 3*y[:,8:,:])/840

def deriv_approx_d2y(y):

    B,L,d = y.shape
    assert L >= 5, print(f"y is not long enough to approximate with 9 points! needs 9 points, has {L}")
    return (-9 * y[:,:-8,:] + 128*y[:,1:-7,:] -1008*y[:,2:-6,:] + 8064*y[:,3:-5,:]- 14350*y[:,4:-4,:] + \
            8064*y[:,5:-3,:] - 1008*y[:,6:-2,:] + 128* y[:,7:-1,:] - 9*y[:,8:,:])/5040

from scipy.interpolate import make_smoothing_spline
def spline_approx_signal(y,dt,lam=5):

    spline_approx = []
    B,L,_ = y.shape
    t = np.arange(0,L*dt +dt/2,dt)[:L]
    for sample in y:

        sample = sample.detach().cpu().numpy().squeeze()
        spl = make_smoothing_spline(t,sample,lam=lam)
        spline_approx.append(spl(t))

    spline_approx = np.stack(spline_approx,axis=0)
    assert spline_approx.shape == (B,L), print(spline_approx.shape)
    return from_numpy(spline_approx[:,None,:]).to('cuda').to(torch.float32)

def sse(yhat,y,reduction='mean'):
    if reduction == 'mean':
        return ((yhat - y)**2).sum(dim=1).mean()
    elif reduction == 'sum':
        return ((yhat - y)**2).sum(dim=1).sum()
    else:
        return ((yhat - y)**2).sum(dim=1)
def sst(y,reduction='mean'):
    if reduction == 'mean':
        return ((y - y.mean(dim=1,keepdims=True))**2).sum(dim=1).mean()
    elif reduction == 'sum':
        return ((y - y.mean(dim=1,keepdims=True))**2).sum(dim=1).sum()
    else:
        return ((y - y.mean(dim=1,keepdims=True))**2).sum(dim=1)
    
def euler_step_k(y,dy,d2y,dt,k=1):

    #print(k)
    d2ydt = d2y*dt 
    ### how do we want to structure this? think about it.
    ### we need to decide if we just want gradients on the k'th step,
    ### or everything inbetween. If everything inbetween, we should output
    ### the corresponding y -- maybe stacked on the last dimension,
    ### so that for each l in L, d1 -> k=1, d2 -> k=2, d3 -> k=3, etc.
    y_out = []
    dy_out = []
    dyhat_out = []
    yhat_out = []
    _,L,_ = y.shape
    for step in range(k):
        #print(step)
        y_step = y[:,step+1:(L - k+1)+step,:]
        dy_step = dy[:,step:L-k+step,:]
        y_out.append(y_step)
        dy_out.append(dy_step)
        #print(y_step.shape,dy_step.shape)
        if step == 0:
            dy_pred = dy_out[0] + d2ydt[:,step:-k+step,:]
        else:
            dy_pred = dyhat_out[step-1] + d2ydt[:,step:-k+step,:]
        dyhat_out.append(dy_pred)

        if step == 0:
            y_pred = y_out[0] + dyhat_out[step] * dt
        else:
            y_pred = yhat_out[step-1] + dyhat_out[step]*dt
        yhat_out.append(y_pred)

    y_out = torch.cat(y_out,dim=-1)
    dy_out = torch.cat(dy_out,dim=-1)
    dyhat_out = torch.cat(dyhat_out,dim=-1)
    yhat_out = torch.cat(yhat_out,dim=-1)
    
    assert np.all(yhat_out.shape == y_out.shape),print(yhat_out.shape,y_out.shape)
    assert np.all(dy_out.shape == dyhat_out.shape),print(dy_out.shape,dyhat_out.shape)
    return (y_out,yhat_out),(dy_out,dyhat_out)

def from_numpy(data,device='cuda'):

    return torch.from_numpy(data).type(torch.FloatTensor).to(device)

def remove_axes(axis):

    axis.set_xticks([])
    axis.set_yticks([])

def get_spec(audio,fs,onset,offset,shoulder=0.05,n_freq_bins = 64,win_len=128,interp=True,normalize=True,min=-6.5,max=5,spec_type='log'):

    """
    make a spectrogram for a given vocalization.
    """
    flag = 0
    if offset - onset <= 0.05:
        flag = 1
    audiotimes = np.linspace(0,len(audio)/fs,len(audio))
    a,t = audio[np.searchsorted(audiotimes,onset-shoulder):np.searchsorted(audiotimes,offset+shoulder)],\
        audiotimes[np.searchsorted(audiotimes,onset-shoulder):np.searchsorted(audiotimes,offset+shoulder)]
    N = len(t)
    w = hann(win_len, sym=True)  # symmetric Gaussian window

    transform = STFFT(w,hop=16,fs=fs,mfft = 1028)

    Sx = transform.stft(a)
    
    t_lo,t_hi,f_lo,f_hi = transform.extent(N)
    tAx = np.linspace(t_lo,t_hi,Sx.shape[1]) + onset
    
    t0,t1 = np.searchsorted(tAx,onset),np.searchsorted(tAx,offset)

    Sx,tAx = Sx[:,t0:t1],tAx[t0:t1]

    fAx = np.linspace(f_lo,f_hi,Sx.shape[0])
    target_freqs = np.linspace(f_lo,f_hi,n_freq_bins)
    target_ts = np.arange(tAx[0],tAx[-1],0.001)
    
    if spec_type == 'log':
        Sx = np.log(np.abs(Sx) + 1e-12)
    elif spec_type == 'abs':
        Sx = np.abs(Sx)
    else:
        print(f"error: {spec_type} spectrogram not implemented")
        assert False 
    if normalize:
        if min == None:
            min = np.amin(Sx)
        if max == None:
            max = np.amax(Sx)
        
        Sx = (Sx - min) / max
        Sx = np.clip(Sx, 0.0, 1.0)

    if interp:
        interp = RegularGridInterpolator((fAx,tAx),Sx,bounds_error=True,fill_value=-1e10)

        newX,newY = np.meshgrid(target_freqs,target_ts,indexing='ij',sparse=True)

        Sx2 = interp((newX,newY))
        Sx = Sx2
    else:
        target_ts,target_freqs = tAx,fAx

    
    return Sx,target_ts,target_freqs,flag



from scipy import signal

def butter(cutoff, fs, order=5,btype='high'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def butter_filter(data, cutoff, fs, order=5,axis=-1,btype='high'):
    b, a = butter(cutoff, fs, order=order,btype=btype)
    y = signal.filtfilt(b, a, data,axis=axis)
    return y

def huber_loss(x,delta=1):

    _,L,_ = x.shape
    mask = (x.abs() <= delta).to(torch.float64)
    nMasked = mask.sum(dim=(1))
    unMasked = torch.maximum(L - nMasked,torch.ones(nMasked.shape,device=mask.device))
    maskedInds = nMasked > 0
    unMaskedInds = unMasked >0

    
    t1 = (((x*mask) **2) /2).sum(dim=1)/nMasked
    #else:
    #    t1 = 0
    #if unMasked > 0:
    t2 = ((delta*(x.abs() - delta/2))*(1 - mask)).sum(dim=1)/unMasked
    #else:
    #    t2 = 0
    #t2NonZero = t2[t2!=0]
    #print(t2NonZero.shape)
    #assert torch.all(t2NonZero >= 0), print(t2NonZero,nMasked,unMasked)
    assert torch.all(t1 >= 0), print('t1 should be greater than or equal to 0')
    assert torch.all(t2 >= 0), print('t2 should be greater than or equal to 0')
    return  t1 + t2

def smooth(data,smooth_len,smooth_type='causal'):

    D,L = data.shape
    if smooth_len == 1:
        return data
    if smooth_type == 'causal':
        start = smooth_len
        end = L
        pad = np.zeros((D,smooth_len))
        padded = np.hstack([pad,data])
    elif smooth_type == 'centered':
        
        frontLen = smooth_len // 2#+ smooth_len - 2* (smooth_len // 2)
        endLen = smooth_len // 2
        start = frontLen
        end = L + smooth_len - endLen - 1
        frontPad = np.zeros((D,frontLen))
        #endPad = np.zeros((D,endLen))
        padded = np.hstack([frontPad,data])
    else:
        start = 1
        end = L + 1
        pad = np.zeros((D,smooth_len))
        padded = np.hstack([np.zeros((D,1)),data,pad])

    cumsum = np.cumsum(padded,axis=-1)
    if smooth_type == 'causal':
        corrected=  cumsum[:,smooth_len:] - cumsum[:,:L]
    elif smooth_type == 'centered':
        corrected = cumsum[:,frontLen:] - cumsum[:,:L]
    else:
        corrected = cumsum[:,smooth_len+1:] - cumsum[:,:L] 
    #print(corrected.shape)
    
    return corrected / float(smooth_len)

def remove_rm(integrated_data,rm_length=5,smooth_type='causal'):

    smoothed = smooth(integrated_data,smooth_len=rm_length,smooth_type=smooth_type)

    return integrated_data - smoothed

def integrate_d2y(d2y,t_samples,init_cond,method='RK45'):

    interp_d2y = lambda t: np.interp(t,t_samples,d2y)


    def dz(t,z):

        dz1 = z[1]
        dz2 = interp_d2y(t)

        return np.hstack([dz1,dz2])
    
    obj = solve_ivp(dz,t_span=(t_samples[0],t_samples[-1]),\
                    y0=init_cond,method=method,t_eval=t_samples)
    
    return obj.y

