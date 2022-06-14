import scipy
from scipy.fftpack import fft, ifft
import numpy as np
import sys
import nibabel 
import argparse
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy.linalg as npl


def hrf(time_frame):
    hrf=gamma.pdf(time_frame,6,loc=0,scale=1)-gamma.pdf(time_frame,16,loc=0,scale=1)/6
    return hrf/np.max(hrf)

def time_frame():
    return np.arange(0,20,TR)

def stimuli(EV):
    evs={}

    for i in range(len(EV)):
        index=str(int(EV[i][-1])-1)
        if index not in evs:
            evs[index]=[]
        evs[index].append(EV[i][:-1])

    stimulus= np.zeros((exp_conditions,vols))
    for i in range(exp_conditions):
        time=np.zeros(vols)
        index=str(i)
        for idx in range(len(evs[index])):
            onset=evs[index][idx][0]
            duration=evs[index][idx][1]
            wt=evs[index][idx][2]
            time[int(onset/TR):int((onset+duration)/TR)]=wt
        stimulus[i]=time

    return stimulus   

def convolve_with_hrf(stimulus,hrf):
   
    return convolve(stimulus,hrf)


def convolve(data,kernel1D): #returns convolution of 1D data with 1D kernel
    kernel=len(kernel1D) # kernel of size 5 picked
    center=int(kernel//2)
    y=np.zeros(len(data))
    h=kernel1D
    g=np.concatenate((np.zeros(kernel//2),data,np.zeros(kernel//2)))
    for i in range(kernel//2,(kernel//2+len(data))):
        i_=i-center
        y[i_]=sum(h[center+k]*g[i-k] for k in range(-1*(kernel//2),(kernel//2)+1))
    return y

def design(stimulus,hrf):
    matrix=np.zeros((vols))
    #matrix[:,0]=1
    matrix[:]=convolve_with_hrf(stimulus,hrf)
    return matrix

def noise():
    error=np.zeros((vols,1))
    error[:,0]=np.random.normal(vols)
    return error

def beta_cap(des):
    beta=np.zeros((fmri_data.shape[0],fmri_data.shape[1],fmri_data.shape[2],exp_conditions))
    for x in range(fmri_data.shape[0]):
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                Y=fmri_data[x][y][z]
                
                X=des
                beta_caps=np.dot(np.dot(npl.pinv(np.dot(X.T,X)),X.T),Y)
                
                beta[x][y][z]=beta_caps
    return beta

def contrasts(beta,contrasts):
    
    c=np.zeros((fmri_data.shape[0],fmri_data.shape[1],fmri_data.shape[2]))
    for x in range(fmri_data.shape[0]):
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                c_values=np.dot(contrasts,beta[x,y,z,:])
                
                c[x][y][z]=c_values
    return c

def zstat(contrast):
    
    z_stat= np.zeros((fmri_data.shape[0],fmri_data.shape[1],fmri_data.shape[2]))
    
       
    for x in range(fmri_data.shape[0]):
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                mean=np.dot(contrast,beta_caps[x,y,z,:])/exp_conditions
                df=vols-exp_conditions
                error=fmri_data[x][y][z]-np.dot(des,beta_caps[x][y][z])
                Sum_Squares = np.dot(error.T,error)
                MSS=Sum_Squares/df
                sigma_sq=(MSS)
                Std_error=np.sqrt(sigma_sq*(np.dot(np.dot(contrast,npl.inv(np.dot(des.T,des))),contrast.T)))
                z_stat[x][y][z]=mean/(Std_error+0.0001)
    return z_stat

def tstat(contrast):
    
    t_stat= np.zeros((fmri_data.shape[0],fmri_data.shape[1],fmri_data.shape[2]))
    
     
    for x in range(fmri_data.shape[0]):
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                mean=np.dot(contrast,beta_caps[x,y,z,:])/exp_conditions
                df=vols-exp_conditions
                error=fmri_data[x][y][z]-np.dot(des,beta_caps[x][y][z])
                Sum_Squares = np.dot(error.T,error)
                MSS=Sum_Squares/df
                sigma_sq=(MSS)
                Std_error=np.sqrt(sigma_sq*(np.dot(np.dot(contrast,npl.pinv(np.dot(des.T,des))),contrast.T)))/np.sqrt(exp_conditions)
                t_stat[x][y][z]=mean/(Std_error+0.0001)
    return t_stat


if __name__ == '__main__':

    scan=nibabel.load(str(sys.argv[1]))

    fmri_data=scan.get_fdata() # returns 4D tuple of size (64, 64, 40, 128); 40=no of slices and 128=no of brain volumes 
    header=scan.header
    details=header.get_zooms() 
    voxel_dim=details[:3] # voxel dimensions
    
    TR=details[-1] # in seconds
    
    slices=fmri_data.shape[2]
    vols=fmri_data.shape[3]
    #vols=128
    datatype=str(fmri_data.dtype)
    duration= TR*vols

    ev=np.loadtxt(str(sys.argv[2]))
    contrast=np.loadtxt(str(sys.argv[3]))

    contrast=np.atleast_2d(contrast)
    
    exp_conditions=int(ev[-1][-1]) #no of exp conditions
    
    stimulus=stimuli(ev)
    hrf_response=hrf(time_frame())

    output=str(sys.argv[4])
    
    des=np.zeros((vols,exp_conditions))
    for i in range(exp_conditions):
        stim=stimulus[i]
        design_matrix=design(stim,hrf_response)
        des[:,i]=design_matrix

    beta_caps=beta_cap(des)

    for i in range(exp_conditions):
        beta=beta_caps[..., i]
        file_name= output+".pe"+str(i+1)+".nii.gz"
        file_to_produce=nibabel.Nifti1Image(beta.astype(datatype),np.eye(4),header=header)
        nibabel.save(file_to_produce,file_name)
        
        
    for i in range(len(contrast)):   
        
        c=contrasts(beta_caps,contrast[i])
        
        file_name= output+".cope"+str(i+1)+".nii.gz"
        file_to_produce=nibabel.Nifti1Image((c).astype(datatype),np.eye(4),header=header)
        nibabel.save(file_to_produce,file_name)
        
        t_=tstat(contrast[i])
        file_name= output+".tstat"+str(i+1)+".nii.gz"
        file_to_produce=nibabel.Nifti1Image((t_).astype(datatype),np.eye(4),header=header)
        nibabel.save(file_to_produce,file_name)
        
        z_=zstat(contrast[i])
        file_name= output+".zstat"+str(i+1)+".nii.gz"
        file_to_produce=nibabel.Nifti1Image((z_).astype(datatype),np.eye(4),header=header)
        nibabel.save(file_to_produce,file_name)