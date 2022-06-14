import scipy
from scipy.fftpack import fft, ifft
import numpy as np
import sys
import nibabel 
import argparse

def correction_per_volume(vol, target, acquisition,tr):
    # Linear interpolation: y=y1+(x-x1)*(y2-y1)/(x2-x1); x2-x1=TR
    y1=vol[:-1] # excluding the last volume of brain
    y2=vol[1:]
    slope=(y2-y1)/tr
    y=y1+slope*(target-acquisition)
    #To include the last volume
    last_vol=vol[-1]
    y=np.concatenate((y,[last_vol]))
    return y

def slice_time_correction(output, target, acquisition_file):
    tr=TR*1000 # in milliseconds
    message=output+'.txt'
    f=open(message,'w')
    if tr<target or target<0:
        f.write("SLICE TIME CORRECTION FAILURE")
        f.write("\n")
        return fmri_data
    else:
        slice_time=np.loadtxt(acquisition_file)

        if slices!=len(slice_time):
            f.write("SLICE TIME CORRECTION FAILURE")
            f.write("\n")
            return fmri_data
        
        else:
            image= np.zeros(fmri_data.shape)
            for z in range(slices):
                if slice_time[z] < 0 or slice_time[z]>tr:
                    f.write("SLICE TIME CORRECTION FAILURE")
                    f.write("\n")
                    return fmri_data
                for x in range(fmri_data.shape[0]):
                    for y in range(fmri_data.shape[1]):
                        vol=fmri_data[x][y][z]
                        image[x][y][z]=correction_per_volume(vol,target,slice_time[z],tr)
        
            f.write("SLICE TIME CORRECTION SUCCESS")
            f.write("\n")
            return image


def filter_per_vol( x,y,z,low,high,freq):
    
    fourier_transform=fft(fmri_data[x][y][z])
    fourier_transform[np.where(freq>high)[0]]=0
    fourier_transform[np.where(freq<low)[0]]=0
    
    return np.real(ifft(fourier_transform))

def band_pass(low,high): # temporal band pass filter
    
    vols=(fmri_data.shape[-1])
    d=TR #sample spacing

    
    if vols%2==0:
        freq=np.concatenate((np.arange(0,vols/2),np.arange(-1*(vols/2),0)))/(d*vols)
    else:
        freq=np.concatenate((np.arange(0,((vols-1)/2+1)),np.arange(-1*(vols-1)/2,0)))/(d*vols)
    
    
    freq=abs(freq) 

    low_=1/high # low frequency cutoff
    high_=1/low # high frequency cutoff
    filtered=np.zeros(fmri_data.shape)
    for x in range(fmri_data.shape[0]):
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                filtered[x][y][z]=filter_per_vol(x,y,z,low_,high_,freq)

    return filtered

    
def sigma(fwhm):
    return fwhm/(np.sqrt(8*np.log(2)))

def kernel_1D(fwhm,dim):
    # kernel size=5 chosen for 1D
    x=np.linspace(-2,2,5)*dim
    sig=sigma(fwhm)
    gaussian_1Dkernel=1/(np.sqrt(2*np.pi)*sig)*(np.exp((-1*((x)**2/(2*(sig)**2)))))
    return gaussian_1Dkernel/sum(gaussian_1Dkernel)

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

def spatial_smoothing(fwhm):
    spatially_smoothened_image=np.zeros(fmri_data.shape)
    kernel1D_x=kernel_1D(fwhm,voxel_dim[0])
    kernel1D_y=kernel_1D(fwhm,voxel_dim[1])
    kernel1D_z=kernel_1D(fwhm,voxel_dim[2])

    for vol in range(fmri_data.shape[-1]):
        
        image=np.zeros(fmri_data.shape[:-1])

        
        for x in range(fmri_data.shape[0]):
            for y in range(fmri_data.shape[1]):
                
                image[x,y,:]=convolve(fmri_data[x,y,:,vol],kernel1D_z)
        
        
        for x in range(fmri_data.shape[0]):
            for z in range(fmri_data.shape[2]):
                image[x,:,z]=convolve(fmri_data[x,:,z,vol],kernel1D_y)
        
        
        for y in range(fmri_data.shape[1]):
            for z in range(fmri_data.shape[2]):
                image[:,y,z]=convolve(fmri_data[:,y,z,vol],kernel1D_x)
        
        spatially_smoothened_image[:,:,:,vol]=image

    return spatially_smoothened_image
        


if __name__ == '__main__':

    sys_args=argparse.ArgumentParser()
    sys_args.add_argument('-i',"--input")
    sys_args.add_argument('-o', "--output")
    sys_args.add_argument('-tc', "--slice_time_correction",nargs='+')
    sys_args.add_argument('-tf','--temporal_filtering',nargs='+')
    sys_args.add_argument('-sm',"--spatial_smoothing")

    inputs=sys_args.parse_args()

    scan=nibabel.load(str(inputs.input))

    fmri_data=scan.get_fdata() # returns 4D tuple of size (64, 64, 40, 128); 40=no of slices and 128=no of brain volumes 
    header=scan.header
    details=header.get_zooms() 
    voxel_dim=details[:3] # voxel dimensions
    
    TR=details[-1] # in seconds
    slices=fmri_data.shape[2]
    datatype=str(fmri_data.dtype)

    output=str(inputs.output)

    for i in range(len(sys.argv)):
        if sys.argv[i]=='-tc':
            
            target=float(inputs.slice_time_correction[0])
            acquisition_file=str(inputs.slice_time_correction[1])
            fmri_data=slice_time_correction(output,target,acquisition_file)
        
        elif sys.argv[i]=='-tf':
            high=float(inputs.temporal_filtering[0])
            low=float(inputs.temporal_filtering[1])
            fmri_data=band_pass(low,high)
        
        elif sys.argv[i]=='-sm':
            fwhm=float(inputs.spatial_smoothing)
            fmri_data=spatial_smoothing(fwhm)

    
    file_name=output+'.nii.gz'
    file_to_produce=nibabel.Nifti1Image(fmri_data.astype(datatype),np.eye(4),header=header)
    nibabel.save(file_to_produce,file_name)
    
    

   

    
    
