## PROJECT- MODELLING VISUAL FIELD OF A MONKEY
Dataset Description- mt-2 dataset of Visual Cortex from CRCNS titled “Extracellular recordings from area MT of awake macaques in response to naturalistic movies    

#### PCA-GLM   
For every neuron, we had different frames of the movie stimuli that covered twice the receptive field of that neuron. To combat the limitations of the dataset, we performed PCA on the stimuli frames and collected mean spike counts for them. We then generated a GLM to establish a connection between the movie frames and spike counts.     

### Results-
| Target Image   | Reconstructed Image |
| ------- | ------- |
|![image](https://user-images.githubusercontent.com/78497850/173567339-0b5620df-c915-40a9-966d-8422dbe9f835.png)         |   ![image](https://user-images.githubusercontent.com/78497850/173567527-34b82804-4c65-4a32-8406-999c8596e67d.png)      |
|![image](https://user-images.githubusercontent.com/78497850/173567661-ba32b7e4-e2b5-4652-91dc-c9bd77470833.png)|   ![image](https://user-images.githubusercontent.com/78497850/173567771-52c62467-21c7-4b4e-ae38-4c28a4ec7cbd.png)      |

