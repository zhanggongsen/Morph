import numpy as np
import os
root=r"SGRT_PC"
PC_number_list=[]
for PC in os.listdir(root):
    PC_root=os.path.join(root,PC)
    PC_array=np.loadtxt(PC_root)
    print(PC_array.shape)
    PC_number_list.append(PC_array.shape[0])
PC_number_array=np.array(PC_number_list)
print(np.min(PC_number_array),np.max(PC_number_array),np.median(PC_number_array))