import os
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
Root = r"DCT"

file_list=[]
dir_list=[]

for root, dirs, files in os.walk(Root):
    for dir_i in dirs:
        dirs_path=os.path.join(root,dir_i)
        if len(dirs_path)==(len(Root)+11) or len(dirs_path)==(len(Root)+12)or len(dirs_path)==(len(Root)+13)or len(dirs_path)==(len(Root)+15):
            dir_list.append(dirs_path)
    for file_i in files:
        file_path = os.path.join(root, file_i)
        if 'RS' in file_i:
            if not os.path.exists(os.path.join(root, 'RS.dcm')):
                os.rename(file_path,os.path.join(root,'RS.dcm'))
                # dir_list.append(dirs_path)

for path_toProcess in dir_list:
    CT_path=path_toProcess
    RTS_path = os.path.join(path_toProcess, "RS.dcm")

    phase_num = path_toProcess.split('\\')[-1]
    file_save=os.path.dirname(CT_path)
    # print(file_save)
    save_path = os.path.join(file_save, 'structs_{}'.format(phase_num))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dcmrtstruct2nii(RTS_path, CT_path, save_path, gzip=False)
    print(save_path)
