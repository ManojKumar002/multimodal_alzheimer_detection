import numpy as np
import skimage.transform as skTrans
import nibabel as nib
import pandas as pd
import os
import sys
import time

# change the directory to preprocessing_image

def normalize_img(img_array):
    maxes = np.quantile(img_array,0.995,axis=(0,1,2))
    #print("Max value for each modality", maxes)
    return img_array/maxes


def create_dataset(meta, meta_all,path_to_datadir):
    current_path=path_to_datadir
    files = os.listdir(path_to_datadir)
    start = '_'
    end = '.nii'
    for file in files:
        current_path1=path_to_datadir+"/"+file
        for i in (os.listdir(current_path1)):
            current_path2=current_path1+"/"+i
            for j in (os.listdir(current_path2)):
                current_path3=current_path2+"/"+j
                for k in (os.listdir(current_path3)):
                    current_path4=current_path3+"/"+k
                    for l in (os.listdir(current_path4)):
                        print(l)
                        if file != '.DS_Store':
                            currentpath5 = os.path.join(current_path4,l)
                            img_id = l.split(start)[-1].split(end)[0]
                            idx = meta[meta["Image Data ID"] == img_id].index[0]
                            im = nib.load(currentpath5).get_fdata()
                            n_i, n_j, n_k = im.shape[0],im.shape[1],im.shape[2]
                            center_i = (n_i - 1) // 2  
                            center_j = (n_j - 1) // 2
                            center_k = (n_k - 1) // 2
                            im1 = skTrans.resize(im[center_i, :, :], (72, 72), order=1, preserve_range=True)
                            im2 = skTrans.resize(im[:, center_j, :], (72, 72), order=1, preserve_range=True)
                            im3 = skTrans.resize(im[:, :, center_k], (72, 72), order=1, preserve_range=True)
                            im = np.array([im1,im2,im3]).T
                            label = meta.at[idx, "Group"]
                            subject = meta.at[idx, "Subject"]
                            norm_im = normalize_img(im)
                            meta_all = meta_all.append({"img_array": im,"label": label,"subject":subject}, ignore_index=True)
            

    meta_all.to_pickle("mri_meta.pkl")
    time.sleep(0.5)



def main():
    args = sys.argv[1:]
    path_to_meta = args[0] 
    path_to_datadir = args[1]
    # print(path_to_meta)

 
    meta = pd.read_csv(path_to_meta)
    print("opened meta")
    print(len(meta))
    #get rid of not needed columns
    meta = meta[["Image Data ID", "Group", "Subject"]] #MCI = 0, CN =1, AD = 2
    meta["Group"] = pd.factorize(meta["Group"])[0]
    #initialize new dataset where arrays will go
    meta_all = pd.DataFrame(columns = ["img_array","label","subject"])
    create_dataset(meta, meta_all, path_to_datadir)
    
if __name__ == '__main__':
    main()
    
