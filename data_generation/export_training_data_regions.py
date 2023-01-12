from lib.laoss_interface import LaossRun
from lib.image_processing import post_process_image
import random
import pandas as pd
import os
import numpy as np
import pathlib
import cv2

script_path = str(pathlib.Path(__file__).parent.absolute())
noise_folder=os.path.join(script_path,r"noise_samples")
names  = ["example_data"]
root = os.path.join(script_path,'..')

def pixel_max_resize(img, h, w):
    source_h, source_w = img.shape
    return img.reshape(h,source_h // h,-1,source_w // w).swapaxes(1,2).reshape(h,w,-1).max(axis=2)

if __name__ == '__main__':
    X_train = []
    y_train = pd.DataFrame()
    X_test = []
    y_test = pd.DataFrame()
    regions = []
    parameters = pd.DataFrame()
    stacked_laoss_paths = []
    for i,name in enumerate(names):
        path = os.path.join(root, name)
        dirpath, dirnames, filenames = next(os.walk(path))
        laoss_paths = [os.path.join(path, laoss_path) for laoss_path in dirnames if os.path.exists(os.path.join(path, laoss_path,'image1.npy'))]
        if i==0:
            stacked_laoss_paths.extend(laoss_paths)
        else:
            stacked_laoss_paths.extend(spath for spath in laoss_paths if os.path.basename(spath)!='image0')

    laoss_kernel_path = "C:\Program Files\Fluxim\LAOSS 4.1\laoss-kernel.exe"
    laoss_tool_path = "C:\Program Files\Fluxim\LAOSS 4.1\laoss-tool.exe"

    test = LaossRun(path,laoss_kernel_path,laoss_tool_path)

    image_size = (40,80)

    test_split=0.96
    n_total = len(stacked_laoss_paths)
    n_train = int(test_split*len(stacked_laoss_paths))
    randomlist = range(n_total)
    for i,laoss_path in enumerate(stacked_laoss_paths):
        print(laoss_path)

        try:
            laoss_result = test.load_results(laoss_path)
            df1 = laoss_result.run_data
            parameters = pd.concat([parameters, df1])

            randomize = random.uniform(.5,2)
            rand_shift = random.randint(-1,1)
            rand_blur = random.choice([1,3])

            image_path = os.path.join(laoss_result.path, "image1.npy")
            img1 = cv2.resize(np.load(image_path),(100,90))[4:84,30:70]
            img1noise = post_process_image(noise_folder,img1,blur=rand_blur,factor=14272973*randomize)#14272973
            img_norm = np.log(img1noise/np.mean(np.mean(img1noise)))* 0.0238+df1['V1']
            img1re = cv2.resize(img_norm, image_size).astype(np.float32)

            image_path = os.path.join(laoss_result.path, "image2.npy")
            img2 = cv2.resize(np.load(image_path),(100,90))[4:84,30:70]
            img2 = post_process_image(noise_folder,img2,blur=rand_blur,factor=14272973*randomize)#14272973
            img2 = np.log(img2 / np.mean(np.mean(img1))) * 0.0238 + df1['V1']

            img2 = cv2.resize(img2, image_size).astype(np.float32)

            regions_file_path = os.path.join(laoss_result.path,"LaossRegions.npy")

            if np.count_nonzero(np.isnan(img1re))==0 and np.count_nonzero(np.isnan(img2))==0:

                regions = np.flipud(np.load(regions_file_path))[4:84,30:70]
                regions = pixel_max_resize(regions, image_size[1],image_size[0]).astype(np.float32)

                n_regions = int(np.max(regions))
                for j in range(n_regions+1):
                    full_image = []
                    region_mask = regions.reshape(image_size[1], image_size[0], 1)==j
                    region_mask =1*region_mask
                    channelized = np.concatenate([img1re.reshape(image_size[1], image_size[0], 1),img2.reshape(image_size[1], image_size[0], 1),region_mask],axis=2)
                    full_image.append(channelized)
                    df1['log_dark_saturation_current_'+str(j)] = np.log10(df1['dark_saturation_current_'+str(j)])
                    df1['ln_r_sheet_' + str(j)] = np.log(df1['r_sheet_' + str(j)])
                    df1['ln_rho_par_' + str(j)] = np.log(df1['rho_par_' + str(j)])
                    region_values = df1[['log_dark_saturation_current_'+str(j), 'ln_r_sheet_'+str(j), 'r_int_'+str(j),'ln_rho_par_' + str(j),'V1', 'V2']]
                    region_values=region_values.rename({'log_dark_saturation_current_'+str(j):'log_dark_saturation_current',
                                          'ln_r_sheet_'+str(j):'ln_r_sheet',
                                          'r_int_'+str(j):'r_int',
                                          'ln_rho_par_' + str(j):'ln_rho_par'})
                    if os.path.basename(laoss_path)!='image0' and i<n_train:
                        X_train.append(channelized)
                        y_train = y_train.append(region_values, ignore_index=True)
                    else:
                        X_test.append(channelized)
                        y_test = y_test.append(region_values, ignore_index=True)

        except:
            pass

    np.save(os.path.join(path, 'x_train.npy'), np.array(X_train))
    y_train.to_csv(os.path.join(path, 'y_train.txt'),index=False)
    np.save(os.path.join(path, 'x_test.npy'), np.array(X_test))
    y_test.to_csv(os.path.join(path, 'y_test.txt'), index=False)


