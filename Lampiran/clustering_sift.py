# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:41:29 2022

@author: gianm
"""

#%%
#Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import pandas as pd
import numpy as np

import util
from clustermodel import ClusterModel

import argparse
import time

from threading import Thread

#%%
#Load Dataset
def load_dataset(dataset_dir, maxheight=600, maxwidth=600):
    dataset = util.get_dataset(dataset_dir, maxheight=maxheight, maxwidth=maxwidth)
    return dataset

#%% 
#Detect keypoint
def detect_keypoint(dataset):
    sift = cv2.SIFT_create()

    keypoints = list()
    labels = list()

    #buat kolom untuk menyimpan gambar asal
    kp_index = 0
    desc_arrays = list()
    for group in dataset.keys():
        for i, img in enumerate(dataset[group]):
            kp, desc = sift.detectAndCompute(img[1], None)
            #masukkan ke dataframe dataset
            desc_arrays.append(desc)
            for i, k in enumerate(kp):
                #masukkan keypoint ke list
                keypoints.append(k)
                #label untuk tiap deskriptor, berisi: nama gambar, kelas gambar, dan index keypoint pada list
                labels.append((group, img[0], kp_index))
                kp_index += 1

    descs = np.concatenate(desc_arrays, axis=0)
    descriptors = pd.DataFrame(descs)            

    #masukkan labels ke dalam dataframe
    descriptors.loc[:, 'img'] = [i[1] for i in labels]
    descriptors.loc[:, 'img_class'] = [i[0] for i in labels]
    descriptors.loc[:, 'kp_idx'] = [i[2] for i in labels]

    return keypoints, descriptors

#%% 
#Cluster per image
def cluster_per_image(descriptors):
    descriptors_ = descriptors
    img_group = descriptors_.groupby('img')

    #one_image_dict = dict()
    descs_cluster_label = list()
    df_centroids = list()

    for image, one_image in img_group:
        one_image_desc_only = one_image.drop(columns=['img', 'img_class', 'kp_idx'])
        sample_size = 100
        if len(one_image_desc_only.index) < sample_size:
            sample_size = None
        threshold = util.combination_distance(
            one_image_desc_only, 
            sample=sample_size,
            metric=util.euclidean_dist
        )
        one_image_cluster_label = util.agglo_cluster(
            one_image_desc_only,
            distance_threshold=threshold,
            affinity='euclidean'
        )
        one_image.loc[:, 'cluster_label'] = one_image_cluster_label
        descs_cluster_label = descs_cluster_label + list(one_image_cluster_label)
        
        cluster_label_group = one_image.groupby('cluster_label')
        
        centroid_arrays = list()
        labels_list = list()
        for label, one_cluster in cluster_label_group:
            one_cluster = one_cluster.drop(columns=['cluster_label', 'img', 'img_class', 'kp_idx'])
            mean_arr = np.array([np.mean(one_cluster.values, axis=0)])
            centroid_arrays.append(mean_arr)
            labels_list.append(label)
        centroid_arr = np.concatenate(centroid_arrays, axis=0)

        centroid_df = pd.DataFrame(centroid_arr)
        centroid_df.loc[:, 'cluster_label'] = labels_list
        centroid_df.loc[:, 'img'] = [image] * len(centroid_df.index)
        centroid_df.loc[:, 'img_class'] = one_image.img_class.head(len(centroid_df.index)).tolist()
        df_centroids.append(centroid_df)

    df_centroid = pd.concat(df_centroids, axis=0)
    df_centroid = df_centroid.reset_index()

    descriptors_['cluster_label'] = descs_cluster_label

    return df_centroid, descriptors_

#%% 
#Cluster centroid
def cluster_centroid(df_centroid):
    df_centroid_ = df_centroid
    sample_size = 100
    threshold = util.combination_distance(
        df_centroid_.drop(columns=['img', 'img_class', 'cluster_label']), 
        sample=sample_size,
        metric=util.euclidean_dist
    )

    cluster2_labels = util.agglo_cluster(
        df_centroid_.drop(columns=['img', 'img_class', 'cluster_label']), 
        distance_threshold=threshold,
        affinity='euclidean'
    )

    df_centroid_['cluster2_label'] = cluster2_labels 
    return df_centroid_

#%%
#Calculate uniqueness
def calculate_uniqueness(df_centroid):
    df_centroid_ = df_centroid
    cluster2_class_count = dict()
    cluster2_group = df_centroid_.groupby('cluster2_label')

    for c2 in cluster2_group.groups.keys():
        one_cluster2 = cluster2_group.get_group(c2)
        one_cluster2 = one_cluster2.drop_duplicates(subset='img')
        cluster2_class_count[c2] = one_cluster2['img_class'].value_counts(normalize=True)
        
    img_class_count = [cluster2_class_count[c2lab][imgc] for c2lab, imgc in zip(df_centroid_.cluster2_label, df_centroid_.img_class)]
    df_centroid_['uniqueness'] = img_class_count

    return df_centroid_

#%% 
#Calculate consistency
def calculate_consistency(df_centroid, num_of_images):
    df_centroid_ = df_centroid
    cluster2_img_class_group = df_centroid_.groupby(['cluster2_label', 'img_class']).img.nunique()
    img_count = list()

    for c2l, ic in zip(df_centroid_.cluster2_label, df_centroid_.img_class):
        img_in_class = cluster2_img_class_group[c2l][ic]
        img_count.append(img_in_class / num_of_images)

    df_centroid_['consistency'] = img_count

    return df_centroid_

#%% 
#Join one image with centroid
def join_with_centroid(df_centroid, descriptors):
    df_centroid_val = df_centroid[['img', 'cluster_label', 'cluster2_label', 'uniqueness', 'consistency']]
    descriptors_ = pd.merge(descriptors, df_centroid_val, on=['img', 'cluster_label'])

    return descriptors_

#%% 
#Save keypoints details
def save_keypoint_details(keypoints, descriptors):
    descriptors_ = descriptors
    point_x = list()
    point_y = list()
    size = list()
    angle = list()
    response = list()
    octave = list()
    class_id = list()
    for k in descriptors_.kp_idx:
        kp = keypoints[k]
        point_x.append(kp.pt[0])
        point_y.append(kp.pt[1])
        size.append(kp.size)
        angle.append(kp.angle)
        response.append(kp.response)
        octave.append(kp.octave)
        class_id.append(kp.class_id)
    descriptors_['point_x'] = point_x
    descriptors_['point_y'] = point_y
    descriptors_['size'] = size
    descriptors_['angle'] = angle
    descriptors_['response'] = response
    descriptors_['octave'] = octave
    descriptors_['class_id'] = class_id

    return descriptors_

def dataset_details(dataset):
    class_names = list(dataset.keys())
    num_of_images = set([len(i) for i in dataset.values()])
    if len(num_of_images) == 1:
        return class_names, list(num_of_images)[0]
    else:
        raise ValueError('Dataset contains uneven number of images!')

def loading_animation(process_name):
    anim = '|/-\\'
    idx = 0
    while True:
        print(f'{process_name} --- {anim[idx % len(anim)]}', end='\r')
        idx += 1
        time.sleep(0.1)
        global stop_threads
        if stop_threads:
            break
    
def main(args):
    dir = args.dir
    maxsize = int(args.maxsize)

    start_time = time.time()
    global stop_threads

    #Load Dataset
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Load Dataset', ))
    thread.start()
    dataset = load_dataset(dir, maxheight=maxsize, maxwidth=maxsize)
    class_names, image_per_class = dataset_details(dataset)
    load_dataset_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Load Dataset --- {round(load_dataset_time - start_time, 2)}s')

    #Detect Keypoints
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Detect Keypoints', ))
    thread.start()
    keypoints, descriptors = detect_keypoint(dataset)
    detect_keypoint_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Detect Keypoints --- {round(detect_keypoint_time - load_dataset_time, 2)}s')

    #Cluster per Image
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Cluster per Image', ))
    thread.start()
    df_centroid, descriptors = cluster_per_image(descriptors)
    cluster_per_image_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Cluster per Image --- {round(cluster_per_image_time - detect_keypoint_time, 2)}s')

    #Cluster Centroid
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Cluster Centroid', ))
    thread.start()
    df_centroid = cluster_centroid(df_centroid)
    cluster_centroid_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Cluster Centroid --- {round(cluster_centroid_time - cluster_per_image_time, 2)}s')

    #Calculate Uniqueness
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Calculate Uniqueness', ))
    thread.start()
    df_centroid = calculate_uniqueness(df_centroid)
    calculate_uniqueness_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Calculate Uniqueness --- {round(calculate_uniqueness_time - cluster_centroid_time, 2)}s')

    #Calculate Consistency
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Calculate Consistency', ))
    thread.start()
    df_centroid = calculate_consistency(df_centroid, image_per_class)
    calculate_consistency_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Calculate Consistency --- {round(calculate_consistency_time - calculate_uniqueness_time, 2)}s')

    #Join with Centroid
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Join with Centroid', ))
    thread.start()
    descriptors = join_with_centroid(df_centroid, descriptors)
    join_with_centroid_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Join with Centroid --- {round(join_with_centroid_time - calculate_consistency_time, 2)}s')

    #Save Keypoint Details
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Save Keypoint Details', ))
    thread.start()
    descriptors = save_keypoint_details(keypoints, descriptors)
    save_keypoint_details_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Save Keypoint Details --- {round(save_keypoint_details_time - join_with_centroid_time, 2)}s')

    #create ClusterModel
    stop_threads = False
    thread = Thread(target=loading_animation, args=('Create ClusterModel', ))
    thread.start()
    n = len(descriptors.index)
    desc_type = ClusterModel.Descriptor.SIFT
    
    cm = ClusterModel(
        n=n,
        desc_type=desc_type,
        class_names=class_names,
        image_per_class=image_per_class
        )

    cm.set_data(dataframe=descriptors)

    dataset_name = dir.split('/')[1]
    cm_name = f'{dataset_name}_sift_agglo_{maxsize}.cm'
    cm.save(cm_name)
    create_clustermodel_time = time.time()
    stop_threads = True
    thread.join()
    print(f'Create ClusterModel --- {round(create_clustermodel_time - save_keypoint_details_time, 2)}s')

    total_time = time.time()
    print(f'  Model saved to {cm_name}')
    print(f'\nTotal time --- {round(total_time - start_time, 2)}s')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clustering')
    parser.add_argument('--dir', help='file dir', default=None)
    parser.add_argument('--maxsize', help='maximum image size', default=600)

    args = parser.parse_args()

    main(args)
