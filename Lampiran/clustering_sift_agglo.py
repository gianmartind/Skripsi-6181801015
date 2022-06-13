# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:41:29 2022

@author: gianm
"""

#%%
import cv2
import pandas as pd
import numpy as np
import util
import pickle

#%%
dataset = util.get_dataset('datasets/book_covers_half/', maxheight=400, maxwidth=400)

#%% Detect keypoint
sift = cv2.xfeatures2d.SIFT_create(250)

keypoints = list()
descriptors = pd.DataFrame()
labels = list()

#buat 128 kolom baru
for i in range(128):
    descriptors.insert(i, i, 0.0)

#buat kolom untuk menyimpan gambar asal
kp_index = 0
for group in dataset.keys():
    for i, img in enumerate(dataset[group]):
        kp, desc = sift.detectAndCompute(img[1], None)
        #hasil deteksi deskriptor menjadi dataframe
        desc_1 = pd.DataFrame(desc)
        #masukkan ke dataframe dataset
        descriptors = descriptors.append(desc_1, ignore_index=True)
        for i, k in enumerate(kp):
            #masukkan keypoint ke list
            keypoints.append(k)
            #label untuk tiap deskriptor, berisi: nama gambar, kelas gambar, dan index keypoint pada list
            labels.append((group, img[0], kp_index))
            kp_index += 1

#masukkan labels ke dalam dataframe
descriptors['img'] = [i[1] for i in labels]
descriptors['img_class'] = [i[0] for i in labels]
descriptors['kp_idx'] = [i[2] for i in labels]

#%% cluster per image
img_group = descriptors.groupby('img')

#one_image_dict = dict()
descs_cluster_label = list()
df_centroid = pd.DataFrame()

for image in img_group.groups.keys():
    one_image = img_group.get_group(image)
    one_image_desc_only = one_image.drop(columns=['img', 'img_class', 'kp_idx'])
    sample_size = 50
    if len(one_image_desc_only.index) < 50:
        sample_size = None
    threshold = util.average_distance(
        one_image_desc_only, 
        sample=sample_size,
        metric=util.euclidean_dist
    )
    one_image_cluster_label = util.agglo_cluster(
        one_image_desc_only,
        distance_threshold=threshold/2,
        affinity='euclidean'
    )
    one_image['cluster_label'] = one_image_cluster_label
    descs_cluster_label = descs_cluster_label + list(one_image_cluster_label)
    
    cluster_label_group = one_image.groupby('cluster_label')
    
    labels_list = list()
    centroid_arr = np.empty([0, 128])
    labels_list = list()
    for label in cluster_label_group.groups.keys():
        one_cluster = cluster_label_group.get_group(label).drop(columns=['cluster_label', 'img', 'img_class', 'kp_idx'])
        mean_arr = np.array([np.mean(one_cluster.values, axis=0)])
        centroid_arr = np.concatenate((centroid_arr, mean_arr))
        labels_list.append(label)

    centroid_df = pd.DataFrame(centroid_arr)
    centroid_df['cluster_label'] = labels_list
    centroid_df['img'] = [image for i in range(len(centroid_df))]
    centroid_df['img_class'] = one_image['img_class'][0:len(centroid_df.index)].tolist()
    df_centroid = df_centroid.append(centroid_df)

descriptors['cluster_label'] = descs_cluster_label

#%% cluster centroid
threshold = util.average_distance(
    df_centroid.drop(columns=['img', 'img_class', 'cluster_label']), 
    sample=100,
    metric=util.euclidean_dist
)
cluster2_labels = util.agglo_cluster(
    df_centroid.drop(columns=['img', 'img_class', 'cluster_label']), 
    distance_threshold=threshold/2,
    affinity='euclidean'
)

df_centroid['cluster2_label'] = cluster2_labels 

#%% cluster2 class count
cluster2_class_count = dict()
cluster2_group = df_centroid.groupby('cluster2_label')

for c2 in cluster2_group.groups.keys():
    one_cluster2 = cluster2_group.get_group(c2)
    one_cluster2 = one_cluster2.drop_duplicates(subset='img')
    cluster2_class_count[c2] = one_cluster2['img_class'].value_counts(normalize=True)
    
#%% put class count in df_centroid
img_class_count = [cluster2_class_count[c2lab][imgc] for c2lab, imgc in zip(df_centroid.cluster2_label, df_centroid.img_class)]
df_centroid['uniqueness'] = img_class_count

#%% count number of different images
cluster2_img_class_group = df_centroid.groupby(['cluster2_label', 'img_class']).img.nunique()
img_count = list()

for c2l, ic in zip(df_centroid.cluster2_label, df_centroid.img_class):
    img_in_class = cluster2_img_class_group[c2l][ic]
    img_count.append(img_in_class / 4)

df_centroid['consistency'] = img_count

#%% join one image with centroid
df_centroid_val = df_centroid[['img', 'cluster_label', 'cluster2_label', 'uniqueness', 'consistency']]
descriptors = pd.merge(descriptors, df_centroid_val, on=['img', 'cluster_label'])

#%% save keypoints details
point_x = list()
point_y = list()
size = list()
angle = list()
response = list()
octave = list()
class_id = list()
for k in descriptors.kp_idx:
    kp = keypoints[k]
    point_x.append(kp.pt[0])
    point_y.append(kp.pt[1])
    size.append(kp.size)
    angle.append(kp.angle)
    response.append(kp.response)
    octave.append(kp.octave)
    class_id.append(kp.class_id)
descriptors['kp_point_x'] = point_x
descriptors['kp_point_y'] = point_y
descriptors['size'] = size
descriptors['angle'] = angle
descriptors['response'] = response
descriptors['octave'] = octave
descriptors['class_id'] = class_id

#%% 
pickle.dump(descriptors, open('{}.pickle'.format(input()), 'wb'))

