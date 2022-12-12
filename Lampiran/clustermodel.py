import pickle
import pandas as pd
import numpy as np
import cv2

from enum import Enum

class ClusterModel:    
    class Meta:
        def __init__(self, n: int, desc_type, class_names, image_per_class):
            self.n = n
            self.desc_type = desc_type
            self.class_names = class_names
            self.image_per_class = image_per_class

    class Descriptor(Enum):
        ORB = 'ORB'
        SIFT = 'SIFT'

    class LocalFeature:
        def __init__(self, descriptor, keypoint, img, img_class, consistency, uniqueness):
            self.descriptor = descriptor
            self.keypoint = keypoint
            self.img = img
            self.img_class = img_class
            self.consistency = consistency
            self.uniqueness = uniqueness

    def __init__(self, n, desc_type, class_names, images_per_class):
        self.meta = self.Meta(
            n=n,
            desc_type=desc_type,
            class_names=class_names,
            image_per_class=images_per_class
        )
    
    def set_data(self, **kwargs):
        if 'dataframe' in kwargs:
            self.from_dataframe(kwargs.get('dataframe'))
            

    def from_dataframe(self, dataset):
        #get descriptor array
        if self.meta.desc_type == self.Descriptor.SIFT:
            self.descriptors = dataset.iloc[:, :128].values
        elif self.meta.desc_type == self.Descriptor.ORB:
            self.descriptors = dataset.iloc[:, :32].values
        
        #get cluster labels
        self.cluster_label = dataset.cluster_label.values
        self.cluster2_label = dataset.cluster2_label.values

        #get keypoints array
        keypoint_columns = ['point_x', 'point_y', 'size', 'angle', 'response', 'octave', 'class_id']
        self.keypoints = dataset.loc[:, keypoint_columns].values

        #get image details
        self.img = dataset.img.values
        self.img_class = dataset.img_class.values
        
        #get consistency and uniqueness
        self.consistency = dataset.consistency.values
        self.uniqueness = dataset.uniqueness.values
    
    def get_local_features(self, min_uniqueness=0.0, min_consistency=0.0, img=None, img_class=None):
        cons_mask = self.consistency >= min_consistency
        uniq_mask = self.uniqueness >= min_uniqueness

        if img:
            img_mask = self.img == img
        else:
            img_mask = np.full(self.img.shape[0], True)
        if img_class:
            img_class_mask = self.img_class == img_class
        else:
            img_class_mask = np.full(self.img_class.shape[0], True)

        mask = cons_mask & uniq_mask & img_mask & img_class_mask

        filtered_descriptors = self.descriptors[mask]
        filtered_keypoints = np.apply_along_axis(self.get_keypoint, axis=1, arr=self.keypoints[mask])
        filtered_img = self.img[mask]
        filtered_img_class = self.img_class[mask]
        filtered_consistency = self.consistency[mask]
        filtered_uniqueness = self.uniqueness[mask]

        filtered_columns = zip(filtered_descriptors, 
                                filtered_keypoints, 
                                filtered_img, 
                                filtered_img_class, 
                                filtered_consistency, 
                                filtered_uniqueness)

        local_features = list()
        for desc, kp, img, img_class, cons, uniq in filtered_columns:
            local_features.append(self.LocalFeature(desc, kp, img, img_class, cons, uniq))
        
        return np.array(local_features, dtype=self.LocalFeature)
    
    def get_dataframe(self):
        df = pd.DataFrame(self.descriptors)
        df['img'] = self.img
        df['img_class'] = self.img_class
        df['cluster_label'] = self.cluster_label
        df['cluster2_label'] = self.cluster2_label
        df['uniqueness'] = self.uniqueness
        df['consistency'] = self.consistency

        return df

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file, 2)

    def load(self, filename):
        with open(filename, 'rb') as file:
            class_dict = pickle.load(file)
            self.__dict__.update(class_dict)

    def get_keypoint(self, row):
        keypoint = cv2.KeyPoint(
            x=float(row[0]),
            y=float(row[1]),
            size=float(row[2]),
            angle=float(row[3]),
            response=float(row[4]),
            octave=int(row[5]),
            class_id=int(row[5])
        )

        return keypoint
