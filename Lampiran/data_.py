import pickle
import pandas as pd
import numpy as np
from util import Image
import cv2

class Data:
    def __init__(self, path):
        self.data = pickle.load(open(path, 'rb'))

    def get_keypoints_list(self, uniqueness_filter=0.0, consistency_filter=0.0):
        train_image = list()
        data_group = self.data.groupby('img')
        for k, v in data_group:
            selected_feature = v[(v['uniqueness'] > uniqueness_filter) & (v['consistency'] > consistency_filter)]
            if len(selected_feature.index) > 0:
                keypoints = list()
                descriptors = list()
                group = selected_feature['img_class'].iloc[0]
                for idx, row in selected_feature.iterrows():
                    descriptors.append(self.get_descriptor(row))
                    keypoints.append(self.get_keypoint(row))
                train_image.append(Image(k, group, tuple(keypoints), np.array(descriptors, dtype='float32')))

        return train_image

    def get_keypoints_dict(self, uniqueness_filter=0.0, consistency_filter=0.0):
        train_image = dict()
        data_group = self.data.groupby('img')
        for k, v in data_group:
            selected_feature = v[(v['uniqueness'] > uniqueness_filter) & (v['consistency'] > consistency_filter)]
            if len(selected_feature.index) > 0:
                keypoints = list()
                descriptors = list()
                group = selected_feature['img_class'].iloc[0]
                for idx, row in selected_feature.iterrows():
                    descriptors.append(self.get_descriptor(row))
                    keypoints.append(self.get_keypoint(row))
                train_image[k] = Image(k, group, tuple(keypoints), np.array(descriptors, dtype='float32'))

        return train_image

    def get_keypoints_with_mapper(self, uniqueness_filter=0.0, consistency_filter=0.0):
        train_kp = list()
        train_desc = list()
        index_mapper = dict()
        index = 0
        data_group = self.data.groupby('img')
        for k, v in data_group:
            selected_feature = v[(v['uniqueness'] > uniqueness_filter) & (v['consistency'] > consistency_filter)]
            if len(selected_feature.index) > 0:
                group = selected_feature['img_class'].iloc[0]
                for idx, row in selected_feature.iterrows():
                    train_desc.append(self.get_descriptor(row))
                    train_kp.append(self.get_keypoint(row))
                    index_mapper[index] = k
                    index += 1
        
        train_desc = np.array(train_desc, dtype='float32')
        
        return train_kp, train_desc, index_mapper

    def get_keypoint(self, row):
        keypoint = cv2.KeyPoint(
            x=row['kp_point_x'],
            y=row['kp_point_y'],
            size=row['size'],
            angle=row['angle'],
            response=row['response'],
            octave=row['octave'],
            class_id=row['class_id']
        )

        return keypoint

    def get_descriptor(self, row):
        descriptor = row[0:128].tolist()
        return descriptor
