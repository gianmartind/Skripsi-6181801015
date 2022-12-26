from modules.bsis import BSIS
from modules.clustermodel import ClusterModel
from modules import util
import cv2
import time
import os

models_dir = './static/models/'

def detect_image(model, consistency, uniqueness, img_dir):
    global models_dir

    #load ClusterModel
    cm = ClusterModel()
    cm.load(f'{models_dir}{model}')

    #get train_data
    train_data = cm.get_local_features(min_consistency=consistency, min_uniqueness=uniqueness)

    #load image
    img_name = img_dir.split('/')[-1]
    img = util.get_image(img_dir, maxheight=600, maxwidth=600)
    
    #detect keypoints
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    query = {
        'kp': kp,
        'desc': desc
    }

    #BSIS
    bsis = BSIS(query)
    bsis.set_train_data(train_data)
    most_similar = bsis.run(num_rotation=20, algorithm=BSIS.FLANN_INDEX.KDTREE, k=100, t=3)

    #Result
    res = list()
    for k, v in sorted(bsis.result.items(), key=lambda i: i[1]['total_weight'], reverse=True)[:10]:
        train_img_class = k.split('_')[0]
        train_img_dir = f'./static/dataset/poi/{train_img_class}/{k}'
        train_img = util.get_image(train_img_dir, maxheight=600, maxwidth=600)

        img_matches = util.show_matches(img, v['query_kp'], train_img, v['train_kp'], return_img=True)
        img_matches_name = f'{time.time()}.jpg'
        img_matches_dir = f'./static/detection/{img_matches_name}'
        cv2.imwrite(img_matches_dir, img_matches)

        if v['total_weight'] > 0:
            res.append((k, round(v['total_weight'], 3), img_matches_name))
        else:
            break
    
    return res
