# %%
#load library
import pandas as pd
import cv2
import time
import pickle

from modules.bsis import BSIS
from modules import util
from modules.clustermodel import ClusterModel

from dfply import *

# %% [markdown]
# #### Parameters

# %%
dataset_name = ''
clustermodel_dir = ''
testdata_dir = ''

# %%
#load train cluster model
cluster_model = ClusterModel()
cluster_model.load(clustermodel_dir)
maxsize = cluster_model.meta.maxsize

if cluster_model.meta.desc_type == ClusterModel.Descriptor.SIFT:
    extract_method = cv2.SIFT_create()
    algorithm = BSIS.FLANN_INDEX.KDTREE
elif cluster_model.meta.desc_type == ClusterModel.Descriptor.ORB:
    extract_method = cv2.ORB_create()
    algorithm = BSIS.FLANN_INDEX.LSH

bsis_param = dict(
    num_rotation=20, 
    algorithm=algorithm,
    k=100, 
    t=3
)

#load test data 
query_dataset = util.get_dataset(testdata_dir, maxwidth=maxsize, maxheight=maxsize)

img_class = cluster_model.get_dataframe().drop_duplicates(subset=['img'])[['img', 'img_class']]
img_class = img_class.set_index('img')

# %% [markdown]
# ### Non-filtered

# %%
#get train data
train_data = cluster_model.get_local_features(min_consistency=0.0, min_uniqueness=0.0)

q_name = list()
q_class = list()
most_similar = list()
most_similar_class = list()
total_weight = list()
same_class_idx = list()
same_class_weight = list()
extract_time = list()
pairing_time = list()
total_bsis_time = list()
bsis_list = list()

# %%
for k, v in query_dataset.items():
    for i in v:
        q_name.append(i[0])
        start_time = time.time()
        kp, desc = extract_method.detectAndCompute(i[1], None)
        ext_time = time.time() - start_time
        extract_time.append(ext_time)

        print(i[0])
        print('feature_extract ---', ext_time)

        query = {'kp': kp, 'desc': desc}
        
        bsis = BSIS(query)
        bsis.set_train_data(train_data)
        
        maximum = bsis.run(**bsis_param)
        most_similar.append(maximum)
        if maximum != '':
            most_similar_class.append(img_class.loc[maximum, 'img_class'])
            total_weight.append(bsis.result[maximum]['total_weight'])
        else:
            most_similar_class.append('')
            total_weight.append(0)

        sorted_result = [(idx, k, v['total_weight']) for idx, (k, v) in enumerate(sorted(bsis.result.items(), key=lambda item: item[1]['total_weight'], reverse=True))]
        for idx, n, w in sorted_result:
            if n != '':
                if img_class.loc[n, 'img_class'] == k:
                    same_class_idx.append(idx)
                    same_class_weight.append(w)
                    break
        else:
            same_class_idx.append('')
            same_class_weight.append('')
        
        q_class.append(k)
        pairing_time.append(bsis.pairing_time)
        total_bsis_time.append(bsis.total_time)
        bsis_list.append(bsis)
        print('')

# %%
#save bsis_list
bsis_list_tupled = list()
for b in bsis_list:
    bsis_list_tupled.append([(idx, k, v['total_weight']) for idx, (k, v) in enumerate(sorted(b.result.items(), key=lambda item: item[1]['total_weight'], reverse=True))])

pickle.dump(bsis_list_tupled, open(f'{dataset_name}_{cluster_model.meta.desc_type.value}_{maxsize}_full.pkl', 'wb'))

# %%
df_result = pd.DataFrame()
df_result['q_name'] = q_name
df_result['q_class'] = q_class
df_result['most_similar'] = most_similar
df_result['most_similar_class'] = most_similar_class
df_result['total_weight'] = total_weight
df_result['same_class_idx'] = same_class_idx
df_result['same_class_weight'] = same_class_weight
df_result['extract_time'] = extract_time
df_result['pairing_time'] = pairing_time
df_result['total_bsis_time'] = total_bsis_time

is_true = list()
total = 0
for q, t in zip(q_class, most_similar):
    if t != '':
        if q == img_class.loc[t, 'img_class']:
            total += 1
            is_true.append(1)
        else:
            is_true.append(0)
    else:
        is_true.append(0)
print(total)
df_result['is_true'] = is_true 
df_result.to_csv(f'{dataset_name}_{cluster_model.meta.desc_type.value}_{maxsize}_full.csv', index=False)

# %% [markdown]
# ### Filtered 

# %%
train_data_filtered = cluster_model.get_local_features(min_consistency=0.3, min_uniqueness=0.3)

q_name_filtered = list()
q_class_filtered = list()
most_similar_filtered = list()
most_similar_class_filtered = list()
total_weight_filtered = list()
same_class_idx_filtered = list()
same_class_weight_filtered = list()
extract_time_filtered = list()
pairing_time_filtered = list()
total_bsis_time_filtered = list()
bsis_list_filtered = list()

# %%
for k, v in query_dataset.items():
    for i in v:
        q_name_filtered.append(i[0])
        start_time = time.time()
        kp, desc = extract_method.detectAndCompute(i[1], None)
        ext_time = time.time() - start_time
        extract_time_filtered.append(ext_time)

        print(i[0])
        print('feature_extract ---', ext_time)
        
        query = {'kp': kp, 'desc': desc}
        
        bsis = BSIS(query)
        bsis.set_train_data(train_data_filtered)
        
        maximum = bsis.run(**bsis_param)
        most_similar_filtered.append(maximum)
        if maximum != '':
            most_similar_class_filtered.append(img_class.loc[maximum, 'img_class'])
            total_weight_filtered.append(bsis.result[maximum]['total_weight'])
        else:
            most_similar_class_filtered.append('')
            total_weight_filtered.append(0)

        sorted_result = [(idx, k, v['total_weight']) for idx, (k, v) in enumerate(sorted(bsis.result.items(), key=lambda item: item[1]['total_weight'], reverse=True))]
        for idx, n, w in sorted_result:
            if n != '':
                if img_class.loc[n, 'img_class'] == k:
                    same_class_idx_filtered.append(idx)
                    same_class_weight_filtered.append(w)
                    break
        else:
            same_class_idx_filtered.append('')
            same_class_weight_filtered.append('')
        
        q_class_filtered.append(k)
        pairing_time_filtered.append(bsis.pairing_time)
        total_bsis_time_filtered.append(bsis.total_time)
        bsis_list_filtered.append(bsis)
        print('')

# %%
#save bsis_list
bsis_list_tupled_filtered = list()
for b in bsis_list_filtered:
    bsis_list_tupled_filtered.append([(idx, k, v['total_weight']) for idx, (k, v) in enumerate(sorted(b.result.items(), key=lambda item: item[1]['total_weight'], reverse=True))])

pickle.dump(bsis_list_tupled_filtered, open(f'{dataset_name}_{cluster_model.meta.desc_type.value}_{maxsize}_filtered.pkl', 'wb'))

# %%
df_result_filtered = pd.DataFrame()
df_result_filtered['q_name'] = q_name_filtered
df_result_filtered['q_class'] = q_class_filtered
df_result_filtered['most_similar'] = most_similar_filtered
df_result_filtered['most_similar_class'] = most_similar_class_filtered
df_result_filtered['total_weight'] = total_weight_filtered
df_result_filtered['same_class_idx'] = same_class_idx_filtered
df_result_filtered['same_class_weight'] = same_class_weight_filtered
df_result_filtered['extract_time'] = extract_time_filtered
df_result_filtered['pairing_time'] = pairing_time_filtered
df_result_filtered['total_bsis_time'] = total_bsis_time_filtered

is_true = list()
total = 0
for q, t in zip(q_class_filtered, most_similar_filtered):
    if t != '':
        if q == img_class.loc[t, 'img_class']:
            total += 1
            is_true.append(1)
        else:
            is_true.append(0)
    else:
        is_true.append(0)
print(total)

df_result_filtered['is_true'] = is_true 
df_result_filtered.to_csv(f'{dataset_name}_{cluster_model.meta.desc_type.value}_{maxsize}_filtered.csv', index=False)


