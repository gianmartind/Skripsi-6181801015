#%%
from modules.clustermodel import ClusterModel
from modules import util
from bs4 import BeautifulSoup
import pandas as pd

#%% 
#Fungsi untuk membuat dictionary berisi batas-batas objek target bagi tiap gambar
def create_target(directory, name):
    target = list()
    xml_name = name.split('.')[0] + '.xml'
    with open(f'{directory}/{xml_name}', 'rb') as file:
        ann = BeautifulSoup(file.read(), 'xml')
    object = ann.findAll('object')
    for o in object:
        bndbox = o.find('bndbox')
        target.append({
            'xmin': int(bndbox.find('xmin').string),
            'xmax': int(bndbox.find('xmax').string),
            'ymin': int(bndbox.find('ymin').string),
            'ymax': int(bndbox.find('ymax').string)
        })
    
    return target

#Fungsi untuk menentukan apakah sebuah titik masuk dalam target
def determine(point_x, point_y, targets):
    for t in targets:
        if (point_x >= t['xmin'] and 
            point_x <= t['xmax'] and 
            point_y >= t['ymin'] and 
            point_y <= t['ymax']):
            return True
    else:
        return False

def divide(x, y):
    if y == 0:
        return 0
    else:
        return (x / y)

#Fungsi menghitung score, dengan cara membagi jumlah titik di dalam target dengan total jumlah titik
def calculate_score(row):
    return divide(row['inside_target'], (row['inside_target'] + row['outside_target']))

#%%
directory = 'datasets/annotated_gsv'
images = util.get_dataset(directory)

targets = dict()
for k, v in images.items():
    for img in v:
        img_dir = f'{directory}/{k}'
        targets[img[0]] = create_target(img_dir, img[0])

#targets = {i[0]: create_target(directory, i[0]) for i in images}

#%%
poi_cm = ClusterModel()
poi_cm.load('result/gsv_sift_agglo_400.cm')

#%%
results = {i: list() for i in images.keys()}

for i in range(6, 11):
    for k, v in images.items():
        names = list()
        inside_target = list()
        outside_target = list()

        for name, img in v:
            try: 
                lf = poi_cm.get_local_features(min_uniqueness=i/10, img=name)
                it = 0
                ot = 0
                for l in lf:
                    point_x = l.keypoint.pt[0]
                    point_y = l.keypoint.pt[1]
                    if (determine(point_x, point_y, targets[name])):
                        it += 1
                    else:
                        ot += 1
                
                names.append(name)
                inside_target.append(it)
                outside_target.append(ot)
            except:
                names.append(name)
                inside_target.append(0)
                outside_target.append(0)

        df_score = dict(
            {
                'img': names,
                'inside_target': inside_target,
                'outside_target': outside_target,
            }
        )
        results[k].append(df_score)


# %%
img = list()
img_class = list()
threshold = list()
inside_target = list()
outside_target = list()

for k, v in results.items():
    for t, i in enumerate(v):
        img = img + i['img']
        img_class = img_class + ([k.split('_')[1]] * 10)
        threshold = threshold + [(0.5 + ((t + 1) / 10))] * 10
        inside_target = inside_target + i['inside_target']
        outside_target = outside_target + i['outside_target']

# %%
df_results = pd.DataFrame({
    'img': img,
    'img_class': img_class,
    'threshold': threshold,
    'inside_target': inside_target,
    'outside_target': outside_target
})

df_results['score'] = df_results.apply(calculate_score, axis=1)

