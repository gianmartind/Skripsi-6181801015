"""
Modul berisi fungsi-fungsi yang akan sering digunakan
"""
#%%
"""
LIBRARIES
"""
import re
import matplotlib.pyplot as plt
import cv2

#untuk memuat file (gambar)
import os
from glob import glob

#untuk pemrosesan data
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import hamming
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

#lain-lain
import time
from itertools import combinations

#%%
def get_image(filename, bw=True, maxheight=None, maxwidth=None):
    """
    Fungsi untuk memuat satu gambar dari file

    Parameter:
        filename    : path dari file gambar
        bw          : apakah gambar hitam putih (grayscale)
        maxheight   : tinggi maksimal dari gambar (default = 400px)
        maxwidth    : lebar maksimal gambar (default = 400px)  
    """
    #load gambar
    img = cv2.imread(filename)

    #convert warna gambar
    if bw:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #ubah ukuran gambar

    if maxheight != None:
        #ukuran awal
        height = img.shape[0]
        width = img.shape[1]

        #cek apakah height > maxheight
        if height > maxheight:
            divider_height = maxheight / height
            new_height = int(height * divider_height)
            new_width = int(width * divider_height)
            img = cv2.resize(img, (new_width, new_height))

    if maxwidth != None:
        #ukuran setelah disesuaikan tingginya
        height = img.shape[0]
        width = img.shape[1]

        #cek apakah width > maxwidth
        if width > maxwidth:
            divider_width = maxwidth / width
            new_height = int(height * divider_width)
            new_width = int(width * divider_width)
            img = cv2.resize(img, (new_width, new_height))
    
    return img

def get_all_images(directory, bw=True, maxwidth=None, maxheight=None):
    images = list()
    for filename in glob(os.path.join(directory, '*.jpg')):
        fname = re.split(r'\\|/', filename)[-1]
        images.append((fname, get_image(filename, bw=bw, maxwidth=maxwidth, maxheight=maxheight)))
    
    return images
        

def get_dataset(directory, bw=True, maxwidth=None, maxheight=None):
    """
    Fungsi untuk mengambil semua gambar dari dataset

    Parameter:
        directory   : path berisi dataset
        bw          : apakah gambar hitam putih (grayscale)
        maxheight   : tinggi maksimal dari gambar (default = 400px)
        maxwidth    : lebar maksimal gambar (default = 400px)  
    """

    imgs = dict()
    directory_ = directory + '/*'
    for dirname in glob(directory_):
        group = re.split(r'\\|/', dirname)[-1]
        for filename in glob(os.path.join(dirname, '*.jpg')):
            fname = re.split(r'\\|/', filename)[-1]
            if group not in imgs.keys():
                imgs[group] = list()
            imgs[group].append((fname, get_image(filename, bw=bw, maxwidth=maxwidth, maxheight=maxheight)))
    
    return imgs

def rename_dataset(directory):
    """
    Fungsi untuk mengubah nama pada file yang tersusun dengan format dataset.
    format nama akan menjadi 'test_{group}_{index}.jpg'

    Parameter:
        directory   : path berisi dataset
    """
    directory_ = directory + '/*'
    for dirname in glob(directory_):
        group = re.split(r'\\|/', dirname)[-1]
        i = 0
        for filename in glob(os.path.join(dirname, '*.jpg')):
            new_fname = 'test_{g}_{i}.jpg'.format(g=group, i=i)
            filepath = filename.split('/')
            filepath[-1] = new_fname
            new_filename = '/'.join(filepath)
            os.rename(filename, new_filename)
            i += 1
        


def agglo_cluster(dataset, n_clusters=None, distance_threshold=None, affinity='euclidean'):
    """
    Fungsi untuk melakukan clustering Agglomerative

    Parameter:
        dataset             : data yang akan di clustering
        n_clusters          : jumlah cluster yang diinginkan
        distance_threshold  : batas untuk memisah cluster
    """
    array_dataset = np.array(dataset.values)
    
    if bool(n_clusters) != bool(distance_threshold):
        if affinity == 'hamming':
            agglo_model = AgglomerativeClustering(
                n_clusters=n_clusters, 
                distance_threshold=distance_threshold,
                affinity=hamming_affinity,
                linkage='complete'
            ).fit(array_dataset)
        elif affinity == 'hamming2':
            agglo_model = AgglomerativeClustering(
                n_clusters=n_clusters, 
                distance_threshold=distance_threshold,
                affinity=hamming_affinity2,
                linkage='complete'
            ).fit(array_dataset)
        else:
            agglo_model = AgglomerativeClustering(
                n_clusters=n_clusters, 
                distance_threshold=distance_threshold,
                affinity=affinity
            ).fit(array_dataset)
        cluster_object = agglo_model.labels_
        return cluster_object
    else:
        raise Exception('n_clusters XOR distance_threshold should return True!')

def dbscan(dataset, eps=0.5, min_pts=5, metric='euclidean'):
    """
        Fungsi untuk melakukan clustering DBSCAN

        Parameter:
            dataset : data yang akan di clustering
            eps     : radius minimum untuk membuat cluster
            min_pts : minimum jumlah elemen pada radius untuk membuat cluster
        """
    array_dataset = np.array(dataset.values)
    if metric == 'hamming':
        dbscan_model = DBSCAN(
            eps=eps, 
            min_samples=min_pts,
            metric=hamming_int
        ).fit(array_dataset)
    else:
        dbscan_model = DBSCAN(
            eps=eps, 
            min_samples=min_pts,
            metric=metric
        ).fit(array_dataset)

    cluster_object = dbscan_model.labels_
    return cluster_object

def euclidean_dist(point1, point2):
    return np.linalg.norm(point1 - point2)

def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

def hamming_dist(i):
    return countSetBits(np.bitwise_xor(int(i[0]), int(i[1])))

def hamming_int2(point1, point2):
    total = 0
    for i, j in zip(point1, point2):
        total += countSetBits(np.bitwise_xor(int(i), int(j)))
    
    return total

def hamming_int(point1, point2):
    arr = np.array([point1, point2])
    return np.sum(np.apply_along_axis(hamming_dist, 0, arr))

def hamming_bin(point1, point2):
    return hamming(point1, point2) * len(point1)

def hamming_affinity(X):
    return pairwise_distances(X, metric=hamming_bin)

def hamming_affinity2(X):
    return pairwise_distances(X, metric=hamming_int2)

def average_distance(dataset, metric, sample=None):
    """
    Fungsi untuk menghitung jarak euclidean rata-rata dari tiap elemen di dataset

    Parameter:
        dataset : data yang akan dihitung rata-rata jaraknya
        sample  : jumlah sampel yang akan digunakan untuk menghitung rata-rata. Jika 'None' maka semua data digunakan
    """
    sample_df = dataset
    if sample != None: 
        sample_df = dataset.sample(sample)               

    dist = 0
    m = 0
    while m < len(sample_df.index) - 1:
        n = m + 1
        while n < len(sample_df.index):
            d = metric(sample_df.iloc[m], sample_df.iloc[n])
            dist = dist + d
            n = n + 1
        m = m + 1
    combination = len(list(combinations(sample_df.index, 2)))
    return dist / combination

def combination_distance(dataset, metric, sample=None):
    sample_df = dataset
    if sample != None: 
        sample_df = dataset.sample(sample).values  
    
    dists = np.array([metric(p1, p2) for p1, p2 in combinations(sample_df, 2)])
    return np.sum(dists) / dists.shape[0]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def int_to_binary(desc):
    """
    Fungsi untuk mengubah descriptor orb yang dihasilkan opencv (32bit integer) menjadi 
    format 256 bit binary
    
    Parameter:
        desc : array berukuran i x 32 berisi integer
    """
    binary_desc = list()
    for row in desc:
        binary_row = list()
        for num in row:
            binary = '{0:08b}'.format(num)
            for i in binary:
                binary_row.append(int(i))
        binary_desc.append(binary_row)

    binary_desc = np.array(binary_desc, dtype='uint8')

    return binary_desc

def binary_to_int(desc):
    int_desc = list()
    for row in desc:
        int_row = list()
        chunked_row = chunks(row, 8)
        for c in chunked_row:
            res = 0
            for ele in c:
                res = (res << 1) | ele
            int_row.append(res)
        int_desc.append(int_row)
    
    int_desc = np.array(int_desc, dtype='uint8')

    return int_desc

def medoid(arr, metric, get_index=False):
    """
    Fungsi untuk mencari medoid dari array berukuran n x n

    Parameter:
        desc    : array 2 dimensi
        metric  : fungsi jarak yang digunakan
    """
    dists = pairwise_distances(arr, metric=metric)

    max = np.zeros(dists.shape[0])
    for i, v in enumerate(dists):
        max[i] = np.sum(v)

    medoid = arr[np.argmin(max)]

    if get_index:
        return np.argmin(max)
    else:
        return np.array([medoid])

def rotate(p, origin=(0, 0), degrees=0):
    """
    Fungsi untuk merotasi suatu koordinat berdasarkan suatu titik pusat

    Parameter:
        p       : koordinat yang akan dirotasi
        origin  : titik pusat untuk melakukan rotasi
        degress : jumlah seberapa besar rotasi
    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rgb_to_bgr(r, g, b):
    return (b, g, r)

def show_keypoints(image, keypoints, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, name=False, return_img=False):
    """
    Fungsi untuk melakukan visualisasi keypoint pada gambar

    Parameter:
        image       : gambar yang digunakan untuk menampilkan
        keypoints   : tuple berisi keypoint
        color       : warna keypoint yang ditampilkan
        name        : nama file jika gambar disimpan
    """
    keypoints_image = cv2.drawKeypoints(image, keypoints, image, color=color, flags=flags)

    if return_img:
        return keypoints_image

    if name:
        cv2.imwrite(name, keypoints_image)
    cv2.imshow('Keypoint', keypoints_image)
    cv2.waitKey(0)
    
def show_matches(img1, kp1, img2, kp2, name=False, return_img=False):
    """
    Fungsi untuk menampilkan pasangan keypoint pada dua gambar
    
    Parameter:
        img1    : gambar pertama
        kp1     : keypoint untuk gambar pertama
        img2    : gambar kedua
        kp2     : keypoint untuk gambar kedua
        name    : nama file jika gambar disimpan
    """
    matches = list()
    for i in range(len(kp1)):
        matches.append(cv2.DMatch(i, i, 0))

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    if return_img:
        return img_matches

    if name:
        cv2.imwrite(name, img_matches)
    cv2.imshow('Keypoint Matches', img_matches)
    cv2.waitKey()

def put_text(img,
             text,
             org,
             font=cv2.FONT_HERSHEY_SIMPLEX,
             fontScale=1,
             color=(0, 255, 0),
             thickness=1):
    """
    Fungsi untuk menambahkan text pada gambar
    """
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img

def show_polygon(img, kp, name=False, return_img=False):
    """
    Fungsi untuk menampilkan polygon yang menyambungkan keypoint-keypoint pada gambar

    Parameter:
        img     : gambar yang digunakan untuk menampilkan
        kp      : tuple berisi keypoint
        name    : nama file jika gambar disimpan
    """
    polygon = np.array([[[i.pt[0], i.pt[1]] for i in kp]], np.int32)
    img_mod = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_mod = cv2.polylines(img_mod, polygon, True, (255, 255, 255), 2)
    for i, p in enumerate(kp):
        img_mod = put_text(img_mod, str(i), (int(p.pt[0]), int(p.pt[1])), fontScale=0.7)

    if return_img:
        return img_mod
    
    if name:
        cv2.imwrite(name, img_mod)
    cv2.imshow('Shapes', img_mod)
    cv2.waitKey()

def display_image(img, cmap='viridis'):
    plt.axis('off')
    plt.imshow(img, cmap=cmap)

"""
Fungsi-fungsi untuk melakukan modifikasi pada gambar, digunakan pada pembuatan data untuk test
"""
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def zoom_center(image, zoom_factor=1.5):
    y_size = image.shape[0]
    x_size = image.shape[1]

    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_factor))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_factor))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_factor))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_factor))

    # first crop image then scale
    img_cropped = image[y1:y2, x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
