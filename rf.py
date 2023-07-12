import os
import cv2
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.utils import img_to_array
from keras.applications.densenet import preprocess_input
from tqdm import tqdm
import pickle

from sklearn import svm



parser = argparse.ArgumentParser()
parser.add_argument('--query_image_path', type=str, default='', help="path of image query to search")
parser.add_argument('--rf', type=bool, default=False, help="use relevance feedback or not")
parser.add_argument('--rf_loop', type=int, default=0, help="number of iterations of relevance feedback")
parser.add_argument('--k_future', type=int, default=100, help="number of samples being labeled in future by active learning")

args = parser.parse_args()


# ====================================================
# VGG19
# ====================================================
class ExtractModel:
    def __init__(self):
        self.model = self.ModelCreator() 

    def ModelCreator(self):
        vgg19_model = VGG19(weights="imagenet")
        extract_model = Model(inputs=vgg19_model.inputs, outputs=vgg19_model.get_layer("fc2").output)
        return extract_model


def preprocessing(img):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def feature_extraction(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = preprocessing(image)
    features = model.predict(img_tensor)[0]
    features = features / np.linalg.norm(features)
    return features


# ==========================================================
# SEARCH ENGINE + RELEVANCE FEEDBACK
# ==========================================================
def search_image(query_image_path, features_db, paths_db, model):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))

    # CNN: VGG19
    query_image_features = feature_extraction(image_path=query_image_path, model=model)
    # print(query_image_features.shape)

    
    distances = np.linalg.norm(features_db - query_image_features, axis=1)
    # K = 50
    K = 100
    indexs = np.argsort(distances)[:K]

    nearest_images = [(features_db[id], paths_db[id], distances[id]) for id in indexs]
    
    return query_image_features, nearest_images


def find_labeled_data(query_image_path, nearest_images):
    class_query_image = query_image_path.split("/")[-2]

    labeled_data = []

    n_pos = 0
    n_neg = 0
    for img in nearest_images:
        paths_img = img[1]
        x_vector = None
        y_label = None
        if class_query_image in paths_img:
            x_vector = img[0]
            y_label = 1
            n_pos += 1
        else:
            x_vector = img[0]
            y_label = 0
            n_neg += 1
        labeled_data.append((x_vector, y_label, img[1]))
    return labeled_data, n_pos, n_neg


def find_unlabeled_data(nearest_images, features_db, paths_db):
    features_db = pickle.load(open(features_db, 'rb'))
    paths_db = pickle.load(open(paths_db, 'rb'))
    
    paths_nearest_img = []
    for img in nearest_images:
        paths_nearest_img.append(img[1])
    
    unlabeled_img_indexs = []
    for i in range(len(paths_db)):
        path = paths_db[i]
        if path not in paths_nearest_img:
            # unlabeled_img_indexs.append(features_db[i])
            unlabeled_img_indexs.append(i)
    return unlabeled_img_indexs


def compute_DS(svc, unlabeled_data_indexs, features_db):
    features_db = pickle.load(open(features_db, 'rb'))
    DS_arr = []
    for i in range(len(unlabeled_data_indexs)):
        idx = unlabeled_data_indexs[i]
        x = features_db[idx]
        x = x.reshape(1, -1)
        dist = abs(svc.decision_function(x))
        # w_norm = np.linalg.norm(svc.coef_)
        # dist = y / w_norm
        DS_arr.append(dist)
    return DS_arr


def compute_DE(svc, query_image_features, unlabeled_data_indexs, features_db):
    features_db = pickle.load(open(features_db, 'rb'))
    DE_arr = []
    for i in range(len(unlabeled_data_indexs)):
        idx = unlabeled_data_indexs[i]
        x = features_db[idx]
        x = x.reshape(1, -1)
        t = svc.decision_function(x)
        if t >= 0:
            dist = np.linalg.norm(x - query_image_features)
        else:
            dist = int(1e9)
        DE_arr.append(dist)
    return DE_arr


def compute_DSE(unlabeled_data_indexs, n_pos, n_neg, DS_arr, DE_arr):
    DSE_arr = []
    for i in range(len(unlabeled_data_indexs)):
        DS_idx = DS_arr[i]
        DE_idx = DE_arr[i]
        dse = (n_pos/(n_pos+n_neg)) * DS_idx + (1-(n_pos/(n_pos+n_neg))) * DE_idx
        DSE_arr.append(dse)
    return DSE_arr


def svm_active_learning(k_future, clf, labeled_data, n_pos, n_neg, unlabeled_data_indexs, query_image_features, query_image_path, nearest_images, features_db, paths_db):

    temp_unlabeled_data_indexs = unlabeled_data_indexs.copy()
    
    # print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")

    X_train = []
    y_train = []
    for d in labeled_data:
        X_train.append(d[0])
        y_train.append(d[1])

    k = k_future
    # define classifier
    clf.fit(X_train, y_train)

    DS_arr = compute_DS(clf, temp_unlabeled_data_indexs, features_db)
    DE_arr = compute_DE(clf, query_image_features, temp_unlabeled_data_indexs, features_db)

    future_labels = []
    for _ in range(k):
        DSE_arr = compute_DSE(temp_unlabeled_data_indexs, n_pos, n_neg, DS_arr, DE_arr)

        DSE_arr = np.array(DS_arr)
        min_dist_index = np.argmin(DSE_arr) # active learning: find the data point closest from boudary

        idx = temp_unlabeled_data_indexs[min_dist_index]
        future_labels.append(idx) # S* set: data to label
        temp_unlabeled_data_indexs.pop(min_dist_index)
        DS_arr.pop(min_dist_index)
        DE_arr.pop(min_dist_index)
    
    return clf, future_labels


def update_nearest_image(clf, query_image_features, query_image_path, old_nearest_images, future_labels, features_db, paths_db):

    paths_db = pickle.load(open(paths_db, 'rb'))
    features_db = pickle.load(open(features_db, 'rb'))

    class_query_image = query_image_path.split("/")[-2]

    images = []
    n_pos = 0
    n_neg = 0

    # classify old nearest images: 1 (relevant), 0 (non-relevant)
    for img in old_nearest_images:
        features_img = img[0]
        paths_img = img[1]
        if class_query_image in paths_img:
            n_pos += 1
            images.append((features_img, paths_img, 1, 1)) # images[i]: (features_vector, path_image, rel/non_rel - 1/0, old/new positive - 1/0)
        else:
            n_neg += 1
            images.append((features_img, paths_img, 0, 1))

    # labeling new labels from svm-active-learning algorithm
    for i in future_labels:
        x = features_db[i]
        x = x.reshape(1, -1)
        y_hat = clf.predict(x)[0]
        if y_hat == 1:
            n_pos += 1
            images.append((features_db[i], paths_db[i], 1, 0))
        else:
            n_neg += 1
            images.append((features_db[i], paths_db[i], 0, 0))


    ds_arr = []
    de_arr = []
    dse_arr = []
    old_postive = []


    # save old positive
    for i in range(len(images)):
        img = images[i]
        rel_or_not = img[2] # 1 (rel), 0 (non-rel)
        old_or_not = img[3] # 1 (old), 0 (new)
        if rel_or_not == 1 and old_or_not == 1: # old positive
            old_postive.append(i)

    # compute DS
    for img in images:
        features = img[0]
        features = features.reshape(1, -1)
        path = img[1]
        dist = abs(clf.decision_function(features))
        # w_norm = np.linalg.norm(svc.coef_)
        # dist = y / w_norm
        ds_arr.append(dist)

    # compute DE
    for img in images:
        features = img[0]
        features = features.reshape(1, -1)
        # t = clf.decision_function(features)
        if img[2] == 1:
            dist = np.linalg.norm(features - query_image_features)
        else:
            dist = int(1e9)
        de_arr.append(dist)

    # compute DSE
    for i in range(len(images)):
        if i in old_postive:
            alpha = 1/4
        else:
            alpha = 4
        DS_idx = ds_arr[i]
        DE_idx = de_arr[i]
        dse = 0.3 * DS_idx + 0.7 * DE_idx
        # dse = (n_pos/(n_pos+n_neg)) * DS_idx + (1-(n_pos/(n_pos+n_neg))) * DE_idx
        dse = dse * alpha # ensure that old positive will be presented at first
        dse_arr.append(dse)

    dse_arr = np.array(dse_arr)
    dse_arr = dse_arr.reshape(-1)
    K = 100
    indexs = np.argsort(dse_arr)[:K]

    nearest_images = [(images[id][0], images[id][1], dse_arr[id]) for id in indexs]
    return nearest_images


def update_current_labeled_data(query_image_path, current_labeled_data, current_n_pos, current_n_neg, new_nearest_images):
    temp_labeled_data_set, temp_n_pos, temp_n_neg = find_labeled_data(query_image_path, new_nearest_images)

    arr_path_current_labeled_data = []
    for labeled_data in current_labeled_data:
        path = labeled_data[2]
        arr_path_current_labeled_data.append(path)

    counter = 0
    for temp_labeled_data in temp_labeled_data_set:
        pos_or_neg = temp_labeled_data[1]
        path_temp_labeled_data = temp_labeled_data[2]
        if path_temp_labeled_data not in arr_path_current_labeled_data:
            current_labeled_data.append(temp_labeled_data)
            if pos_or_neg == 1:
                current_n_pos += 1
            elif pos_or_neg == 0:
                current_n_neg += 1
            counter += 1
    # print(counter)
    return current_labeled_data, current_n_pos, current_n_neg


def update_current_unlabeled_data_indices(paths_db, current_labeled_data, current_unlabeled_data_indices):
    # print(f"==== {len(current_unlabeled_data_indices)}")
    paths_db = pickle.load(open(paths_db, 'rb'))

    for labeled_data in current_labeled_data:
        path = labeled_data[2]
        paths_db_index = paths_db.index(path)
        if paths_db_index in current_unlabeled_data_indices:
            current_unlabeled_data_indices.remove(paths_db_index)
    return current_unlabeled_data_indices


def plot_result(nearest_images, n_pos):
    """
    PLOT
    """
    # grid_size = int(math.sqrt(K))
    grid_row = 5
    grid_col = 20
    fig, axes = plt.subplots(grid_row, grid_col, figsize=(15, 8))
    k = 0
    for i in range(grid_row):
        for j in range(grid_col):
            if i == 0 and j == 0:
                axes[i, j].set_title(f"Accuracy: {n_pos}%")
                image = cv2.imread(query_image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(image)
                axes[i, j].axis('off')
                k += 1
            else:
                features_vector, file_path, distance = nearest_images[k-1]
                # axes[i, j].set_title(distance)
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(image)
                axes[i, j].axis('off')
                k += 1
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
   

    vgg19_model = ExtractModel().model
    # vgg19_model.summary()

    DATA_FOLDER = 'RF_Active_SVM/db/Corel/'

    # Corel
    features_db_file = "db/features/features.pkl"
    file_path_db_file = "db/features/paths.pkl"

    query_image_path = args.query_image_path
    if query_image_path == '':
        print("Input query is empty")
    rf = args.rf
    rf_loop = args.rf_loop
    k_future = args.k_future
    if not rf:
        query_image_features, nearest_images = search_image(query_image_path=query_image_path,features_db=features_db_file, paths_db=file_path_db_file, model=vgg19_model)
        labeled_data_set, n_pos, n_neg = find_labeled_data(query_image_path, nearest_images)
        plot_result(nearest_images, n_pos)
    else:
        kernel = 'rbf'
        clf = svm.SVC(kernel=kernel)

        if rf_loop == 0:
            rf_loop = 1
        if k_future < 100:
            k_future = 100

        query_image_features, nearest_images = search_image(query_image_path=query_image_path,features_db=features_db_file, paths_db=file_path_db_file, model=vgg19_model)

        labeled_data_set, n_pos, n_neg = find_labeled_data(query_image_path, nearest_images)
        plot_result(nearest_images, n_pos)
        
        unlabeled_data_set_indices = find_unlabeled_data(nearest_images, features_db=features_db_file, paths_db=file_path_db_file)


        for i in range(rf_loop):
            print(f"====> RF {i+1}:")
            clf, future_labels = svm_active_learning(k_future, clf, labeled_data_set, n_pos, n_neg, unlabeled_data_set_indices, query_image_features, query_image_path, nearest_images, features_db_file, file_path_db_file)
            nearest_images = update_nearest_image(clf, query_image_features, query_image_path, nearest_images, future_labels, features_db_file, file_path_db_file)
            labeled_data_set, n_pos, n_neg = update_current_labeled_data(query_image_path, labeled_data_set, n_pos, n_neg, nearest_images)
            unlabeled_data_set_indices = update_current_unlabeled_data_indices(file_path_db_file, labeled_data_set, unlabeled_data_set_indices)
            plot_result(nearest_images, n_pos)
        
        print(f"n_pos : {n_pos} ====== n_neg : {n_neg}")