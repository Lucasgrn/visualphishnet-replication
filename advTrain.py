

import numpy as np
import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten,Subtract,Reshape
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D,MaxPooling2D,Input,Lambda,GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.applications.vgg16 import VGG16
from skimage.io import imsave

from matplotlib.pyplot import imread
from skimage.transform import rescale, resize
import os
from keras import optimizers
from keras.models import load_model

margin = 2.2
start_lr = 0.00002

dataset_path = './VisualPhish/'
reshape_size = [224,224,3]
num_targets = 155 
batch_size = 32 
n_iter = 6000

input_shape = [224,224,3]
saved_model_name = 'model.h5' #from first training 

new_saved_model_name = 'model_adv'
output_dir = './'
save_interval = 1000
lr_interval = 300

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Memória configurada com crescimento dinâmico para {len(gpus)} GPU(s).")


def read_imgs_per_website(data_path,targets,imgs_num,reshape_size,start_target_count):
    all_imgs = np.zeros(shape=[imgs_num,224,224,3])
    all_labels = np.zeros(shape=[imgs_num,1])
    
    all_file_names = []
    targets_list = targets.splitlines()
    count = 0
    for i in range(0,len(targets_list)):
        target_path = data_path + targets_list[i]
        print(target_path)
        file_names = sorted(os.listdir(target_path))
        for j in range(0,len(file_names)):
            try:
                img = imread(target_path+'/'+file_names[j])
                img = img[:,:,0:3]
                all_imgs[count,:,:,:] = resize(img, (reshape_size[0], reshape_size[1]),anti_aliasing=True)
                all_labels[count,:] = i + start_target_count
                all_file_names.append(file_names[j])
                count = count + 1
            except:
                #some images were saved with a wrong extensions 
                try:
                    img = imread(target_path+'/'+file_names[j],format='jpeg')
                    img = img[:,:,0:3]
                    all_imgs[count,:,:,:] = resize(img, (reshape_size[0], reshape_size[1]),anti_aliasing=True)
                    all_labels[count,:] = i + start_target_count
                    all_file_names.append(file_names[j])
                    count = count + 1
                except:
                    print('failed at:')
                    print('***')
                    print(file_names[j])
                    break 
    return all_imgs,all_labels,all_file_names

# Read images legit (train)
data_path = dataset_path + 'trusted_list/'
targets_file = open(data_path+'targets.txt', "r")
targets = targets_file.read()
imgs_num = 9363
all_imgs_train,all_labels_train,all_file_names_train = read_imgs_per_website(data_path,targets,imgs_num,reshape_size,0)

# Read images phishing
data_path = dataset_path + 'phishing/'
targets_file = open(data_path+'targets.txt', "r")
targets = targets_file.read()
imgs_num = 1195
all_imgs_test,all_labels_test,all_file_names_test = read_imgs_per_website(data_path,targets,imgs_num,reshape_size,0)

X_train_legit = all_imgs_train
y_train_legit = all_labels_train

# Load indices of training and test split
idx_train = np.load(output_dir+'train_idx.npy')
idx_test = np.load(output_dir+'test_idx.npy')
X_test_phish = all_imgs_test[idx_test,:]
y_test_phish = all_labels_test[idx_test,:]

X_train_phish = all_imgs_test[idx_train,:]
y_train_phish = all_labels_test[idx_train,:]

def order_random_array(orig_arr,y_orig_arr,targets):
    sorted_arr = np.zeros(orig_arr.shape)
    y_sorted_arr = np.zeros(y_orig_arr.shape)
    count = 0
    for i in range(0,targets):
        for j in range(0,orig_arr.shape[0]):
            if y_orig_arr[j] == i:
                sorted_arr[count,:,:,:] = orig_arr[j,:,:,:]
                y_sorted_arr[count,:] = i
                count = count + 1
    return sorted_arr,y_sorted_arr 

X_test_phish,y_test_phish = order_random_array(X_test_phish,y_test_phish,155)
X_train_phish,y_train_phish = order_random_array(X_train_phish,y_train_phish,155)

#get start and end of each label
def start_end_each_target_not_complete(num_target,labels):
    prev_target = labels[0]
    start_end_each_target = np.zeros((num_target,2))
    start_end_each_target[0,0] = labels[0]
    if not labels[0] == 0:
        start_end_each_target[0,0] = -1
        start_end_each_target[0,1] = -1
    count_target = 0
    for i in range(1,labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[int(labels[i-1]),1] = int(i-1)
            #count_target = count_target + 1
            start_end_each_target[int(labels[i]),0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]),1] = int(labels.shape[0]-1)
    
    for i in range(1,num_target):
        if start_end_each_target[i,0] == 0:
            start_end_each_target[i,0] = -1
            start_end_each_target[i,1] = -1
    return start_end_each_target

labels_start_end_train_phish = start_end_each_target_not_complete(num_targets,y_train_phish)
labels_start_end_test_phish = start_end_each_target_not_complete(num_targets,y_test_phish)


def start_end_each_target(num_target,labels):
    prev_target = 0
    start_end_each_target = np.zeros((num_target,2))
    start_end_each_target[0,0] = 0
    count_target = 0
    for i in range(1,labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[count_target,1] = i-1
            count_target = count_target + 1
            start_end_each_target[count_target,0] = i
            prev_target = prev_target + 1
    start_end_each_target[num_target-1,1] = labels.shape[0]-1
    return start_end_each_target

labels_start_end_train_legit = start_end_each_target(num_targets,y_train_legit)


def custom_loss(margin):
    def loss(y_true,y_pred):
        loss_value = tf.math.maximum(y_true, margin + y_pred)
        loss_value = tf.reduce_mean(loss_value,axis=0)
        return loss_value
    return loss
my_loss = custom_loss(30)

def loss(y_true,y_pred):
    loss_value = tf.math.maximum(y_true, margin + y_pred)
    loss_value = tf.reduce_mean(loss_value,axis=0)
    return loss_value

model = load_model(output_dir+saved_model_name, custom_objects={'loss': loss})
optimizer = optimizers.Adam(lr = start_lr)
model.compile(loss=custom_loss(margin),optimizer=optimizer)
model.summary()
sess = K.get_session()


def pick_first_img_idx(labels_start_end,num_targets):
    random_target = -1
    while (random_target == -1):
        random_target = np.random.randint(low = 0,high = num_targets)
        if labels_start_end[random_target,0] == -1:
            random_target = -1
    class_idx_start_end = labels_start_end[random_target,:]
    img_from_target_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
    return img_from_target_idx

def pick_pos_img_idx(prob_phish,img_label):
    if np.random.uniform() > prob_phish:
        #take image from legit
        class_idx_start_end = labels_start_end_train_legit[img_label,:]
        same_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
        img = X_train_legit[same_idx,:]
    else:
        #take from phish
        if not labels_start_end_train_phish[img_label,0] == -1:
            class_idx_start_end = labels_start_end_train_phish[img_label,:]
            same_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
            img = X_train_phish[same_idx,:]
        else:
            class_idx_start_end = labels_start_end_train_legit[img_label,:]
            same_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
            img = X_train_legit[same_idx,:]
    return img

def pick_neg_img(anchor_idx,num_targets):
    if anchor_idx == 0:
        targets = np.arange(1,num_targets)
    elif anchor_idx == num_targets -1:
        targets = np.arange(0,num_targets-1)
    else:
        targets = np.concatenate([np.arange(0,anchor_idx),np.arange(anchor_idx+1,num_targets)])
    diff_target_idx = np.random.randint(low = 0,high = num_targets-1)
    diff_target = targets[diff_target_idx]
    
    class_idx_start_end = labels_start_end_train_legit[diff_target,:]
    idx_from_diff_target = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
    img = X_train_legit[idx_from_diff_target,:]
    
    return img,diff_target

targets_file = open(data_path+'targets.txt', "r")
all_targets = targets_file.read()
all_targets = all_targets.splitlines()

def get_idx_of_target(target_name,all_targets):
    for i in range(0,len(all_targets)):
        if all_targets[i] == target_name:
            found_idx = i
            return found_idx

target_lists = [['microsoft','ms_outlook','ms_office','ms_bing','ms_onedrive','ms_skype'],['apple','itunes','icloud'],['google','google_drive'],['alibaba','aliexpress']]

def get_associated_targets_idx(target_lists,all_targets):
    sub_target_lists_idx = []
    parents_ids = []
    for i in range(0,len(target_lists)):
        target_list = target_lists[i]
        parent_target = target_list[0]
        one_target_list = []
        parent_idx = get_idx_of_target(parent_target,all_targets)
        parents_ids.append(parent_idx)
        for child_target in target_list[1:]:
            child_idx = get_idx_of_target(child_target,all_targets)
            one_target_list.append(child_idx)
        sub_target_lists_idx.append(one_target_list)
    return parents_ids,sub_target_lists_idx 

parents_ids,sub_target_lists_idx  = get_associated_targets_idx(target_lists,all_targets)

def check_if_same_category(img_label1,img_label2):
    if_same = 0
    if img_label1 in parents_ids:
        if img_label2 in sub_target_lists_idx[parents_ids.index(img_label1)]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[0]:
        if img_label2 in sub_target_lists_idx[0] or img_label2 == parents_ids[0]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[1]:
        if img_label2 in sub_target_lists_idx[1] or img_label2 == parents_ids[1]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[2]:
        if img_label2 in sub_target_lists_idx[2] or img_label2 == parents_ids[2]:
            if_same = 1
    return if_same

# Sample triplets (of normal data)
def get_batch(batch_size,num_targets):
   
    # initialize 3 empty arrays for the input image batch
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple=[np.zeros((batch_size, h, w,3)) for i in range(3)]

    for i in range(0,batch_size):
        img_idx_pair1 = pick_first_img_idx(labels_start_end_train_legit,num_targets)
        triple[0][i,:,:,:] = X_train_legit[img_idx_pair1,:]
        img_label = int(y_train_legit[img_idx_pair1])
        
        #get image for the second: positive
        triple[1][i,:,:,:] = pick_pos_img_idx(0.15,img_label)
            
        #get image for the thrid: negative from legit
        img_neg,label_neg = pick_neg_img(img_label,num_targets)
        while check_if_same_category(img_label,label_neg) == 1:
            img_neg,label_neg = pick_neg_img(img_label,num_targets)

        triple[2][i,:,:,:] = img_neg
          
    return triple

# Generate adv example for one image 
def get_adv_example(triple, epsilon, batch_size):
    # triple = [anchor, positive, negative] -> cada um é um np.array ou tf.Tensor

    anchor = tf.convert_to_tensor(triple[0], dtype=tf.float32)
    positive = tf.convert_to_tensor(triple[1], dtype=tf.float32)
    negative = tf.convert_to_tensor(triple[2], dtype=tf.float32)

    target = tf.zeros([batch_size, 1], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(anchor)  # precisamos rastrear o anchor para derivar sobre ele
        output = model([anchor, positive, negative], training=False)
        loss_val = my_loss(target, output)

    # gradiente em relação ao anchor
    grads = tape.gradient(loss_val, anchor)

    # sinal do gradiente
    delta = tf.sign(grads)

    # ruído adversarial
    anchor_noise = delta.numpy()

    # exemplo adversarial
    anchor_adv = anchor.numpy() + epsilon * delta.numpy()

    return anchor_noise, anchor_adv

# Get batch of adv examples 
def get_batch_adv(batch_size,num_targets):
   
    # initialize 3 empty arrays for the input image batch
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple=[np.zeros((batch_size, h, w,3)) for i in range(3)]

    for i in range(0,batch_size):
        img_idx_pair1 = pick_first_img_idx(labels_start_end_train_legit,num_targets)
        triple[0][i,:,:,:] = X_train_legit[img_idx_pair1,:]
        img_label = int(y_train_legit[img_idx_pair1])
        
        #get image for the second: positive
        triple[1][i,:,:,:] = pick_pos_img_idx(0.15,img_label)
            
        #get image for the thrid: negative from legit
        img_neg,label_neg = pick_neg_img(img_label,num_targets)
        while check_if_same_category(img_label,label_neg) == 1:
            img_neg,label_neg = pick_neg_img(img_label,num_targets)

        triple[2][i,:,:,:] = img_neg
        
    epsilon = np.random.uniform(low=0.003, high=0.01) 
    triple_noise,triple_adv = get_adv_example(triple,epsilon,batch_size)
    triple[0] = triple_adv
    return triple

# Sample two batches (one for adv examples and one for normal images)
def get_two_batches(batch_size,num_targets):
    half_batch = int(batch_size/2)
    triple1 = get_batch(half_batch,num_targets)
    triple2 = get_batch_adv(half_batch,num_targets)
    
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple =  [np.zeros((batch_size, h, w,3)) for i in range(3)]
    
    triple[0][0:half_batch,:] = triple1[0]
    triple[1][0:half_batch,:] = triple1[1]
    triple[2][0:half_batch,:] = triple1[2]

    triple[0][half_batch:batch_size,:] = triple2[0]
    triple[1][half_batch:batch_size,:] = triple2[1]
    triple[2][half_batch:batch_size,:] = triple2[2]
    
    return triple

def save_keras_model(model):
    model.save(output_dir+new_saved_model_name+'.h5')
    print("Saved model to disk")


print("Starting training process!")
print("-------------------------------------")

targets_train = np.zeros([batch_size,1])
for i in range(1, n_iter):
    inputs=get_two_batches(batch_size,num_targets)
    loss_value=model.train_on_batch(inputs,targets_train)
    
    print("\n ------------- \n")
    print('Iteration: '+ str(i) +'. '+ "Loss: {0}".format(loss_value))
    
    if i % save_interval == 0:
        save_keras_model(model)
        
    if i%lr_interval ==0:
        start_lr = 0.99*start_lr
        tf.keras.backend.set_value(model.optimizer.lr, start_lr)


shared_model = model.layers[3]

whitelist_emb = shared_model.predict(X_train_legit,batch_size=64)
np.save(output_dir+'whitelist_emb_adv',whitelist_emb)
np.save(output_dir+'whitelist_labels_adv',y_train_legit )

phishing_emb = shared_model.predict(all_imgs_test,batch_size=64)
np.save(output_dir+'phishing_emb_adv',phishing_emb)
np.save(output_dir+'phishing_labels_adv',all_labels_test )

