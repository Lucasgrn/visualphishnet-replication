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
from keras.models import load_model
from sklearn.model_selection import train_test_split

print(tf.version)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3800)]  # em MB
    )

# Dataset parameters 
dataset_path = './VisualPhish/'
reshape_size = [224,224,3]
phishing_test_size = 0.4
num_targets = 155
# Model parameters
input_shape = [224,224,3]
margin = 2.2
new_conv_params = [5,5,512]

# Training parameters
start_lr = 0.00002
output_dir = './'
saved_model_name = 'model'
save_interval = 2000
batch_size = 8 #32 era o original
n_iter = 21000
lr_interval = 100



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

# Split phishing to training and test, load train and test indices.
idx_test = np.load(output_dir+'test_idx.npy')
idx_train = np.load(output_dir+'train_idx.npy')
X_test_phish = all_imgs_test[idx_test,:]
y_test_phish = all_labels_test[idx_test,:]
X_train_phish = all_imgs_test[idx_train,:]
y_train_phish = all_labels_test[idx_train,:]

def define_triplet_network(input_shape, new_conv_params):
    
    # Input_shape: shape of input images
    # new_conv_params: dimension of the new convolution layer [spatial1,spatial2,channels] 
    
    # Define the tensors for the three input images
    anchor_input = Input(input_shape)
    positive_input = Input(input_shape)
    negative_input = Input(input_shape)
    
    # Use VGG as a base model 
    base_model = VGG16(weights='imagenet',  input_shape=input_shape, include_top=False)

    x = base_model.output
    x = Conv2D(new_conv_params[2],(new_conv_params[0],new_conv_params[1]),activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(2e-4)) (x)
    x = GlobalMaxPooling2D() (x)
    model = Model(inputs=base_model.input, outputs=x)

    # Generate the encodings (feature vectors) for the two images
    encoded_a = model(anchor_input)
    encoded_p = model(positive_input)
    encoded_n = model(negative_input)
    
    mean_layer = Lambda(lambda x: tf.reduce_mean(x,axis=1)) #mean_layer = Lambda(lambda x: K.mean(x,axis=1))
    
    square_diff_layer = Lambda(lambda tensors:tf.square(tensors[0] - tensors[1])) #square_diff_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    square_diff_pos = square_diff_layer([encoded_a,encoded_p])
    square_diff_neg = square_diff_layer([encoded_a,encoded_n])
    
    square_diff_pos_l2 = mean_layer(square_diff_pos)
    square_diff_neg_l2 = mean_layer(square_diff_neg)
    
    # Add a diff layer
    diff = Subtract()([square_diff_pos_l2, square_diff_neg_l2])
    diff = Reshape((1,)) (diff)

    # Connect the inputs with the outputs
    triplet_net = Model(inputs=[anchor_input,positive_input,negative_input],outputs=diff)
    
    # return the model
    return triplet_net


def custom_loss(margin):
    def loss(y_true,y_pred):
        loss_value = tf.math.maximum(y_true, margin + y_pred)
        loss_value = tf.reduce_mean(loss_value,axis=0)
        return loss_value
    return loss
def loss(y_true,y_pred):
    loss_value = tf.math.maximum(y_true, margin + y_pred)
    loss_value = tf.reduce_mean(loss_value,axis=0)
    return loss_value


model = define_triplet_network(input_shape, new_conv_params)
model.summary()

from keras import optimizers
optimizer = optimizers.Adam(learning_rate= start_lr)
model.compile(loss=custom_loss(margin),optimizer=optimizer)

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

X_test_phish,y_test_phish = order_random_array(X_test_phish,y_test_phish,num_targets)
X_train_phish,y_train_phish = order_random_array(X_train_phish,y_train_phish,num_targets)


# Store the start and end of each target in the phishing set (used later in triplet sampling)
# Not all targets might be in the phishing set 
def targets_start_end(num_target,labels):
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
            start_end_each_target[int(labels[i]),0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]),1] = int(labels.shape[0]-1)
    
    for i in range(1,num_target):
        if start_end_each_target[i,0] == 0:
            start_end_each_target[i,0] = -1
            start_end_each_target[i,1] = -1
    return start_end_each_target

labels_start_end_train_phish = targets_start_end(num_targets,y_train_phish)
labels_start_end_test_phish = targets_start_end(num_targets,y_test_phish)


# Store the start and end of each target in the training set (used later in triplet sampling)
def all_targets_start_end(num_target,labels):
    print(f'O QUE TEM DENTRO DE LABELS:{labels}')
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

labels_start_end_train_legit = all_targets_start_end(num_targets,y_train_legit)

def pick_first_img_idx(labels_start_end,num_targets):
    random_target = -1
    while (random_target == -1):
        random_target = np.random.randint(low = 0,high = num_targets)
        if labels_start_end[random_target,0] == -1:
            random_target = -1
    return random_target

def pick_pos_img_idx(prob_phish,img_label):
    if np.random.uniform() > prob_phish:
        class_idx_start_end = labels_start_end_train_legit[img_label,:]
        same_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
        img = X_train_legit[same_idx,:]
    else:
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
        
#targets names of parent and sub websites
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

def get_batch(batch_size,num_targets):
   
    # initialize 3 empty arrays for the input image batch
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple=[np.zeros((batch_size, h, w,3)) for i in range(3)]

    for i in range(0,batch_size):
        img_idx_pair1 = pick_first_img_idx(labels_start_end_train_legit,num_targets)
        triple[0][i,:,:,:] = X_train_legit[img_idx_pair1,:]
        img_label = int(y_train_legit[img_idx_pair1])
        
        # get image for the second: positive
        triple[1][i,:,:,:] = pick_pos_img_idx(0.15,img_label)
            
        # get image for the thrid: negative from legit
        # don't sample from the same cluster
        img_neg,label_neg = pick_neg_img(img_label,num_targets)
        while check_if_same_category(img_label,label_neg) == 1:
            img_neg,label_neg = pick_neg_img(img_label,num_targets)

        triple[2][i,:,:,:] = img_neg
          
    return triple

def save_keras_model(model):
    model.save(output_dir+saved_model_name+'.h5')
    print("Saved model to disk")
print("Starting training process!")
print("-------------------------------------")

targets_train = np.zeros([batch_size,1])
for i in range(1, n_iter):
    inputs=get_batch(batch_size,num_targets)
    loss_value=model.train_on_batch(inputs,targets_train)
    

    print("\n ------------- \n")
    print('Iteration: '+ str(i) +'. '+ "Loss: {0}".format(loss_value))
    
    if i % save_interval == 0:
        save_keras_model(model)
        
    if i % lr_interval ==0:
        start_lr = 0.99*start_lr
        tf.keras.backend.set_value(model.optimizer.learning_rate, start_lr)

save_keras_model(model)
shared_model = model.layers[3]

whitelist_emb = shared_model.predict(X_train_legit,batch_size=64)
np.save(output_dir+'whitelist_emb',whitelist_emb)
np.save(output_dir+'whitelist_labels',y_train_legit )

phishing_emb = shared_model.predict(all_imgs_test,batch_size=64)
np.save(output_dir+'phishing_emb',phishing_emb)
np.save(output_dir+'phishing_labels',all_labels_test )