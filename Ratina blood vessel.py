import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.models import load_model
from numpy.random import randint
from datetime import datetime 
import os
import glob
import cv2
import random
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
seed = 42
np.random.seed = seed


def Discriminator(image_shape):
    initial_weight = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)  
    target_image = Input(shape=image_shape)  
    merged = Concatenate()([input_image, target_image])
    
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=initial_weight)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=initial_weight)(d)
    output = Activation('sigmoid')(d)
    
    model = Model([input_image, target_image], output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5],)
    return model

def encoder_block(layer_in, n_filters, batchnorm=True):
    initial_weight = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    initial_weight = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = LeakyReLU(alpha=0.2)(g)
    return g


def Generator(image_shape=(256,256,3)):
    initial_weight = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)
    e1 = encoder_block(input_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(e7)
    b = LeakyReLU(alpha=0.2)(b)
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=initial_weight)(d7) 
    output_image = Activation('tanh')(g)
    model = Model(input_image, output_image)
    return model

def GAN(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False       
    input_image = Input(shape=image_shape)
    gen_out = g_model(input_image)
    dis_out = d_model([input_image, gen_out])
    model = Model(input_image, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002,beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model
    
def generate_real_samples(dataset, n_samples, patch_shape):
    input_images, target_images = dataset
    index = randint(0, input_images.shape[0], n_samples)
    X1, X2 = input_images[index], target_images[index]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def generate_real_samples(dataset, n_samples, patch_shape):
    input_images, target_images = dataset
    index = randint(0, input_images.shape[0], n_samples)
    X1, X2 = input_images[index], target_images[index]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def summarize_performance(step, g_model, dataset, n_samples=3):
    [input_image, target_image], _ = generate_real_samples(dataset, n_samples, 1)
    fake_image, _ = generate_fake_samples(g_model, input_image, 1)
    input_image = (input_image + 1) / 2.0
    target_image = (target_image + 1) / 2.0
    fake_image = (fake_image + 1) / 2.0
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(input_image[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(fake_image[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i)
        plt.axis('off')
        plt.imshow(target_image[i])
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    
Discrinator_loss = []
GAN_total_loss = []
GEN_L1_loss = []
GEN_gan_loss = []
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    train_X, train_Y = dataset
    bat_per_epo = int(len(train_X) / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        [input_image, target_image], real_label = generate_real_samples(dataset, n_batch, n_patch)
        fake_image, fake_label = generate_fake_samples(g_model, input_image, n_patch)
        d_loss1 = d_model.train_on_batch([input_image, target_image], real_label)
        d_loss2 = d_model.train_on_batch([input_image,fake_image], fake_label)
        d_loss = d_loss1 + d_loss2
        Discrinator_loss.append(d_loss)
        g_loss,gen_gan_loss, gen_l1_loss = gan_model.train_on_batch(input_image, [real_label, target_image])
        GAN_total_loss.append(g_loss)
        GEN_gan_loss.append(gen_gan_loss)
        GEN_L1_loss.append(gen_l1_loss)
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        if (i+1) % (bat_per_epo * 50) == 0:
            summarize_performance(i, g_model, dataset)
            

SIZE_X = 256
SIZE_Y = 256

tar_images = []

for directory_path in glob.glob(r'C:/Users/Asus/OneDrive/Desktop/M.TECH COURSE/Seminar/GAN/CombineData/training/New _Manual/'):
    for img_path in glob.glob(os.path.join(directory_path, "*.tiff")):
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        tar_images.append(img)


tar_images = np.array(tar_images)

src_images = [] 
for directory_path in glob.glob(r'C:/Users/Asus/OneDrive/Desktop/M.TECH COURSE/Seminar/GAN/CombineData/training/New_Images/'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tiff")):
        mask = cv2.imread(mask_path, 1)   
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  
        src_images.append(mask)
        
         
src_images = np.array(src_images)

plt.figure(figsize = (10,10))
n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_images[i])


for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_images[i])


image_shape = src_images.shape[1:]
d_model = Discriminator(image_shape)
g_model = Generator(image_shape)
gan_model = GAN(g_model, d_model, image_shape)


data = [src_images, tar_images]

def preprocess_data(data):
    X1, X2 = data[0], data[1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
    
dataset = preprocess_data(data)

start1 = datetime.now() 
train(d_model, g_model, gan_model, dataset, n_epochs=200, n_batch=1) 

stop1 = datetime.now()
execution_time = stop1-start1
print("Execution time is: ", execution_time)
g_model.save('ratina_blood_vessel_model10.h5')

model = load_model('ratina_blood_vessel_model8.h5')

def plot_images2(src_img, gen_img, tar_img):
    src_img = np.reshape(src_img, (256,256,3))
    gen_img = np.reshape(gen_img, (256,256,3))
    tar_img = np.reshape(tar_img, (256,256,3))
    
    titles = ['Input-image', 'Generated-image', 'Original-image']
    plt.figure(figsize=(18,18))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(src_img)
    plt.title(titles[0])
    
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(gen_img)
    plt.title(titles[1])
    
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(tar_img)
    plt.title(titles[2])
    plt.show()


[X1, X2] = dataset

index = randint(0, len(X1), 1)
src_image, tar_image = X1[index], X2[index]

gen_image = model.predict(src_image)
src_image = (src_image + 1) / 2.0
gen_image = (gen_image + 1) / 2.0
tar_image = (tar_image + 1) / 2.0

plot_images2(src_image,gen_image,tar_image)

def Save_Images(Data,model):
    [X1, X2] = Data
    for index in range(len(X1)):
        src_image, tar_image = X1[[index]], X2[[index]]
        gen_image = model.predict(src_image)
        src_image = (src_image + 1) / 2.0
        gen_image = (gen_image + 1) / 2.0
        tar_image = (tar_image + 1) / 2.0
        
        src_image = np.reshape(src_image, (256,256,3))
        tar_image = np.reshape(tar_image, (256,256,3))
        gen_image = np.reshape(gen_image, (256,256,3))
        
        
        plt.figure(figsize=(30,30))
        plt.axis('off')
        plt.imshow(gen_image)
        filename1 = 'generated_image%d.png' % (index+1)
        plt.savefig(filename1)
        plt.close()
        print("saving image no ",index+1)


Save_Images(dataset,model)

gen_images = model.predict(X1)
tar_images = (X2 + 1)/2.0
gen_images = (gen_images + 1)/2.0

M1 = tf.keras.metrics.FalseNegatives()
M1.update_state(tar_images, gen_images)
FN = M1.result().numpy()
FN

M2 = tf.keras.metrics.FalsePositives()
M2.update_state(tar_images , gen_images)
FP = M2.result().numpy()
FP

M4 = tf.keras.metrics.TruePositives()
M4.update_state(tar_images, gen_images)
TP = M4.result().numpy()
TP

Model_Accuray = (TP+TN)/(TP+TN+FP+FN)
Model_Accuray

Model_Precision = TP/(TP+FP)
Model_Precision 

Model_Sensitivity = TP/(TP+FN)
Model_Sensitivity

Model_Specificity = TN/(TN+FP)
Model_Specificity

test_images = []

for directory_path in glob.glob(r'C:/Users/Asus/OneDrive/Desktop/M.TECH COURSE/Seminar/GAN/CombineData/test/New_Images'):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        test_img = cv2.imread(img_path, 1)
        test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))
        test_images.append(test_img)
test_images = np.array(test_images)


test_mask_images = [] 
for directory_path in glob.glob(r'C:/Users/Asus/OneDrive/Desktop/M.TECH COURSE/Seminar/GAN/CombineData/test/New_Manual'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        test_mask = cv2.imread(mask_path, 1)   
        test_mask = cv2.cvtColor(test_mask,cv2.COLOR_BGR2RGB)
        test_mask = cv2.resize(test_mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST) 
        test_mask_images.append(test_mask)
test_mask_images = np.array(test_mask_images)


test_data = [test_images, test_mask_images]

test_dataset = preprocess_data(test_data)

[tX1, tX2] = test_dataset

ix = randint(0, len(tX1), 1)
test_input_image, test_target_image = tX1[ix], tX2[ix]

test_gen_image = model.predict(test_input_image)

test_input_image = (test_input_image + 1) / 2.0
test_gen_image = (test_gen_image  + 1) / 2.0
test_target_image = (test_target_image + 1) / 2.0


plot_images2(test_input_image,test_gen_image,test_target_image)

generated = model.predict(tX1)
generated = (generated + 1)/2.0
tX2 = (tX2 + 1)/2.0


n1 = tf.keras.metrics.FalseNegatives()
n1.update_state(tX2,generated)
fn = n1.result().numpy()
fn

n1 = tf.keras.metrics.FalseNegatives()
n1.update_state(tX2,generated)
fn = n1.result().numpy()

n2 = tf.keras.metrics.FalsePositives()
n2.update_state(tX2,generated)
fp = n2.result().numpy()

n3 = tf.keras.metrics.TrueNegatives()
n3.update_state(tX2,generated)
tn = n3.result().numpy()

n4 = tf.keras.metrics.TruePositives()
n4.update_state(tX2,generated)
tp = n4.result().numpy()

model_accuracy = (tp+tn)/(tp+tn+fp+fn)
model_accuracy

model_precision = (tp)/(tp+fp)
model_precision

model_sensitivity = (tp)/(tp+fn)
model_sensitivity

model_specificity = (tn)/(tn+fp)
model_specificity