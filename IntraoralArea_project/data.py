from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import cv2
import six
import random

Background = [0,0,0]
IntraoralArea = [255,255,255]
IntraoralArea_1 = [255,0,0]

class_colors = np.array([Background, IntraoralArea])
#class_colors = [(random.randint(0, 255), random.randint(
#    0, 255), random.randint(0, 255)) for _ in range(5000)]

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,n_classes=100,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256), mask_target_size = (256,256), seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = mask_target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        Y = []
        for i in range(mask.shape[0]):
            Y.append(get_segmentation_array(mask[i], n_classes, mask_target_size[0], mask_target_size[1]))
        yield (img, np.array(Y))

def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types) :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: Can't process input type {0}".format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels

class DataLoaderError(Exception):
    pass

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = cv2.imread(os.path.join(test_path,"%d.png"%i), cv2.IMREAD_COLOR)
        #img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = cv2.resize(img, target_size)
        #img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def testGeneratorForImg(img,target_size = (256,256),flag_multi_class = False,as_gray = True):
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

    img = img / 255
    img = cv2.resize(img, target_size)
    #img = trans.resize(img,target_size)
    #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img,(1,)+img.shape)
    yield img

def saveResultForOtherModel(save_path,npyfile, output_width = 264, output_height = 264, n_classes = 100, flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        seg_img = getResultForSingleImg(item, output_width = output_width, output_height = output_height, n_classes = n_classes)
        cv2.imwrite(os.path.join(save_path,"%d_predict.png"%i),seg_img)

def getResultForSingleImg(img, output_width = 264, output_height = 264, n_classes = 100):
    img = img.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (256, 256))
    return seg_img