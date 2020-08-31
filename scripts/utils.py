import numpy as np
import os, sys, fnmatch
from skimage import io
from skimage.transform import rotate
from skimage.util.shape import view_as_windows
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.layers import Convolution2D, MaxPooling2D, Lambda, Reshape, Flatten, merge
from keras.layers import Input, Concatenate, UpSampling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical, Sequence

##################
# Path functions #
##################

# Collect paths of images recursively
def get_paths(path, img_extension, subfolder=None):
  if subfolder:
    path = os.path.join(path, subfolder)
    
  img_paths = []
  for root, directories, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, "*" + img_extension):
      img_paths.append(os.path.join(root,filename))
  
  return img_paths

# Collects image and mask paths for training
def training_paths(path, img_extension, mask_extension):
  img_paths = get_paths(path, img_extension, subfolder='images')
  mask_paths = get_paths(path, mask_extension, subfolder='masks')
  img_paths.sort()
  mask_paths.sort()
  return img_paths, mask_paths

# Loads images and makes training patches
def get_patches(train_data_path, segmentation_channels, patch_shape, img_extension, mask_extension, workflow_type, patch_thresh=0.05):
  # Gather paths
  img_paths, mask_paths = training_paths(train_data_path, img_extension, mask_extension)
  if len(img_paths) != len(mask_paths):
    sys.exit("Number of images and masks is inconsistent. Maybe your directories are not named 'images' and 'masks'? Exiting... (# images: {} | # masks: {})".format(len(img_paths), len(mask_paths)))
  
  train_images = []
  train_masks = []
  n_images = len(img_paths)
  for ind in range(n_images):
    #if (ind+1) % 2 == 0 or ind == 0:
    print (" >> Generating patches for image", ind+1, "of", str(n_images) + "...")
    print ("    >> Currently at", len(train_images), "total image pairs")
      
  return np.array(train_images, dtype=np.float32), np.array(train_masks, dtype=np.float32)
  
###################
# Image Functions #
###################

def change_dim_order(image, dim_order, dim_default):
  print ("  >> Converting dimension order from " + dim_order + '->' + dim_default)
  if len(image.shape) < len(dim_order):
    print ("  >> The script requires a C dimension of 1... adding C @ dimension " + str(dim_order.index('C')))
    image = np.expand_dims(image, dim_order.index('C'))
  image = np.einsum(dim_order + '->' + dim_default, image)
  return image

def open_image(image_path, dtype=np.float32, channel_dim=True):
  img = io.imread(img_path).astype(dtype)
  if channel_dim and len(img.shape) < 3:
    img = np.expand_dims(img, axis=-1)
  return img

def channel_selector(img, selected_channels):
  n_ch = len(selected_channels)
  new_img = np.zeros((img.shape[:-1] + (n_ch,)), dtype=np.float32)
  try:
    for ch_i in range(n_ch):
      new_img[...,ch_i] = img[...,selected_channels[ch_i]]
  except IndexError:
    sys.exit("Error: Too many channels ({}) selected for an image with {} channels! Exiting...".format(str(selected_channels), img.shape[-1]))
  return new_img
  
def max_norm(img):
  img = new_img
  return img / np.max(img)
  
# def convert_if_categorical(label, workflow_type):
  # if (len(label.shape) == 2 or (label.shape[-1] == 1 and np.max(label) > 1)) and workflow_type=='segmentation':
    # print ("    >> Label looks like grayscale values... converting to categorical representation.")
    # label = to_categorical(label)
    # print ("    >> Number of classes in this label: {}".format(label.shape[-1]))
  # else:
    # label = max_norm(label)
    # if len(label.shape) < 3:
      # label = np.expand_dims(label, axis=-1)
      
  # return label
  
def is_above_thresh(img, threshold):
  # Check if this image has enough positive pixels
  return (img > 0).sum() > (threshold * np.array(img.shape).prod())
   
# Individual augmentations
   
def rotate(img, deg=None):
  if deg and deg % 90 == 0:
    its = int(deg/90)
    for i in range(its):
      img = np.rot90(img)
    return img
  else:
    return img

def mirror(img, order=None):
  if order == 'lr':
    return np.fliplr(img)
  elif order == 'ud':
    return np.flipud(img)
  else:
    return img

# Pairwise Augmentations
    
def rotate_pair(img, label, deg=None):
  return rotate(img, deg), rotate(label, deg)

def mirror_pair(img, label, order=None):
  return mirror(img, order), mirror(label, order)
flip_pair = mirror_pair

def random_crop(img, label, random_crop_size, threshold=None):
  height, width = img.shape[0], img.shape[1]
  dy, dx = random_crop_size
  x = np.random.randint(0, width - dx + 1)
  y = np.random.randint(0, height - dy + 1)
  if threshold:
    for _ in range(10):
      if is_above_thresh(label[y:(y+dy), x:(x+dx)], threshold):
        return img[y:(y+dy), x:(x+dx)], label[y:(y+dy), x:(x+dx)]
      else:
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
  return img[y:(y+dy), x:(x+dx)], label[y:(y+dy), x:(x+dx)]

# Apply random augmentations to image pair

def augment_image_pair(img, label, augmentations=None):
  if not augmentations:
    return img, label
  else:
    aug_fn = np.random.choice(augmentations, 1)[0]
    aug_imgs = [aug_fn(img) for img in imgs]
    return aug_imgs  

# Noise Functions

def guassian_noise(image, mean=0, var=0.1):
  row,col,ch= image.shape
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  noisy = image + gauss
  return noisy

def saltpepper_noise(image, svp=0.5, amount=0.004):
  row,col,ch = image.shape
  out = np.copy(image)
  # Salt mode
  try:
    num_salt = np.ceil(amount * image.size * svp)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - svp))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    out[coords] = 0
  except:
    pass
  
  return out

def poisson_noise(image):
  vals = len(np.unique(image))
  vals = 2 ** np.ceil(np.log2(vals))
  noisy = np.random.poisson(image * vals) / float(vals)
  return noisy
  
def speckle_noise(image):
  row,col,ch = image.shape
  gauss = np.random.randn(row,col,ch)
  gauss = gauss.reshape(row,col,ch)        
  noisy = image + image * gauss
  return noisy

##################
# Data Generator #
##################

class PatchDataGen(Sequence):
  'Generates data for Keras'
  def __init__(self, data_list, label_list, patch_shape, batch_size, 
               segmentation_channels, workflow_type, shuffle=True, preloaded_imgs=None, preloaded_labels=None):
    self.data_list = data_list
    self.label_list = label_list
    self.batch_size = batch_size
    self.shuffle = shuffle
    
    # If Preloaded Data exists...
    self.preloaded_imgs = preloaded_imgs
    self.preloaded_labels = preloaded_labels
    
    self.patch_shape = patch_shape
    self.segmentation_channels = segmentation_channels
    self.workflow_type = workflow_type
    
    self.on_epoch_end()
    
  def __len__(self):
    return int(np.floor(len(self.data_list) / self.batch_size) )

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Generate data
    return self._data_generation(indexes)

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_list))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def _data_generation(self, temp_ids):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    patch_shape = self.patch_shape
    segmentation_channels = self.segmentation_channels
    
    imgs = []
    labels = []
    # Generate data
    for i, id in enumerate(temp_ids):
      # Get sample
      tmp_img_data = self.data_list[id]
      tmp_label_data = self.label_list[id]
      
      # Ex: [img_paths[ind], patch_ind_0, patch_ind_1, 'fliplr']
      tmp_img_path, tmp_patch_ind_0, tmp_patch_ind_1, tmp_augmentation = tmp_img_data
      
      tmp_mask_path = tmp_label_data[0]
      
      # Read image, split only the segmentation_channels, and normalize
      if self.preloaded_imgs is not None:
        img = self.preloaded_imgs[tmp_img_path]
      else:
        img = io.imread(tmp_img_path).astype(np.float32)
        if len(img.shape) < 3:
          img = np.expand_dims(img, axis=-1)
        n_ch = len(segmentation_channels)
        new_img = np.zeros((img.shape[:-1] + (n_ch,))) #, dtype=np.float32)
        for ch_i in range(n_ch):
          new_img[...,ch_i] = img[...,segmentation_channels[ch_i]]
        img = new_img
        img = img / np.max(img)
        
      img_patch_shape = patch_shape + (img.shape[-1],)
      
      # Load masks and normalize
      if self.preloaded_labels is not None:
        mask = self.preloaded_labels[tmp_mask_path]
      else:
        mask = io.imread(tmp_mask_path).astype(np.float32)
        if (len(mask.shape) == 3) and (mask.shape[0] < mask.shape[-1]):
          mask = np.moveaxis(mask, 0, -1)
        
        # If mask is a single channel with grayscale values as labels, convert to channel format
        if (len(mask.shape) == 2 or (mask.shape[-1] == 1 and np.max(mask) > 1)) and self.workflow_type=='segmentation':
          mask = to_categorical(mask)
        else:
          mask = mask / np.max(mask)
          if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=-1)
      mask_patch_shape = patch_shape + (mask.shape[-1],)
      
      # Create windows from both the training images and the masks
      img_patches = view_as_windows(img, img_patch_shape, step=patch_shape[0]//4)
      mask_patches = view_as_windows(mask, mask_patch_shape, step=patch_shape[0]//4)
      img_patches = np.squeeze(img_patches)
      mask_patches = np.squeeze(mask_patches)
      
      # Select the correct patch
      img_patch = np.array(img_patches[tmp_patch_ind_0,tmp_patch_ind_1])
      mask_patch = np.array(mask_patches[tmp_patch_ind_0,tmp_patch_ind_1])

      # Add Channel dim if not included
      if len(img_patch.shape) < 3:
        img_patch = np.expand_dims(img_patch, axis=-1)
      if len(mask_patch.shape) < 3:
        mask_patch = np.expand_dims(mask_patch, axis=-1)

      # Do augmentation if needed
      if tmp_augmentation != '':
        
        # Geometric Augmentations
        if tmp_augmentation[:3] == 'rot':
          img_patch, mask_patch = rotate_pair(img_patch, mask_patch, deg=int(tmp_augmentation[3:]))
        elif tmp_augmentation[:4] == 'flip':
          img_patch, mask_patch = flip_pair(img_patch, mask_patch, order=tmp_augmentation[4:])
        
        # Noise Augmentations
        elif 'noise' in tmp_augmentation:
          if 'gaussian' in tmp_augmentation:
            img_patch = guassian_noise(img_patch)
          elif 'speckle' in tmp_augmentation:
            img_patch = speckle_noise(img_patch)
          elif 'poisson' in tmp_augmentation:
            img_patch = poisson_noise(img_patch)
          elif 'saltpepper' in tmp_augmentation:
            img_patch = saltpepper_noise(img_patch)
        
        # TODO: Add Color Augmentations

      imgs.append(img_patch)
      labels.append(mask_patch)
          
    return np.array(imgs), np.array(labels)
    
#################################
# Calculate Valid Image Patches #
#################################

# Loads images and makes training patches
def calculate_valid_patches_and_stats(train_data_path, segmentation_channels, patch_shape, img_extension, mask_extension, workflow_type, validation_split=0.2, patch_thresh=0.05, augmentations=[]):
  # Gather paths
  img_paths, mask_paths = training_paths(train_data_path, img_extension, mask_extension)
  if len(img_paths) != len(mask_paths):
    sys.exit("Number of images and masks is inconsistent. Exiting... (# images: %d | # masks: %d)".format(len(img_paths), len(mask_paths)))
  
  print (" >> Training augmentations to perform: ", augmentations)
  categorical_notice = True
  input_channels = None
  output_channels = None
  train_images = []
  train_masks = []
  val_images = []
  val_masks = []
  n_images = len(img_paths)
  
  # Image Data Dictionaries
  loaded_imgs = {}
  loaded_masks = {}
  
  # Calculate Validation Images
  n_val_images = int(np.ceil(n_images * validation_split))
  val_inds = list(np.random.choice(n_images, n_val_images, replace=False))
    
  for ind in range(n_images):
    if ind not in val_inds:
      training_flag = True
    else:
      training_flag = False
      
    if (ind+1) % 10 == 0 or ind == 0 or not training_flag:
      print (" >> Calculating valid **", "TRAINING" if training_flag else "VALIDATION", "** patches for image", ind+1, "of", str(n_images) + "...")
      print ("    >> Currently at", len(train_images) if training_flag else len(val_images), "image pairs")
      
    # Read image, split only the segmentation_channels, and normalize
    img = io.imread(img_paths[ind]).astype(np.float32)
    if len(img.shape) < 3:
      img = np.expand_dims(img, axis=-1)
    n_ch = len(segmentation_channels)
    new_img = np.zeros((img.shape[:-1] + (n_ch,)), dtype=np.float32)
    for ch_i in range(n_ch):
      new_img[...,ch_i] = img[...,segmentation_channels[ch_i]]
    img = new_img
    img = img / np.max(img)
    img = (img - np.mean(img)) / np.std(img)
    img_patch_shape = patch_shape + (img.shape[-1],)
    
    # Load masks and normalize
    mask = io.imread(mask_paths[ind]).astype(np.float32)
    if (len(mask.shape) == 3) and (mask.shape[0] < mask.shape[-1]):
      print ("    >> Mask looks like channels-first representation (Shape: {})... converting to channels-last representation (Shape: {}).".format(mask.shape, np.moveaxis(mask, 0, -1).shape))
      mask = np.moveaxis(mask, 0, -1)
    
    # If mask is a single channel with grayscale values as labels, convert to channel format
    if (len(mask.shape) == 2 or (mask.shape[-1] == 1 and np.max(mask) > 1)) and workflow_type=='segmentation':
      mask = to_categorical(mask)
      if categorical_notice:
        print ("    >> Number of classes in this mask: {}".format(mask.shape[-1]))
        print ("    >> Mask looks like grayscale values... converting to categorical representation.")
        categorical_notice = False
    else:
      mask = mask / np.max(mask)
      if len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=-1)
    mask_patch_shape = patch_shape + (mask.shape[-1],)
      
    # Append the images to a dictionary
    loaded_imgs[img_paths[ind]] = img
    loaded_masks[mask_paths[ind]] = mask
    
    # Create windows from both the training images and the masks
    img_patches = view_as_windows(img, img_patch_shape, step=patch_shape[0]//4)
    mask_patches = view_as_windows(mask, mask_patch_shape, step=patch_shape[0]//4)
    img_patches = np.squeeze(img_patches)
    mask_patches = np.squeeze(mask_patches)
    
    # Calculate input and output channels
    if input_channels is None:
      if n_ch >= 1:
        input_channels = n_ch
      else:
        input_channels = 1
    if output_channels is None:
      output_channels = mask_patch_shape[-1]
    
    # Set the threshold number of pixels required to add the training image to the set
    # If at least 5% of the mask patch has true values 
    mask_thresh = patch_thresh * patch_shape[0] * patch_shape[1] if patch_thresh < 1 else patch_thresh
    
    # Iterate over the grid of window views
    for patch_ind_0 in range(img_patches.shape[0]):
      for patch_ind_1 in range(img_patches.shape[1]):
        img_patch = np.array(img_patches[patch_ind_0,patch_ind_1])
        mask_patch = np.array(mask_patches[patch_ind_0,patch_ind_1])
        
        #print([(mask_patch[...,i] > 0).sum() > mask_thresh for i in range(mask_patch.shape[-1]) ] )  
        #print (mask_thresh)
        
        # Only include training images where the mask patch is above the threshold
        if all( [(mask_patch[...,i] > 0).sum() > mask_thresh for i in range(mask_patch.shape[-1])] ) or workflow_type=='image2image':
          
          tmp_img_list = [img_paths[ind], patch_ind_0, patch_ind_1, '']
          tmp_masks_list = [mask_paths[ind], patch_ind_0, patch_ind_1, '']
          
          if training_flag:
            # Append Images w/patch coords and augmentations (only if training) cues to list
            train_images.append(tmp_img_list)
            train_masks.append(tmp_masks_list)
          
            for augment in augmentations:
              tmp_img_list = [img_paths[ind], patch_ind_0, patch_ind_1, augment]
              tmp_masks_list = [mask_paths[ind], patch_ind_0, patch_ind_1, augment]
              train_images.append(tmp_img_list)
              train_masks.append(tmp_masks_list)
              
          else:
            # If validation set, append those images to the correct lists
            val_images.append(tmp_img_list)
            val_masks.append(tmp_masks_list)
    
  return train_images, train_masks, val_images, val_masks, input_channels, output_channels, loaded_imgs, loaded_masks


##########
# Models #
##########

def unet(patch_height, patch_width, in_ch, out_ch, final_act='sigmoid', loss_fn=None):
  inputs = Input((patch_height, patch_width, in_ch))
  conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = BatchNormalization()(conv3)

  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv3), conv2])
  conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1)
  conv4 = BatchNormalization()(conv4)

  up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv1])
  conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up2)
  conv5 = BatchNormalization()(conv5)

  conv7 = Convolution2D(16, (1, 1), activation='relu')(conv5)
  conv7 = BatchNormalization()(conv7)
  conv7 = Dropout(rate=0.33)(conv7)
  last_conv = Convolution2D(out_ch, (1, 1), activation=final_act)(conv7)
  
  model = Model(inputs=inputs, outputs=last_conv)
  if loss_fn:
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_fn, metrics=['accuracy'])
  return model
  
def big_unet(patch_height, patch_width, in_ch, out_ch, final_act='sigmoid', loss_fn=None):
  inputs = Input((patch_height, patch_width, in_ch))
  conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = BatchNormalization()(conv2)
  conv2 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = BatchNormalization()(conv3)
  conv3 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv3a = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool3)
  conv3a = BatchNormalization()(conv3a)
  conv3a = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv3a)
  pool3a = MaxPooling2D(pool_size=(2, 2))(conv3a)
  
  conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool3a)
  conv4 = BatchNormalization()(conv4)
  conv4 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv4)
  
  up1a = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv3a])
  conv5a = Convolution2D(512, (3, 3), activation='relu', padding='same')(up1a)
  conv5a = BatchNormalization()(conv5a)
  conv5a = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5a)
  
  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv5a), conv3])
  conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up1)
  conv5 = BatchNormalization()(conv5)
  conv5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv5)
  
  up1 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv5), conv2])
  conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up1)
  conv6 = BatchNormalization()(conv6)
  conv6 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv6)

  up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv6), conv1])
  conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up2)
  conv7 = BatchNormalization()(conv7)
  conv7 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv7)
  
  conv8 = Convolution2D(32, (1, 1), activation='relu')(conv7)
  conv8 = BatchNormalization()(conv8)
  conv8 = Dropout(rate=0.33)(conv8)
  last_conv = Convolution2D(out_ch, (1, 1), activation=final_act)(conv8)

  model = Model(inputs=inputs, outputs=last_conv)
  if loss_fn:
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_fn, metrics=['accuracy'])
  return model

##########
# Losses #
##########

smooth = 1e-5
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  
def dice_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

def tversky_loss(y_true, y_pred):
  alpha = 0.5
  beta  = 0.5
  ones = K.ones(K.shape(y_true))
  p0 = y_pred      # proba that voxels are class i
  p1 = ones-y_pred # proba that voxels are not class i
  g0 = y_true
  g1 = ones-y_true
  
  num = K.sum(p0*g0, (0,1,2))
  den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
  T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
  
  Ncl = K.cast(K.shape(y_true)[-1], 'float32')
  return Ncl-T

def multi_dice_coef_loss(num_classes, channel_weights=None):
  def _multi_dice_coef_loss(y_true, y_pred):
    loss = 0.
    total = K.sum(K.flatten(y_true)) + smooth
    denominator = 0.
    for i in range(num_classes):
      denominator += total / (K.sum(K.flatten(y_true[...,i])) + smooth)
    
    for i in range(num_classes):
      ratio_i = total / K.sum(K.flatten(y_true[...,i]))
      ratio_i = ratio_i / denominator
      if channel_weights:
        ratio_i = ratio_i * channel_weights[i]
      loss += ratio_i * dice_loss(y_true, y_pred)
    return loss
  return _multi_dice_coef_loss
