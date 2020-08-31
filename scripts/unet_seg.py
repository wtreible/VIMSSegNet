from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import argparse
import numpy as np

########################
# Arguments and Params #
########################
def get_args(valid_modes):
  parser = argparse.ArgumentParser(description='Unet Driver')
  ###
  parser.add_argument("--seed", type=int, dest='seed', default=1337)
  ###
  parser.add_argument("-m", "--mode", type=str, choices=valid_modes, required=True,
                      help="Enter a run mode for the script")
  ###
  parser.add_argument("-i", "--input-path", type=str, dest='input_path', required=True,
                      help="Path to training or prediction data folder")
  ###
  parser.add_argument("-c", "--seg-channels", type=str, dest='seg_channels', default="0,1,2",
                      help="Channels of input image to segment. Specified as a commas separated list of channels (Default: 0,1,2 [RGB])")
  ###
  parser.add_argument("-e", "--image-ext", type=str, dest='input_img_extension', default='.tif',
                      help="Extension of the training images (Default: .tif)")
  
  # Inference Specific Params
  ###
  parser.add_argument("--weights-path", type=str, dest='weights_path', 
                      help="Path to trained network weights")
  ###
  parser.add_argument("--output-path", type=str, dest='output_path', 
                      help="Path to save segmented images")
  ###
  parser.add_argument("--num-classes", type=int, dest='num_classes', default=1,
                      help="Number of classes to segment (Default: 1 [Binary Segmentation])")
  ###
  parser.add_argument("--output-ext", type=str, dest='output_img_extentsion', default='.tif',
                      help="Extension of the output image (Default: .tif)")
  ###
  parser.add_argument("--dim-order", type=str, dest='dim_order', default='TXYC',
                      help="Dimension ordering for input images. (Default: \'XYC\' for single images and \'TXYC\' for time series images)" )
                      
  # Training Specific Params
  ### 
  parser.add_argument("--checkpoint-directory", type=str, dest='checkpoint_dir', default='checkpoints',
                      help="Directory to save weight checkpoints (Default: checkpoints)")
  ###
  parser.add_argument("--epochs", type=int, dest='epochs', default=30,
                      help="Number of epochs for training (Default: 30)")
  ###
  parser.add_argument("--batch-size", type=int, dest='batch_size', default=32,
                      help="Size of a batch of patches to be run through the network. *Reduce* this if you encounter GPU memory problems. (Default: 32)")
  ###
  parser.add_argument("--patch-size", type=int, dest='patch_size', default=256,
                      help="Size of the dimensions of square patches for training. *Decrease* this if you encounter GPU memory problems. (Default: 256)")
  ###
  parser.add_argument("--patch-thresh", type=float, dest='patch_thresh', default=0.05,
                      help="Patches will only be kept if at least <patch_thresh> percent of the corresponding mask contains true pixels (Default: 0.05)")
  ###
  parser.add_argument("--mask-ext", type=str, dest='train_mask_extension', default='.tif',
                      help="Extension of the training masks (Default: .tif)")
  ###
  parser.add_argument("--workflow-type", type=str, dest='workflow_type', default='segmentation',
                      help="Type of Unet workflow. \'segmentation\' (Default) or \'image2image\'")
  ###
  parser.add_argument("--no-shuffle", action='store_true',  dest='no_shuffle', 
                      help="Don't shuffle training data.")
  ###
  parser.add_argument("--augmentations", type=str, dest='augmentations', default='flipud,fliplr,rot90,rot180,rot270',
                      help="Augmentations for training data. (Default: \'flipud,fliplr,rot90,rot180,rot270\')")
  ###
  return parser.parse_args()

# Parse inference params
valid_modes = ['train', 'predict', 'predict_ts']
args = get_args(valid_modes)
mode = args.mode

# Random Seed
seed = args.seed
np.random.seed(seed)

# General Params
input_img_extension = args.input_img_extension
input_path = args.input_path
segmentation_channels = [int(ch) for ch in args.seg_channels.split(',')]
workflow_type = args.workflow_type if args.workflow_type in ['segmentation', 'image2image'] else 'segmentation'
if workflow_type == 'image2image':
  segmentation_channels = [0]

# Training Params
checkpoint_dir = args.checkpoint_dir
epochs = args.epochs
batch_size = args.batch_size
patch_size = args.patch_size
patch_thresh = args.patch_thresh
train_mask_extension = args.train_mask_extension
shuffle = not args.no_shuffle
valid_augmentations = ['flipud', 'fliplr', 'rot90', 'rot180', 'rot270', 'gaussian_noise','speckle_noise', 'saltpepper_noise']
augmentations = [str(aug) for aug in args.augmentations.split(',') if aug in valid_augmentations]

# Inference Params
weights_path = args.weights_path
output_path = args.output_path
output_image_extension = args.output_img_extentsion
model_output_channels = args.num_classes
dim_order = args.dim_order.upper()
default_dim_order = 'TXYC'

print ("====================")

#############################
# Import Required Libraries #
#############################
import os, sys
from datetime import datetime
from skimage import io
from utils import *
  
####################
# Model Parameters #
####################
MODEL = big_unet
if workflow_type == 'segmentation':
  FINAL_ACT = 'sigmoid'
  LOSS = tversky_loss
elif workflow_type == 'image2image':
  FINAL_ACT = 'relu'
  LOSS = 'mean_absolute_error'

###############################
# Print Command for Debugging #
###############################
print ("Input Command:")
print (" ".join(sys.argv))
print ("====================")

############
# Training #
############
if mode == 'train':

  # Load Training Images
  patch_shape = (patch_size, patch_size)
  print ("Calculating valid Training Patches...")
  data_list, label_list, val_images, val_masks, in_ch, out_ch, preloaded_imgs, preloaded_labels = calculate_valid_patches_and_stats(input_path, 
                                                                                                                                    segmentation_channels, 
                                                                                                                                    patch_shape, 
                                                                                                                                    input_img_extension, 
                                                                                                                                    train_mask_extension,
                                                                                                                                    workflow_type=workflow_type,
                                                                                                                                    patch_thresh=patch_thresh,
                                                                                                                                    augmentations=augmentations)
  print ("Calculated Valid Patches!")
  
  # Create Data Generators
  print ("====================")
  print ("Creating Data Generators!")
  train_datagen = PatchDataGen(data_list=data_list, 
                               label_list=label_list, 
                               patch_shape=patch_shape, 
                               batch_size=batch_size, 
                               segmentation_channels=segmentation_channels, 
                               workflow_type=workflow_type, 
                               shuffle=shuffle,
                               preloaded_imgs=preloaded_imgs,
                               preloaded_labels=preloaded_labels)
                               
  val_datagen   = PatchDataGen(data_list=val_images, 
                               label_list=val_masks, 
                               patch_shape=patch_shape, 
                               batch_size=batch_size, 
                               segmentation_channels=segmentation_channels, 
                               workflow_type=workflow_type, 
                               shuffle=shuffle,
                               preloaded_imgs=preloaded_imgs,
                               preloaded_labels=preloaded_labels)
  
  print ("Initialized Data Generators!")
  
  # Setup Checkpoint Path
  print ("====================")
  print ("Setting up checkpoint callback and checkpoint directory...")
  if not os.path.isdir(checkpoint_dir):
    print (" >> checkpoint_directory doesn't exist; attempting to create it...")
    try:
      os.makedirs(checkpoint_dir)
      print ("   >> Created checkpoint_directory:", checkpoint_dir)
    except:
      sys.exit("Invalid checkpoint_dir path! Exiting...")
  date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
  weights_filename = date_str + '_E{epoch:03d}_L{val_loss:.4f}'
  weights_filename += '_in' + str(in_ch) + '_out' + str(out_ch) + '.h5'
  checkpoint_path = os.path.join(checkpoint_dir, weights_filename)
  print (" >> Checkpoint filename format:", weights_filename)
  
  # Define Model and Model Checkpoint Callback
  print ("====================")
  print ("Creating model and training...")
  print (" >> Model input channels:", in_ch, "| Model output channels:", out_ch)
  print (" >> Batch shape:", (batch_size,) + patch_shape + (in_ch,))
  
  model = MODEL(patch_size, patch_size, in_ch=in_ch, out_ch=out_ch, final_act=FINAL_ACT, loss_fn=LOSS)
  ckpt = ModelCheckpoint(str(checkpoint_path), verbose=2, monitor='val_loss', mode='auto', save_best_only=True)
  print ("Model created!")
  print ("====================")
    
  # Train the Model
  model.fit_generator(train_datagen, 
                      steps_per_epoch=int(np.ceil(len(data_list)/batch_size)), 
                      epochs=epochs, 
                      verbose=1, 
                      callbacks=[ckpt], 
                      validation_data=val_datagen, 
                      validation_steps=int(np.ceil(len(val_images)/batch_size)), 
                      shuffle=False, 
                      initial_epoch=0)

  print ("====================")
  print ("Training Completed!")

#############
# Inference #
#############
elif mode == 'predict' or mode == 'predict_ts':

  # Gather inference image filenames
  print ("====================")
  print ("Gathering input image paths...")
  inference_image_files = get_paths(input_path, input_img_extension) 
  num_image_files = len(inference_image_files)
  print (" >> Total number of images:", num_image_files)
  print ("Paths gathered!")
  
  # If time series, set the time_series flag
  time_series = mode == 'predict_ts'
  
  print ("====================")
  print ("Setting up output path...")
  if os.path.isdir(output_path):
    print (" >> Output directory exists; this may overwrite data!")
  else:
    print (" >> Output directory does not exist; attempting to create it...")
    try:
      os.makedirs(output_path)
      print (" >> Output directory created!")
    except:
      pass
  print ("Output path set up!")
  
  # Load one image and compute shape to initialize network
  print ("====================")
  print ("Calculating image size and pad amount...")
  mult16 = [16*i for i in range(512)]
  init_img = io.imread(inference_image_files[0]).astype(np.float32)
  if len(init_img.shape) < 3:
    init_img = np.expand_dims(init_img, axis=-1)
  original_image_shape = init_img.shape if not time_series else init_img.shape[1:]
  print (" >> Initial image xy shape:", original_image_shape[:2])
  if original_image_shape[0] > mult16[-1] or original_image_shape[1] > mult16[-1]:
    sys.exit("Image shape too large! Maximum dimension is %d. Exiting...".format(mult16[-1]))
  if original_image_shape[0] not in mult16:
    y_ind = next(x[0] for x in enumerate(mult16) if x[1] > original_image_shape[0])
    new_y_sz = mult16[y_ind]
    pad_y = new_y_sz - original_image_shape[0]
  else:
    new_y_sz = original_image_shape[0]
    pad_y = 0
  if original_image_shape[1] not in mult16:
    x_ind = next(x[0] for x in enumerate(mult16) if x[1] > original_image_shape[1])
    new_x_sz = mult16[x_ind]
    pad_x = new_x_sz - original_image_shape[1]
  else:
    new_x_sz = original_image_shape[1]
    pad_x = 0
  if pad_x > 0 or pad_y > 0:
    print (" >> New image xy shape:", (new_y_sz, new_x_sz))
  print ("Image size and padding calculated and image padded!")
  
  # # Load normalization parameters
  # print ("====================")
  # print ("Loading normalization parameters...")
  # means = []
  # stds = []
  # with open(norm_path, 'r') as fp:
    # for line in fp.readlines():
      # tmp_mean, tmp_std = line.strip().split(',')
      # means.append(float(tmp_mean))
      # stds.append(float(tmp_std))
  # if len(means) != len(segmentation_channels):
    # print (" >> Number of means/stds:", len(means), "| Number of segmentation channels:", len(segmentation_channels))
    # sys.exit("Error: Number of normalization parameters do not match the number of segmentation channels... Exiting...")
  # print ("Normalization parameters loaded!")
  
  # Create model and load weights
  print ("====================")
  print ("Creating model and loading weights...")
  print (" >> Model input shape:", (new_y_sz, new_x_sz))
  print (" >> Model input channels:", len(segmentation_channels), "| Model output channels:", model_output_channels)
  model = MODEL(new_y_sz, new_x_sz, in_ch=len(segmentation_channels), out_ch=model_output_channels, final_act=FINAL_ACT)
  model.load_weights(weights_path)
  print ("Model created!")

  def infer_on_image(img, output_name, time_point=None):
    print ("  ===")
    print ("  Begin image inference...")
    if time_point:
      print ("   >> Time Point: " + str(time_point) )
    print ("  Splitting input image into only segmentation channels and normalizing...")
    # Give image a single channel dimension if needed
    #if len(img.shape) < 3:
    #  print ("   >> No channel dimension, expanding image to have 1 channel...")
    #  img = np.expand_dims(img, axis=-1)
        
    # Split by segmentation_channels and normalize
    print ("   >> Segmentation channels:", segmentation_channels)
    print ("   >> Input image channel dimension:", img.shape[-1])
    n_ch = len(segmentation_channels)
    if n_ch < img.shape[-1]:
      new_img = np.zeros((img.shape[:-1] + (n_ch,)), dtype=np.float32)
      for ch_i in range(n_ch):
        new_img[...,ch_i] = img[...,segmentation_channels[ch_i]]
      img = new_img
    elif n_ch != 1:
      print ("   >> Warning: Too many segmentation_channels given! Ignoring segmentation_channels and using the entire input image...")
    
    # Normalize Image
    inp_img = np.array(img)
    img = img / np.max(img)
    img = (img - np.mean(img)) / np.std(img)
    print ("  Image split and normalized!")
    norm_img = np.array(img)
    
    # Pad image size to be multiple of 16 for inference
    if pad_x > 0 or pad_y > 0:
      print ("  ---")
      print ("  Padding image so that its xy dimensions are multiples of 16...")
      print ("   >> Initial image shape:", img.shape)
      img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), 'constant')
      print ("   >> Padded image shape:", img.shape)
    
    # Give image a batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict the segmentation of the image and save
    print ("  ---")
    print ("  Predicting segmentation of image...")
    tmp_seg_img = model.predict(img)
    print ("  Prediction complete!")
  
    # Resize output image to the correct shape
    print ("  ---")
    print ("  Resizing output segmentation...")
    seg_img = tmp_seg_img.reshape((img.shape[1], img.shape[2], model_output_channels))
    
    # Set output path and save (save image as float 16)
    print ("  ---")
    print ("  Saving image...")
    if time_series:
      print ("   >> Saving time point image as frames in directory...")
      ts_output_dir = os.path.join(output_path, output_name)
      print ("   >> Directory of frame images:", ts_output_dir)
      if not os.path.isdir(ts_output_dir):
        print ("   >> Directory doesn't exist... creating!")
        try:
          os.makedirs(ts_output_dir)
        except:
          pass
      elif os.path.isdir(ts_output_dir) and time_point == 0:
        print ("   >> Directory exists on first time point... this may overwrite files!")
      output_file_name = "t" + str(time_point) + "_out" + output_image_extension
      output_file_path = os.path.join(ts_output_dir, output_file_name)
    else:
      output_file_name = output_name + "_out" + output_image_extension
      output_file_path = os.path.join(output_path, output_file_name)
    
    # Reshape image to original size
    print ("  ---")
    print ("  Resizing image to original unpadded shape...")
    seg_img_unpadded = seg_img[:original_image_shape[0],:original_image_shape[1]]
    print ("   >> Padded image shape:", seg_img.shape, "| New shape:", seg_img_unpadded.shape)
    
    
    tmp_for_conv = seg_img_unpadded.astype(np.float32)
    io.imsave(output_file_path, tmp_for_conv) #(tmp_for_conv/np.max(tmp_for_conv) * 255.).astype(np.uint8) )
    print ("  Image saved to:", output_file_path)
    

  # Loop over all images...
  for image_i, inference_image_path in enumerate(inference_image_files):
  
    # Get inference image path, load image, and normalize
    print ("====================")
    print ("Working on image:", inference_image_path)
    print (" >> Image ", image_i+1, "/", num_image_files)
    img = io.imread(inference_image_path).astype(np.float32)
    print ("  >> Original image shape:", img.shape)
    
    output_name = inference_image_path.replace(" ","_").split("/")[-1]
    output_name = output_name[:-len(input_img_extension)]
    print ("  >> Output image name:", output_name)
    
    if time_series:
      print("  >> Reading this image as " + dim_order + "... ")
      img = change_dim_order(img, dim_order, default_dim_order)
      img_shp = img.shape
      
      print("  >> (T)ime dimension is " + str(img_shp[0]) )
      print("  >> (X) and (Y) dimensions are " + str(img_shp[1]) + " and " + str(img_shp[2]) )
      print("  >> (C)hannel dimension is " + str(img_shp[3]) )
      for time_point in range(img.shape[0]):
        infer_on_image(img[time_point], output_name, time_point=time_point)
    else:
      print("  >> Reading this image as " + dim_order + "... ")
      dim_order = dim_order.replace('T', '')
      default_dim_order = default_dim_order.replace('T', '')
      img = change_dim_order(img, dim_order, default_dim_order)
      img_shp = img.shape
      print("  >> (X) and (Y) dimensions are " + str(img_shp[0]) + " and " + str(img_shp[1]) )
      print("  >> (C)hannel dimension is " + str(img_shp[2]) )
      infer_on_image(img, output_name)
