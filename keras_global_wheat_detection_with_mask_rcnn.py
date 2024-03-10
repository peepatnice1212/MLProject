#!/usr/bin/env python
# coding: utf-8

# **Version Update [4]**<br>
# You can try the inference [Huggingface Space](https://huggingface.co/spaces/innat/Global.Wheat.Detection.MaskRCNN).
# 
# **Version Update [3]**<br>
# Added **ResNet-101**. It improves score `0.61+ -> 0.63+`.
# 
# **Version Update [2]**<br>
# Added **Weather** augmentation (**Rain/Snow Fall**) using `img_aug` library. It improves the scores. `0.59+ -> 0.61+`. However, the latest `img_aug` library MUST be installed, by default `(0.2.6)`, upgrade to `(0.4)`. It's done in the augmentation section.
# 
# **Version Update [1]**<br>
# There was a bug in data loader. The defined function returned all the images for `train` and `validation`. The function is re-write and OK so far.
# 
# # Global Wheat Detection
# 
# 
# Hi.<br>
# This is a baseline [Matterport](https://github.com/matterport/Mask_RCNN) Keras implementation of **Mask-RCNN** for **Global Wheat Detection** task. 
# 
# ---
# **Please Note**
# 
# I will be using [Matterport](https://github.com/matterport/Mask_RCNN), Inc implementation. Initially I planned to use it in `TF 2.1` but ended up with `TF 1.x` because of compatible error issue. So previously when working on `TF 2.1`, I manually upgrade the necessary scripts of [**Mask-RCNN**](https://github.com/matterport/Mask_RCNN) using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade). But though I am now using `TF 1.x` but still the converted scripts are usable. One can find the upgraded files from here [MaskRCNN Keras Source Code](https://www.kaggle.com/ipythonx/maskrcnn-keras-source-code). In this, we removed some unnecessary example notebooks, unwanted sample images and anything that are not necessary to keep work space neat and clean.
# 
# 
# ---
# 
# ## Content
# * [EDA and Model Config](#1)
#     * [Simple EDA](#1)
#     * [Mask RCNN Model Configuration](#2)
# * [Preparing the Training Set](#3)  
#     * [Mask-RCNN Dataloader](#3)
#     * [Data Split](#4)
# * [Training Sample Visualization](#5)
#     * [Top Mask Position](#5)
#     * [All Mask | Sample with Masked BBox](#6)
# * [Augmentation](#7)
# * [Model Definition and Training || Inference](#8)
# * [Evaluation](#9)
#     * [Visual Evaluation](#9)
#     * [Numerical Evaluation (Comp. Metrics)](#10)
# * [Inference on Test Set](#11)
#     * [Visual Prediction](#11)
#     * [Submission](#12)

# In[2]:


# copy to working directory
# get_ipython().system('cp -r ../input/maskrcnn-keras-source-code/MaskRCNN/* ./')


# **Imports**

# In[3]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os, random, glob, cv2, math

from mrcnn import utils
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config


# In[4]:


# for reproducibility
def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_all(42)
sns.set(style="darkgrid")
# get_ipython().run_line_magic('matplotlib', 'inline')


# # Simple EDA <a id="1"></a>

# In[5]:


ORIG_SIZE     = 1024
epoch         = 100
data_root     = '/kaggle/input'
packages_root = '/kaggle/working'


# In[6]:


# load annotation files
df = pd.read_csv(os.path.join(data_root , 'global-wheat-detection/train.csv'))
df.head()


# In[7]:


# information summary
df.info()


# **Check source distribution**

# In[8]:


plt.figure(figsize=(9,5))
sns.countplot(df.source)
plt.show()


# Organization informed that ` Not all images include wheat heads / bounding boxes.` We can justify that easily by following. There're about 49 image that doesn't have bbox.

# In[9]:


# image directory
img_root = '../input/global-wheat-detection/train/'
len(os.listdir(img_root)) - len(df.image_id.unique())


# Let's modify the annotation file for feasible use. The `bbox` values are in one column, we will make them separate in different attributes.

# In[10]:


df['bbox'] = df['bbox'].apply(lambda x: x[1:-1].split(","))

df['x'] = df['bbox'].apply(lambda x: x[0]).astype('float32')
df['y'] = df['bbox'].apply(lambda x: x[1]).astype('float32')
df['w'] = df['bbox'].apply(lambda x: x[2]).astype('float32')
df['h'] = df['bbox'].apply(lambda x: x[3]).astype('float32')

df = df[['image_id','x', 'y', 'w', 'h']]
df.head()


# # Mask-RCNN Model Configuration <a id="2"></a>

# In[11]:


class WheatDetectorConfig(Config):
    # Give the configuration a recognizable name  
    NAME = 'wheat'
    
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = 'resnet101'
    
    # number of classes (we would normally add +1 for the background)
    # BG + Wheat
    NUM_CLASSES = 2
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120
    
    # Use different size anchors because our target objects are multi-scale (wheats are some too big, some too small)
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    # Learning rate
    LEARNING_RATE = 0.005
    WEIGHT_DECAY  = 0.0005
    
    # Maximum number of ROI’s, the Region Proposal Network (RPN) will generate for the image
    TRAIN_ROIS_PER_IMAGE = 350 
    
    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.60
    
    # Increase with larger training
    VALIDATION_STEPS = 60
    
    # Maximum number of instances that can be detected in one image.
    MAX_GT_INSTANCES = 500 # 200 
 
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0
        }

config = WheatDetectorConfig()
config.display()


# # Data Preparing <a id="3"></a>

# In[12]:


def get_jpg(img_dir, anns):
    '''
    input:
        img_dir: image directory of the train sets
        anns: specified image ids for train or validation
    return:
        img files with specified image ids
    '''
    id      = []
    jpg_fps = []

    for index, row in anns.iterrows():
        id.append(row['image_id'])

    for i in os.listdir(img_dir):
        if os.path.splitext(i)[0] not in id:
            continue
        else:
            jpg_fps.append(os.path.join(img_dir, i))

    return list(set(jpg_fps))

def get_dataset(img_dir, anns): 
    image_fps = get_jpg(img_dir, anns)

    image_annotations = {fp: [] for fp in image_fps}

    for index, row in anns.iterrows(): 
        fp = os.path.join(img_dir, row['image_id'] + '.jpg')
        image_annotations[fp].append(row)

    return image_fps, image_annotations 


# # Data Generator for Mask-RCNN <a id="3"></a>

# In[13]:


class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('GlobalWheat', 1 , 'Wheat') # only one class, wheat
        
        # add images 
        for id, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('GlobalWheat', image_id=id, 
                           path=fp, annotations=annotations, 
                           orig_height=orig_height, orig_width=orig_width)

    # load bbox, most important function so far        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
    
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), 
                            dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count),
                            dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                x = int(a['x'])
                y = int(a['y'])
                w = int(a['w'])
                h = int(a['h'])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                mask[:, :, i] = mask_instance
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)
    
    # simple image loader 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = cv2.imread(fp, cv2.IMREAD_COLOR)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
    
    # simply return the image path
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# # Splits Data Sets <a id="4"></a>

# In[14]:


image_ids = df['image_id'].unique()

valid_ids = image_ids[-700:]
train_ids = image_ids[:-700]

valid_df = df[df['image_id'].isin(valid_ids)]
train_df = df[df['image_id'].isin(train_ids)]
train_df.shape, valid_df.shape


# In[15]:


len(train_df.image_id.unique()), len(valid_df.image_id.unique())


# ## Build Train Set

# In[16]:


# grab all image file path with concern annotation
train_image_fps, train_image_annotations = get_dataset(img_root,
                                                       anns=train_df)

# make data generator with that
dataset_train = DetectorDataset(train_image_fps, 
                                train_image_annotations,
                                ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# ## Build Validation Set

# In[17]:


# grab all image file path with concern annotation
valid_image_fps, valid_image_annotations = get_dataset(img_root, 
                                           anns=valid_df)

# make data generator with that
dataset_valid = DetectorDataset(valid_image_fps, valid_image_annotations,
                                ORIG_SIZE, ORIG_SIZE)
dataset_valid.prepare()

print("Class Count: {}".format(dataset_valid.num_classes))
for i, info in enumerate(dataset_valid.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# # Training Samples <a id="5"></a>
# 
# Using `dataset_train`, let's observe some sample data.

# In[18]:


class_ids = [0]

while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(class_ids)
plt.show()


# # Top Mask Position <a id="5"></a>
# 
# Let's display some sample and corresponding mask (here which is bounding box indicator).

# In[19]:


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids,5)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, 
                                dataset_train.class_names, limit=1)


# # BBoxes with Masked Sample <a id="6"></a>
# 
# In `Mask-RCNN`, the aspect ratio is preserved, though. If an image is not square, then zero padding is added at the `top/bottom` or `right/left`.

# In[20]:


# Load random image and mask.
image_id = np.random.choice(dataset_train.image_ids, 1)[0]
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
original_shape = image.shape

# Resize
image, window, scale, padding, _ = utils.resize_image(image, 
                                                      min_dim=config.IMAGE_MIN_DIM, 
                                                      max_dim=config.IMAGE_MAX_DIM,
                                                      mode=config.IMAGE_RESIZE_MODE)
mask = utils.resize_mask(mask, scale, padding)

# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("Original shape: ", original_shape)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)

# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, 
                            dataset_train.class_names)


# # Augmentation <a id="7"></a>
# 
# Augmentation is the key part to boost performance. Here are some `weather` looking augmentation which are implemented using `img_aug` library. However the current version of `img_aug` is `0.2.6` which needs to upgrade to `0.4` for such augmentation.
# 
# ```
# - Afine Transform
# - Flip
# - Cutout + CoarseDropout
# - Snowflakes
# - Rain
# ```

# In[21]:


# get_ipython().system('pip install ../input/img-aug-v04/imgaug-0.4.0-py2.py3-none-any.whl -q')


# In[22]:


import warnings
from imgaug import augmenters as iaa
warnings.filterwarnings("ignore")

augmentation = iaa.Sequential([
        iaa.OneOf([ ## rotate
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
        ]),

        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),

        iaa.OneOf([ # drop out augmentation
            iaa.Cutout(fill_mode="constant", cval=255),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            ]),

        iaa.OneOf([ ## weather augmentation
            iaa.Snowflakes(flake_size=(0.2, 0.4), speed=(0.01, 0.07)),
            iaa.Rain(speed=(0.3, 0.5)),
        ]),  

        iaa.OneOf([ ## brightness or contrast
            iaa.Multiply((0.8, 1.0)),
            iaa.contrast.LinearContrast((0.9, 1.1)),
        ]),

        iaa.OneOf([ ## blur or sharpen
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.Sharpen(alpha=(0.0, 0.1)),
        ])
    ],
    # do all of the above augmentations in random order
    random_order=True
)


# In[23]:


# from official repo
def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load the image multiple times to show augmentations
limit = 4
ax = get_ax(rows=2, cols=limit//2)

for i in range(limit):
    image, image_meta, class_ids,\
    bbox, mask = modellib.load_image_gt(
        dataset_train, config, image_id, use_mini_mask=False, 
        augment=False, augmentation=augmentation)
    
    visualize.display_instances(image, bbox, mask, class_ids,
                                dataset_train.class_names, ax=ax[i//2, i % 2],
                                show_mask=False, show_bbox=False)


# # Build Model <a id="8"></a>
# 
# Time to build the model. I will use [`mask_rcnn_coco.h5`](https://www.kaggle.com/ipythonx/cocowg) pre-trained model and train the model by initializing with it.

# In[24]:


def model_definition():
    print("loading mask R-CNN model")
    model = modellib.MaskRCNN(mode='training', 
                              config=config, 
                              model_dir=packages_root)
    
    # load the weights for COCO
    model.load_weights(data_root + '/cocowg/mask_rcnn_coco.h5',
                       by_name=True, 
                       exclude=["mrcnn_class_logits",
                                "mrcnn_bbox_fc",  
                                "mrcnn_bbox","mrcnn_mask"])
    return model   

model = model_definition()


# In[25]:


from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger)

def callback():
    cb = []
    checkpoint = ModelCheckpoint(packages_root+'wheat_wg.h5',
                                 save_best_only=True,
                                 mode='min',
                                 monitor='val_loss',
                                 save_weights_only=True, verbose=1)
    cb.append(checkpoint)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3, patience=5,
                                   verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=1, min_lr=0.00001)
    log = CSVLogger(packages_root+'wheat_history.csv')
    cb.append(log)
    cb.append(reduceLROnPlat)
    return cb


# **Inference Configuration**
# 
# I've trained the model on-site. I set `epoch` 100 but the model converged within `50` but later slighty improved in next few more epoch. I didn't have the intention to train longer though. I started the training and went to sleep; next is history. :D

# In[26]:


# get_ipython().run_cell_magic('time', '', "CB = callback()\nTRAIN = False\n\nclass WheatInferenceConfig(WheatDetectorConfig):\n    GPU_COUNT = 1\n    IMAGES_PER_GPU = 1\n\nif TRAIN:\n    model.train(dataset_train, dataset_valid, \n                augmentation=augmentation, \n                learning_rate=config.LEARNING_RATE,\n                custom_callbacks = CB,\n                epochs=epoch, layers='all') \nelse:\n    inference_config = WheatInferenceConfig()\n    # Recreate the model in inference mode\n    model = modellib.MaskRCNN(mode='inference', \n                              config=inference_config,\n                              model_dir=packages_root)\n    \n    model.load_weights(data_root + '/096269-wheat-r101/wheat_096269_101_1024.h5', \n                       by_name = True)\n")


# **Learning Curves**

# In[27]:


history = pd.read_csv(data_root + '/wheatweight/wheat_history.csv') 

# find the lowest validation loss score
print(history.loc[history['val_loss'].idxmin()])
history.head()


# In[28]:


plt.figure(figsize=(19,6))

plt.subplot(131)
plt.plot(history.epoch, history.loss, label="Train loss")
plt.plot(history.epoch, history.val_loss, label="Valid loss")
plt.legend()

plt.subplot(132)
plt.plot(history.epoch, history.mrcnn_class_loss, label="Train class ce")
plt.plot(history.epoch, history.val_mrcnn_class_loss, label="Valid class ce")
plt.legend()

plt.subplot(133)
plt.plot(history.epoch, history.mrcnn_bbox_loss, label="Train box loss")
plt.plot(history.epoch, history.val_mrcnn_bbox_loss, label="Valid box loss")
plt.legend()

plt.show()


# # Evaluation <a id="9"></a>  
# 
# We will evaluate the model performance in both ways: `visual interpretation` and `numerical` or mainly competition metrices (`mAP(0.5:0.75:0.05)`. But I know that most of the cases `visual interpretation` doesn't really matter (except in medical domain). 

# In[29]:


image_id = np.random.choice(dataset_valid.image_ids, 2)

for img_id in image_id:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_valid, inference_config,     
                               img_id, use_mini_mask=False)

    info = dataset_valid.image_info[img_id]
    results = model.detect([original_image], verbose=1)
    r = results[0]

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_valid.class_names, r['scores'], ax=get_ax(), title="Predictions")
    
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)


# # Competition Metrics <a id="10"></a>
# 
# The following functons takes really long amount of time (about an hour) to evaluate the average precision scores withing the given `IoU` threshold scores on the validation set. So, please consider if you want to use it. I will comment out here.

# In[30]:


# get_ipython().run_cell_magic('time', '', '\n# thresh_score = [0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]\n\n# def evaluate_threshold_range(test_set, image_ids, model, \n#                              iou_thresholds, inference_config):\n#     \'\'\'Calculate mAP based on iou_threshold range\n#     inputs:\n#         test_set        : test samples\n#         image_ids       : image ids of the test samples\n#         model           : trained model\n#         inference_config: test configuration\n#         iou_threshold   : by default [0.5:0.75:0.05]\n#     return:\n#         AP : mAP[@0.5:0.75] scores lists of the test samples\n#     \'\'\'\n#     # placeholder for all the ap of all classes for IoU socres 0.5 to 0.95 with step size 0.05\n#     AP = []\n#     np.seterr(divide=\'ignore\', invalid=\'ignore\') \n    \n#     for image_id in image_ids:\n#         # Load image and ground truth data\n#         image, image_meta, gt_class_id, gt_bbox, gt_mask =\\\n#             modellib.load_image_gt(test_set, inference_config,\n#                                    image_id, use_mini_mask=False)\n\n#         # Run object detection\n#         results = model.detect([image], verbose=0)\n#         r = results[0]\n#         AP_range = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, \n#                                           r["rois"], r["class_ids"], r["scores"], r[\'masks\'],\n#                                           iou_thresholds=iou_thresholds, verbose=0)\n        \n#         if math.isnan(AP_range):\n#             continue\n            \n#         # append the scores of each samples\n#         AP.append(AP_range)   \n        \n#     return AP\n\n# AP = evaluate_threshold_range(dataset_valid, dataset_valid.image_ids,\n#                               model, thresh_score, inference_config)\n\n# print("AP[0.5:0.75]: ", np.mean(AP))\n')


# # Inference on Test Set <a id="1"></a>

# In[31]:


def get_jpg(img_dir):
    jpg_fps = glob.glob(img_dir + '*.jpg')
    return list(set(jpg_fps))

# Get filenames of test dataset jpg images
test_img_root  = data_root + '/global-wheat-detection/test/'
test_image_fps = get_jpg(test_img_root)


# # Visual Prediction <a id="11"></a>

# In[32]:


# show a few test image detection example
for image_id in test_image_fps:
    image = cv2.imread(image_id, cv2.IMREAD_COLOR)

    # assume square image 
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1) 

    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    image_id = os.path.splitext(os.path.basename(image_id))[0]

    results = model.detect([resized_image])
    r = results[0]
    for bbox in r['rois']: 
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2] * resize_factor)
        cv2.rectangle(image, (x1,y1), (x2,y2), (77, 255, 9), 3, 1)
        width  = x2 - x1 
        height = y2 - y1 

    plt.figure(figsize=(25,25)) 
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(f"{image_id}.png", bbox_inches='tight', dpi=500)


# # Submission <a id="12"></a>
# 
# Yes bro! Like you, I've also faced stupid `Submission Scoring Error` around `15` times. And when I solved, it felt as same as winning the competition. LoL :D

# In[33]:


# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.50):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    with open(filepath, 'w') as file:
        file.write("image_id,PredictionString\n")

        for image_id in tqdm(image_fps):
            image = cv2.imread(image_id, cv2.IMREAD_COLOR)
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
                
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

            image_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += image_id
            out_str += ","
            
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])
                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                               
                        out_str += ' '
                        out_str += "{0:.4f}".format(r['scores'][i])
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format( x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor )
                        out_str += bboxes_str

            file.write(out_str+"\n")


# In[34]:


submission = os.path.join(packages_root, 'submission.csv')
predict(test_image_fps, filepath=submission)


# In[41]:


submit = pd.read_csv(submission)
submit.head(10)
print(submit['PredictionString'].head(1))

