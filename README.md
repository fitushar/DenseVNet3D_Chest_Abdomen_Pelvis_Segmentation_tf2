# DenseVNet3D_Chest_Abdomen_Pelvis_Segmentation_tf2
This Repo containes the implemnetation of DenseVent in tensorflow 2.0 for chest-abdomen-pelvis (CAP) Segmentation

## Description:
This is a implemnation of the DenseVnet in tensorflow 2.0. DesnVnet(Gibson et al.,"Automatic multi-organ segmentation on abdominal CT with dense V-networks" 2018.) a 3D state-of-art segmentation model for chest-abdomen-pelvis (CAP) Segmentation.

```
    Input
      |
      --[ DFS ]-----------------------[ Conv ]------------[ Conv ]------[+]-->
           |                                       |  |              |
           -----[ DFS ]---------------[ Conv ]------  |              |
                   |                                  |              |
                   -----[ DFS ]-------[ Conv ]---------              |
                                                          [ Prior ]---
```

Reference Implementation:     
* a)https://github.com/baibaidj/vision4med/blob/5c23f57c2836bfabd7bd95a024a0a0b776b181b5/nets/DenseVnet.py
* b)https://niftynet.readthedocs.io/en/dev/_modules/niftynet/network/dense_vnet.html#DenseVNet

## Files:
*   i) `DenseVnet_config.py -|--> All the Netword and Training configuration`
*  ii) `DenseVnet_Loss       |--> Losses and Matrics function. Binary And Multi-class Dice Coefficent and Dice Loss`
* iii) `DenseVnet3D          |--> Network architecture`
*  iv) `Train_DenseVnet3D    |--> Training Script. it has tfrecord decoder, tfdataset reading pipeline and training loop.`

## How to run

To run the model all is to need to configure the `DenseVnet_config.py` based on your requiremnet.
```ruby
###---Number-of-GPU
NUM_OF_GPU=2
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1"]

###----Resume-Training
'''
if want to resume training from the weights Set
RESUME_TRAINING=1
'''
RESUME_TRAINING=0
RESUME_TRAIING_MODEL='/Path/of/the/model/weight/Model.h5'
TRAINING_INITIAL_EPOCH=0

#####-----Configure DenseVnet3D---##########
SEG_NUMBER_OF_CLASSES=31
SEG_INPUT_PATCH_SIZE=(128,160,160, 1)
NUM_DENSEBLOCK_EACH_RESOLUTION=(4, 8, 16)
NUM_OF_FILTER_EACH_RESOLUTION=(12,24,24)
DILATION_RATE=(5, 10, 10)
DROPOUT_RATE=0.25

##Training Hyper-Parameter
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
SEG_LOSS=Avg_Dice_loss
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
SEG_METRICS=Avg_Dice_matrix
BATCH_SIZE=2
TRAINING_STEP_PER_EPOCH=math.ceil((76)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((6)/BATCH_SIZE)
TRAING_EPOCH=10000
NUMBER_OF_PARALLEL_CALL=2
PARSHING=2*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='Model_31_SEG_DenseVnet_April16_2020'
TRAINING_SAVE_MODEL_PATH=''/Path/to/save/model/weight/Model.h5''
TRAINING_CSV='Log_31_SEG_DenseVnet.csv'
LOG_FILE_NAME="Log_31_SEG_DenseVnet." #|lOG FOLDER NAME
SAVE_MODEL_NAME="Org31SEG_DenseVnet_{val_loss:.2f}_{epoch}.h5" #|Model name

#tfrecords--paths
TRAINING_TF_RECORDS='/Training/tfrecords/path/'
VALIDATION_TF_RECORDS='/Val/tfrecords/path/'

```

## Dice Loss
Loss function and matrix are very import. as tensorflow doesn't have a build in multi-class segmentation dice in this project I gave implemened Average Dice-Loss and Dice Matrix. in `DenseVnet_Loss.py` contains the loss and matrix functions. 

```ruby

#|Dice Coefficient for binary Segmentation task
def dice_calculator(mask_true, mask_pred):

    num_sum = 2.0 * tf.keras.backend.sum(mask_true * mask_pred) + tf.keras.backend.epsilon()
    den_sum = tf.keras.backend.sum(tf.keras.backend.square(mask_true)) + tf.keras.backend.sum(tf.keras.backend.square(mask_pred))+ tf.keras.backend.epsilon()
    dise=(num_sum/den_sum)

    return dise
#|Average Dice Loss for multiclass-Segmenatiom task
def Avg_Dice_loss(y_true, y_predicted,num_classes=31):

    clas_dice_list=[]
    for i in range(0,num_classes):
        mask_true = tf.keras.backend.flatten(y_true[:, :, :, :, i])#
        mask_pred = tf.keras.backend.flatten(y_predicted[:, :, :, :, i])#
        class_dice=dice_calculator(mask_true,mask_pred)
        clas_dice_list.append(class_dice)
    avg_dice=tf.math.reduce_mean(clas_dice_list,axis=0)
    dice_loss=1-avg_dice
    return dice_loss

#|Average Dice matricx for multiclass-Segmenatiom task
def Avg_Dice_matrix(y_true, y_predicted,num_classes=31):
    clas_dice_list=[]
    for i in range(0,num_classes):
        mask_true = tf.keras.backend.flatten(y_true[:, :, :, :, i])#
        mask_pred = tf.keras.backend.flatten(y_predicted[:, :, :, :, i])#
        class_dice=dice_calculator(mask_true,mask_pred)
        clas_dice_list.append(class_dice)
    avg_dice=tf.math.reduce_mean(clas_dice_list,axis=0)
    dice_score=tf.print(clas_dice_list[2:9],summarize=10)

    return avg_dice
```

