import tensorflow as tf
import math
from DenseVnet_Loss import*
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
TRAINING_TF_RECORDS='/image_data/nobackup/Lung_Segmentation_tfrecords_April16_2020/combo/'
VALIDATION_TF_RECORDS='/image_data/nobackup/Lung_Segmentation_tfrecords_April16_2020/Val_SegXCAT_tfrecords/'
