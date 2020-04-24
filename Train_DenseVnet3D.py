from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

'''
@Author: Fakrul Islam Tushar,
RA Duke University Medical Center
4/23/2020, NC,USA.
ft42@duke.edu,f.i.tushar.eee@gmail.com
'''

'''
tf.config.optimizer.set_jit(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)
'''

from tensorflow.keras.optimizers import Adam
from DenseVnet_config import*
import os
import datetime
from DenseVnet3D import DenseVnet3D
from DenseVnet_Loss import*
import numpy as np
import random

#---| Decode Segmentation Data
@tf.function
def decode_SEGct(Serialized_example):

    features={
       'image':tf.io.FixedLenFeature([],tf.string),
       'mask':tf.io.FixedLenFeature([],tf.string),
       'Height':tf.io.FixedLenFeature([],tf.int64),
       'Weight':tf.io.FixedLenFeature([],tf.int64),
       'Depth':tf.io.FixedLenFeature([],tf.int64),
        'Sub_id':tf.io.FixedLenFeature([],tf.string)

     }
    examples=tf.io.parse_single_example(Serialized_example,features)
    ##Decode_image_float
    image_1 = tf.io.decode_raw(examples['image'], float)
    #Decode_mask_as_int32
    mask_1 = tf.io.decode_raw(examples['mask'], tf.int32)
    ##Subject id is already in bytes format
    #sub_id=examples['Sub_id']
    img_shape=[examples['Height'],examples['Weight'],examples['Depth']]
    #img_shape2=[img_shape[0],img_shape[1],img_shape[2]]
    print(img_shape)
    #Resgapping_the_data
    img=tf.reshape(image_1,img_shape)
    mask=tf.reshape(mask_1,img_shape)
    #Because CNN expect(batch,H,W,D,CHANNEL)
    img=tf.expand_dims(img, axis=-1)
    #mask=tf.expand_dims(mask, axis=-1)
    mask = tf.one_hot(tf.cast(mask, tf.int32), 31)
    mask = tf.cast(mask, tf.float32)
    ###casting_values
    img=tf.cast(img, tf.float32)
    #mask=tf.cast(mask,tf.int32)
    return img,mask


#---|Getting the tfrecords list
def getting_list(path):
    a=[file for file in os.listdir(path) if file.endswith('.tfrecords')]
    all_tfrecoeds=random.sample(a, len(a))
    #all_tfrecoeds.sort(key=lambda f: int(filter(str.isdigit, f)))
    list_of_tfrecords=[]
    for i in range(len(all_tfrecoeds)):
        tf_path=path+all_tfrecoeds[i]
        list_of_tfrecords.append(tf_path)
    return list_of_tfrecords

#--Traing Decoder
def load_training_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_SEGct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(TRAING_EPOCH).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset

#--Validation Decoder
def load_validation_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(tf.data.TFRecordDataset,cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_SEGct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(TRAING_EPOCH).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset


def Training():
    #TensorBoard
    logdir = os.path.join(LOG_FILE_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    ##csv_logger
    csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV)
    ##Model-checkpoings
    path=TRAINING_SAVE_MODEL_PATH
    model_path=os.path.join(path, SAVE_MODEL_NAME)
    Model_callback= tf.keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=False,save_weights_only=True,monitor=ModelCheckpoint_MOTITOR,verbose=1)

    tf_train=getting_list(TRAINING_TF_RECORDS)
    tf_val=getting_list(VALIDATION_TF_RECORDS)

    traing_data=load_training_tfrecords(tf_train,BATCH_SIZE)
    Val_batched_dataset=load_validation_tfrecords(tf_val,BATCH_SIZE)

    #--|Single GPU Setup
    if (NUM_OF_GPU==1):
        if RESUME_TRAINING==1:
            inputs = tf.keras.Input(shape=SEG_INPUT_PATCH_SIZE, name='CT')
            Model_3D=DenseVnet3D(inputs,nb_classes=SEG_NUMBER_OF_CLASSES,encoder_nb_layers=NUM_DENSEBLOCK_EACH_RESOLUTION,growth_rate=NUM_OF_FILTER_EACH_RESOLUTION,dilation_list=DILATION_RATE,dropout_rate=DROPOUT_RATE)
            Model_3D.load_weights(RESUME_TRAIING_MODEL)
            initial_epoch_of_training=TRAINING_INITIAL_EPOCH
            Model_3D.compile(optimizer=OPTIMIZER, loss=[SEG_LOSS], metrics=[SEG_METRICS])
            Model_3D.summary()
        else:
            initial_epoch_of_training=0
            inputs = tf.keras.Input(shape=SEG_INPUT_PATCH_SIZE, name='CT')
            Model_3D=DenseVnet3D(inputs,nb_classes=SEG_NUMBER_OF_CLASSES,encoder_nb_layers=NUM_DENSEBLOCK_EACH_RESOLUTION,growth_rate=NUM_OF_FILTER_EACH_RESOLUTION,dilation_list=DILATION_RATE,dropout_rate=DROPOUT_RATE)
            Model_3D.compile(optimizer=OPTIMIZER, loss=[SEG_LOSS], metrics=[SEG_METRICS])
            Model_3D.summary()
        #--| Training
        Model_3D.fit(traing_data,
                   steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                   epochs=TRAING_EPOCH,
                   initial_epoch=initial_epoch_of_training,
                   validation_data=Val_batched_dataset,
                   validation_steps=VALIDATION_STEP,
                   callbacks=[tensorboard_callback,csv_logger,Model_callback])

    ###Multigpu----
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(DISTRIIBUTED_STRATEGY_GPUS)
        with mirrored_strategy.scope():
                if RESUME_TRAINING==1:
                    inputs = tf.keras.Input(shape=SEG_INPUT_PATCH_SIZE, name='CT')
                    Model_3D=DenseVnet3D(inputs,nb_classes=SEG_NUMBER_OF_CLASSES,encoder_nb_layers=NUM_DENSEBLOCK_EACH_RESOLUTION,growth_rate=NUM_OF_FILTER_EACH_RESOLUTION,dilation_list=DILATION_RATE,dropout_rate=DROPOUT_RATE)
                    Model_3D.load_weights(RESUME_TRAIING_MODEL)
                    initial_epoch_of_training=TRAINING_INITIAL_EPOCH
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[SEG_LOSS], metrics=[SEG_METRICS])
                    Model_3D.summary()
                else:
                    initial_epoch_of_training=0
                    inputs = tf.keras.Input(shape=SEG_INPUT_PATCH_SIZE, name='CT')
                    Model_3D=DenseVnet3D(inputs,nb_classes=SEG_NUMBER_OF_CLASSES,encoder_nb_layers=NUM_DENSEBLOCK_EACH_RESOLUTION,growth_rate=NUM_OF_FILTER_EACH_RESOLUTION,dilation_list=DILATION_RATE,dropout_rate=DROPOUT_RATE)
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[SEG_LOSS], metrics=[SEG_METRICS])
                    Model_3D.summary()
                #--| Training
                Model_3D.fit(traing_data,steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                             epochs=TRAING_EPOCH,
                             initial_epoch=initial_epoch_of_training,
                             validation_data=Val_batched_dataset,
                             validation_steps=VALIDATION_STEP,
                             callbacks=[tensorboard_callback,csv_logger,Model_callback])

if __name__ == '__main__':
   Training()
