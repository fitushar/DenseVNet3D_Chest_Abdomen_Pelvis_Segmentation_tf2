from __future__ import absolute_import, print_function, division
import numpy as np
import tensorflow as tf

'''
@Author: Fakrul Islam Tushar,
RA Duke University Medical Center
4/23/2020, NC,USA.
ft42@duke.edu,f.i.tushar.eee@gmail.com
'''

#|Dice Coefficient for binary Segmentation task
def dice_coe(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)

#---|Dice Loss for binary Segmentation task
def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))


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
