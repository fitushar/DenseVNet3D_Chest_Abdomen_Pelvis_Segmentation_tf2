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

