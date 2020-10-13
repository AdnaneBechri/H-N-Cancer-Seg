
"""
Source
All the code here is based on theseGithub repositories :

https://github.com/QuantScientist/V-Net.pytorch

https://github.com/josedolz/LiviaNET

https://github.com/josedolz/HyperDenseNet
"""


from os.path import isfile, join
import os
import numpy as np
from sampling import reconstruct_volume
from sampling import my_reconstruct_volume
from sampling import load_data_trainG
from sampling import load_data_test
from sampling import extract_patches
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from VNet_FEE_2mod import *
from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib
import torch.nn.functional as F


class DicexCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DicexCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        CE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = CE - torch.log(dice_loss)
        #Dice_BCE = dice_loss
        return Dice_BCE

def mergeit(x,y):
    x_train = np.zeros(x.shape)
    y_train = np.zeros(y.shape)
    valid_idxs1 = np.where(np.sum(y, axis=(1, 2, 3)) != 0)[0]
    np.random.shuffle(valid_idxs1)
    valid_idxs0 = np.where(np.sum(y, axis=(1, 2, 3)) == 0)[0]
    np.random.shuffle(valid_idxs0)
    i = 0
    for a,b in zip(valid_idxs1, valid_idxs0):
        x_train[i,:,:,:,:] = x[a,:,:,:,:]
        y_train[i, :, :, :] = y[a, :, :, :]
        x_train[i+1, :, :, :, :] = x[b, :, :, :, :]
        y_train[i+1, :, :, :] = y[b, :, :, :]
        i+=2

    return x_train, y_train


def evaluateSegmentation(gt,pred):
    pred = pred.astype(dtype='int')
    numClasses = np.unique(gt)

    dsc = np.zeros((1, len(numClasses) - 1))

    for i_n in range(1,len(numClasses)):
        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==i_n)]=1
        y_c[np.where(pred==i_n)]=1

        dsc[0, i_n - 1] = dc(gt_c, y_c)
    return dsc
    
def numpy_to_var(x):
    torch_tensor = torch.from_numpy(x).type(torch.FloatTensor)
    
    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    return Variable(torch_tensor)


def inferencee(network, x_train, y_train, imageNames, epoch, folder_save, number_modalities):
    a = 64
    b = 64
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_2 = root_dir + 'Training/T1_IR'
    moda_3 = root_dir + 'Training/T2_FLAIR'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Sigmoid()
    numClasses = 1
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()

    patchSize = a
    patchSize_gt = b

    pred_numpy = np.zeros((0, patchSize_gt, patchSize_gt, patchSize_gt))
    pred_numpy = np.vstack((pred_numpy, np.zeros((x_train.shape[0], patchSize_gt, patchSize_gt, patchSize_gt))))
    # pred = network(numpy_to_var(x[0,:,:,:,:]).view(1,number_modalities,patchSize,patchSize,patchSize))
    for i_p in range(x_train.shape[0]):
        pred = network(
            numpy_to_var(x_train[i_p, :, :, :, :].reshape(1, number_modalities, patchSize, patchSize, patchSize)))
        pred_y = softMax(pred.reshape(patchSize_gt,patchSize_gt,patchSize_gt))
        pred_numpy[i_p, :, :, :] = pred_y.cpu().data.numpy()

        printProgressBar(i_p + 1, x_train.shape[0],
                         prefix="[Training_eval] ",
                         length=15)

    # To reconstruct the predicted volume
    extraction_step_value = b
    pred_classes = np.round(pred_numpy)

    pred_classes = pred_classes.reshape((x_train.shape[0], patchSize_gt, patchSize_gt, patchSize_gt))
    # bin_seg = reconstruct_volume(pred_classes, (img_shape[1], img_shape[2], img_shape[3]))


    # bin_seg = bin_seg[:,:,extraction_step_value:img_shape[3]-extraction_step_value]

    dsc = dc(y_train, pred_classes)
    acc = accuracy_score(y_train.flatten(), pred_classes.flatten())


    return dsc, acc


def inference(network, moda_n, moda_g, imageNames, epoch, folder_save, number_modalities):
    a = 64
    b = 64
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_2 = root_dir + 'Training/T1_IR'
    moda_3 = root_dir + 'Training/T2_FLAIR'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Sigmoid()
    numClasses = 1
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()

    dscAll = []
    accall = []
    for i_s in range(len(imageNames)):
        if number_modalities == 2:
            patch_1, patch_2, patch_g, img_shape = load_data_test(moda_n, moda_g, imageNames[i_s], number_modalities)  # hardcoded to read the first file. Loop this to get all files
        if number_modalities == 3:
            patch_1, patch_2, patch_3, patch_g, img_shape = load_data_test([moda_n], moda_g, imageNames[i_s], number_modalities) # hardcoded to read the first file. Loop this to get all files
       # Normalization

        patchSize = a
        patchSize_gt = b

        x = np.zeros((0, number_modalities, patchSize, patchSize, patchSize))
        x = np.vstack((x, np.zeros((patch_1.shape[0], number_modalities, patchSize, patchSize, patchSize))))
        x[:, 0, :, :, :] = patch_1
        x[:, 1, :, :, :] = patch_2
        if (number_modalities==3):
            x[:, 2, :, :, :] = patch_3
        
        pred_numpy = np.zeros((0,patchSize_gt,patchSize_gt,patchSize_gt))
        pred_numpy = np.vstack((pred_numpy, np.zeros((patch_1.shape[0], patchSize_gt, patchSize_gt, patchSize_gt))))
        totalOp = len(imageNames)*patch_1.shape[0]
        #pred = network(numpy_to_var(x[0,:,:,:,:]).view(1,number_modalities,patchSize,patchSize,patchSize))
        for i_p in range(patch_1.shape[0]):
            pred = network(numpy_to_var(x[i_p, :, :, :, :].reshape(1, number_modalities, patchSize, patchSize, patchSize)))
            pred_y = softMax(pred.reshape(patchSize_gt,patchSize_gt,patchSize_gt))
            pred_numpy[i_p,:,:,:] = pred_y.cpu().data.numpy()

            printProgressBar(i_s * ((totalOp + 0.0) / len(imageNames)) + i_p + 1, totalOp,
                             prefix="[Validation] ",
                             length=15)

        # To reconstruct the predicted volume
        extraction_step_value = b
        pred_classes = np.round(pred_numpy)
        
        pred_classes = pred_classes.reshape((patch_1.shape[0], patchSize_gt, patchSize_gt, patchSize_gt))
        #bin_seg = reconstruct_volume(pred_classes, (img_shape[1], img_shape[2], img_shape[3]))

        bin_seg = my_reconstruct_volume(pred_classes,
                                        (img_shape[1], img_shape[2], img_shape[3]),
                                        patch_shape=(a,a,a),
                                        extraction_step=(b,b,b))

        #bin_seg = bin_seg[:,:,extraction_step_value:img_shape[3]-extraction_step_value]
        #label_selector = [slice(None)] + [slice(9, 117) for i in range(3)]
        gt = nib.load(moda_g + '/' + imageNames[i_s]).get_fdata()
        gt_patches = extract_patches(gt, (a,a,a), (b,b,b))
        #gt_patches = gt_patches[label_selector]
        img_pred = nib.Nifti1Image(bin_seg, np.eye(4))
        img_gt = nib.Nifti1Image(gt, np.eye(4))

        img_name = imageNames[i_s].split('.nii')
        name = 'Pred_' + img_name[0] + '_Epoch_' + str(epoch) + '.nii.gz'

        namegt = 'GT_' + img_name[0] + '_Epoch_' + str(epoch) + '.nii.gz'

        if not os.path.exists(folder_save + 'Segmentations/'):
            os.makedirs(folder_save + 'Segmentations/')

        if not os.path.exists(folder_save + 'GT/'):
            os.makedirs(folder_save + 'GT/')

        nib.save(img_pred, folder_save + 'Segmentations/' + name)
        nib.save(img_gt, folder_save + 'GT/' + namegt)

        dsc = dc(gt_patches,pred_classes)
        acc = accuracy_score(gt_patches.flatten(), pred_classes.flatten())
        dscAll.append(dsc)
        accall.append(acc)
    return dscAll, accall
        
def runTraining(opts):
    print('' * 41)
    print('~' * 50)
    print('~~~~~~~~~~~~~~~~~  PARAMETERS ~~~~~~~~~~~~~~~~')
    print('~' * 50)
    print('  - Number of image modalities: {}'.format(opts.numModal))
    print('  - Number of classes: {}'.format(opts.numClasses))
    print('  - Directory to load images: {}'.format(opts.root_dir))
    for i in range(len(opts.modality_dirs)):
        print('  - Modality {}: {}'.format(i+1,opts.modality_dirs[i]))
    print('  - Directory to save results: {}'.format(opts.save_dir))
    print('  - To model will be saved as : {}'.format(opts.modelName))
    print('-' * 41)
    print('  - Number of epochs: {}'.format(opts.numClasses))
    print('  - Batch size: {}'.format(opts.batchSize))
    print('  - Number of samples per epoch: {}'.format(opts.numSamplesEpoch))
    print('  - Learning rate: {}'.format(opts.l_rate))
    print('' * 41)

    print('-' * 41)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 41)
    print('' * 40)
    a = 64
    b = 64
    samplesPerEpoch = opts.numSamplesEpoch
    batch_size = opts.batchSize

    lr = opts.l_rate
    epoch = opts.numEpochs
    
    root_dir = opts.root_dir
    model_name = opts.modelName

    if not (len(opts.modality_dirs)== opts.numModal): raise AssertionError

    moda_1 = root_dir + 'Training/' + opts.modality_dirs[0]
    moda_2 = root_dir + 'Training/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3 = root_dir + 'Training/' + opts.modality_dirs[2]

    moda_g = root_dir + 'Training/GT'

    print(' --- Getting image names.....')
    print(' - Training Set: -')
    if os.path.exists(moda_1):
        imageNames_tr = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f))]
        np.random.seed(1)
        np.random.shuffle(imageNames_tr)
        imageNames_val = imageNames_tr[0:40]
        imageNames_train = list(set(imageNames_tr) - set(imageNames_val))
        print(' ------- Images found ------')
        for i in range(len(imageNames_train)):
            print(' - {}'.format(imageNames_train[i]))
    else:
        raise Exception(' - {} does not exist'.format(moda_1))
    moda_1_val = root_dir + 'Training/' + opts.modality_dirs[0]
    moda_2_val = root_dir + 'Training/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3_val = root_dir + 'Training/' + opts.modality_dirs[2]
    moda_g_val = root_dir + 'Training/GT'

    print(' --------------------')
    print(' - Validation Set: -')
    if os.path.exists(moda_1):
        # imageNames_val = [f for f in os.listdir(moda_1_val) if isfile(join(moda_1_val, f))]
        # imageNames_val.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_val)):
            print(' - {}'.format(imageNames_val[i]))
    else:
        raise Exception(' - {} does not exist'.format(moda_1_val))
          
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = opts.numClasses
    
    # Define HyperDenseNet
    # To-Do. Get as input the config settings to create different networks
    if (opts.numModal == 2):
        hdNet = VNet_FEE()

    #

    '''try:
        hdNet = torch.load(os.path.join(model_name, "Best_" + model_name + ".pkl"))
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    #softMax = nn.Softmax()
    softMax = nn.Sigmoid()
    CE_loss = DicexCELoss()
    
    if torch.cuda.is_available():
        hdNet.cuda()
        softMax.cuda()
        CE_loss.cuda()

    # To-DO: Check that optimizer is the same (and same values) as the Theano implementation
    optimizer = torch.optim.Adam(hdNet.parameters(), lr=lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(hdNet.parameters(), lr=lr, momentum = 0.9)
    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    dscAll = []
    accall = []
    train_eval = []
    dsc_eval = []
    acc_eval = []

    d1 = 0

    if (opts.numModal == 2):
        imgPaths = [moda_1, moda_2]

    if (opts.numModal == 3):
        imgPaths = [moda_1, moda_2, moda_3]
    val_epochs = [0,21,41,61,81,101,121,141]
    x_train, y_train, img_shape = load_data_trainG(imgPaths, moda_g, imageNames_train, samplesPerEpoch, opts.numModal)
    #x_train = np.moveaxis(x_train, 1, -1)
    print(x_train.shape)
    numBatches = int(x_train.shape[0] / batch_size)
    idx = np.arange(x_train.shape[0])
    for e_i in range(epoch):
        hdNet.train()
        lossEpoch = []
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]
        for b_i in range(numBatches):
            optimizer.zero_grad()
            hdNet.zero_grad()
            MRIs         = numpy_to_var(x_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:,:])
            Segmentation = numpy_to_var(y_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:])

            segmentation_prediction = hdNet(MRIs)
            #print("segmentation_prediction : ", segmentation_prediction.shape)
            #predClass_y = softMax(segmentation_prediction)
            segmentation_prediction = softMax(segmentation_prediction)
            # To adapt CE to 3D
            # LOGITS:
            if e_i == 0 and b_i == 0:
                print("MRIS : ", MRIs.shape)
                print("Segmentation : ", Segmentation.shape)
                print("segmentation_prediction : ", segmentation_prediction.shape)
            #segmentation_prediction = segmentation_prediction.reshape((MRIs.shape[0], b, b, b))
            #segmentation_prediction = segmentation_prediction.permute(0,2,3,4,1).contiguous()
            segmentation_prediction = segmentation_prediction.reshape(-1)
            CE_loss_batch = CE_loss(segmentation_prediction, Segmentation.reshape(-1).type(torch.cuda.FloatTensor))
            loss = CE_loss_batch
            loss.backward()
            
            optimizer.step()
            lossEpoch.append(CE_loss_batch.cpu().data.numpy())

            printProgressBar(b_i + 1, numBatches,
                             prefix="[Training] Epoch: {} ".format(e_i),
                             length=15)
              
            del MRIs
            del Segmentation
            del segmentation_prediction
            # del predClass_y

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        np.save(os.path.join(model_name, model_name + '_loss.npy'), dscAll)

        print(' Epoch: {}, loss: {}'.format(e_i,np.mean(lossEpoch)))

        if (e_i%5)==0 :

            if (opts.numModal == 2):
                moda_n = [moda_1_val, moda_2_val]
            if (opts.numModal == 3):
                moda_n = [moda_1_val, moda_2_val, moda_3_val]

            dsct, acct = inferencee(hdNet, x_train, y_train, imageNames_train, e_i, opts.save_dir, opts.numModal)
            dsc_eval.append(dsct)
            acc_eval.append(acct)
            print(' Metrics: The mean of train Accuracy is : {} '.format(acct))
            print(' Metrics: The mean of train DSC is : {} '.format(dsct))
            np.save(os.path.join(model_name, model_name + 'train_DSCs.npy'), dsc_eval)
            np.save(os.path.join(model_name, model_name + 'train_ACC.npy'), acc_eval)

            dsc, acc = inference(hdNet,moda_n, moda_g_val, imageNames_val, e_i, opts.save_dir,opts.numModal)

            dscAll.append(dsc)
            accall.append(acc)

            print(' Metrics: The mean of Accuracy is : {} '.format(np.mean(acc)))
            print(' Metrics: The mean of DSC is : {} '.format(np.mean(dsc)))
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            
            np.save(os.path.join(model_name, model_name + '_DSCs.npy'), dscAll)
            np.save(os.path.join(model_name, model_name + '_ACC.npy'), accall)

            if np.mean(dsc)>0.64:
                if not os.path.exists(model_name):
                    os.makedirs(model_name)
                torch.save(hdNet, os.path.join(model_name, "Best2_" + model_name + str(e_i) + ".pkl"))
        """
        if ((10 + e_i) % 10) == 0:
            lr = lr / 2
            print(' Learning rate decreased to : {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./data/', help='directory containing the train and val folders')
    parser.add_argument('--modality_dirs', nargs='+', default=['modality_1','modality_2'], help='subdirectories containing the multiple modalities')
    parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
    parser.add_argument('--modelName', type=str, default='VNet_FEE_2mod', help='name of the model')
    parser.add_argument('--numModal', type=int, default=2, help='Number of image modalities')
    parser.add_argument('--numClasses', type=int, default=1, help='Number of classes (Including background)')
    parser.add_argument('--numSamplesEpoch', type=int, default=19000, help='Number of samples per epoch')
    parser.add_argument('--numEpochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=4, help='Batch size')
    parser.add_argument('--l_rate', type=float, default=0.0002, help='Learning rate')

    opts = parser.parse_args()
    print(opts)
    
    runTraining(opts)
