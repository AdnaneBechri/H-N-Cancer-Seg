"""
Source

All the code here is based on these Github repositories :

https://github.com/UdonDa/3D-UNet-PyTorch

https://github.com/josedolz/LiviaNET

https://github.com/josedolz/HyperDenseNet

"""
import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import pdb
import itertools

def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    #poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    pad_shape = (b,b,b)
    poss_shape = [patch_shape[i + 1] * ((expected_shape[i] - pad_shape[i] * 2) // patch_shape[i + 1]) + pad_shape[i] * 2 for i in range(ndims - 1)]

    #idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]
    idxs = [range(pad_shape[i], poss_shape[i] - pad_shape[i], patch_shape[i + 1]) for i in range(ndims - 1)]
    #pdb.set_trace()
    return itertools.product(*idxs)

    
def extract_patches(volume, patch_shape, extraction_step):
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches,) + patch_shape)
    #return patches.reshape((len(patchesList), ) + patch_shape)

# Double check that number of labels is continuous
def get_one_hot(targets, nb_classes):
    #return np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return np.swapaxes(np.eye(nb_classes)[np.array(targets)],0,3) # Jose. To have the same shape as pytorch (batch_size, numclasses,x,y,z)

def build_set(imageData, numModalities) :
    num_classes = 1
    a = 64
    b = 64
    patch_shape = (a,a,a)
    extraction_step=(b,b,b)
    #extraction_step=(9, 9, 3)
    #label_selector = [slice(None)] + [slice(9, 117) for i in range(3)]

    # Extract patches from input volumes and ground truth
    imageData_1 = np.squeeze(imageData[0,:,:,:])
    imageData_2 = np.squeeze(imageData[1,:,:,:])
    if (numModalities==3):
        imageData_3 = np.squeeze(imageData[2,:,:,:])
        imageData_g = np.squeeze(imageData[3,:,:,:])
    if (numModalities == 2):
        imageData_g = np.squeeze(imageData[2, :, :, :])

    num_classes = len(np.unique(imageData_g))
    x = np.zeros((0, numModalities, a,a,a))
    y = np.zeros((0, b,b,b))

    #for idx in range(len(imageData)) :
    y_length = len(y)

    label_patches = extract_patches(imageData_g, patch_shape, extraction_step)
    #label_patches = label_patches[label_selector]

    # Select only those who are important for processing


    x = np.vstack((x, np.zeros((len(label_patches), numModalities, a,a,a))))
    y = np.vstack((y, np.zeros((len(label_patches), b,b,b))))  # Jose
    
    y = label_patches
    del label_patches
    
    # Sampling strategy: reject samples which labels are only zeros
    T1_train = extract_patches(imageData_1, patch_shape, extraction_step)
    x[y_length:, 0, :, :, :] = T1_train
    del T1_train

    # Sampling strategy: reject samples which labels are only zeros
    T2_train = extract_patches(imageData_2, patch_shape, extraction_step)
    x[y_length:, 1, :, :, :] = T2_train
    del T2_train

    if (numModalities==3):
        # Sampling strategy: reject samples which labels are only zeros
        Fl_train = extract_patches(imageData_3, patch_shape, extraction_step)
        x[y_length:, 2, :, :, :] = Fl_train[valid_idxs]
        del Fl_train

    return x, y

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        #pdb.set_trace()
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

def my_reconstruct_volume(patches, expected_shape, patch_shape, extraction_step) :

    reconstructed_img = np.zeros(expected_shape)
    idx = 0
    #pdb.set_trace()
    for x_i in range(0,expected_shape[0]-patch_shape[0],extraction_step[0]):
        for y_i in range(0,expected_shape[1]-patch_shape[1],extraction_step[1]):
            for z_i in range(0,expected_shape[2]-patch_shape[2],extraction_step[2]):
                #pdb.set_trace()
                reconstructed_img[(x_i) :(x_i + extraction_step[0]),
                                  (y_i) :(y_i + extraction_step[1]),
                                  (z_i) :(z_i + extraction_step[2])] = patches[idx]

                #reconstructed_img[(x_i + extraction_step[0]):(x_i + 2 * extraction_step[0]),
                #                  (y_i + extraction_step[1]):(y_i + 2 * extraction_step[1]),
                #                  (z_i ):(z_i + extraction_step[2])] = patches[idx]
                idx = idx + 1

    return reconstructed_img



def my_reconstruct_volumee(patches, expected_shape, patch_shape, extraction_step) :
    reconstructed_img = np.zeros(expected_shape)
    idx = 0
    #pdb.set_trace()

    for x_i in range(0,expected_shape[0]-patch_shape[0],108):
        for y_i in range(0,expected_shape[1]-patch_shape[1],108):
            for z_i in range(0,expected_shape[2]-patch_shape[2],108):
                #pdb.set_trace()
                reconstructed_img[(x_i + extraction_step[0]):(x_i + 12 * extraction_step[0]),
                                  (y_i + extraction_step[1]):(y_i + 12 * extraction_step[1]),
                                  (z_i + extraction_step[2]):(z_i + 12 * extraction_step[2])] = patches[idx]
                idx = idx + 1
    return reconstructed_img





def load_data_trainG(paths, pathg, imageNames, numSamples, numModalities):
    samplesPerImage = int(numSamples / len(imageNames))
    # print(' - Extracting {} samples per image'.format(samplesPerImage))
    X_train = []
    Y_train = []
    for num in range(len(imageNames)):
        imageData_1 = nib.load(paths[0] + '/' + imageNames[num]).get_fdata()
        imageData_1 = (imageData_1 - imageData_1.min()) / (imageData_1.max() - imageData_1.min())
        imageData_2 = nib.load(paths[1] + '/' + imageNames[num]).get_fdata()
        imageData_2 = (imageData_2 - imageData_2.min()) / (imageData_2.max() - imageData_2.min())
        if (numModalities == 3):
            #imageData_3 = nib.load(paths[2] + '/' + imageNames[num]).get_fdata()
            imageData_3 = imageData_2 - (imageData_2 - imageData_2[:,:,-1])/2
            imageData_3 = (imageData_3 - imageData_3.min()) / (imageData_3.max() - imageData_3.min())
        imageData_g = nib.load(pathg + '/' + imageNames[num]).get_fdata()

        num_classes = len(np.unique(imageData_g))

        if (numModalities == 2):
            imageData = np.stack((imageData_1, imageData_2, imageData_g))
        if (numModalities == 3):
            imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))

        img_shape = imageData.shape
        x_train, y_train = build_set(imageData, numModalities)
        # print('here is it : ', x_train)

        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]
        X_train.append(x_train)
        Y_train.append(y_train)

        del x_train
        del y_train

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X = np.concatenate(X_train, axis=0)
    del X_train

    Y = np.concatenate(Y_train, axis=0)
    del Y_train

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], Y[idx], img_shape

def load_data_train(path1, path2, path3, pathg, imageNames, numSamples):

    samplesPerImage = int(numSamples/len(imageNames))

    X_train = []
    Y_train = []
  
    for num in range(len(imageNames)):
        imageData_1 = nib.load(path1 + '/' + imageNames[num]).get_fdata()
        imageData_2 = nib.load(path2 + '/' + imageNames[num]).get_fdata()
        imageData_3 = nib.load(path3 + '/' + imageNames[num]).get_fdata()
        imageData_g = nib.load(pathg + '/' + imageNames[num]).get_fdata()
        
        num_classes = len(np.unique(imageData_g))

        imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
        img_shape = imageData.shape

        x_train, y_train = build_set(imageData)
        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)

        x_train = x_train[idx[:samplesPerImage],]
        y_train = y_train[idx[:samplesPerImage],]

        X_train.append(x_train)
        Y_train.append(y_train)

        del x_train
        del y_train

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X = np.concatenate(X_train, axis=0)
    del X_train

    Y = np.concatenate(Y_train, axis=0)
    del Y_train

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], Y[idx], img_shape

def load_data_test(path_n, pathg, imgName, number_modalities):
    a = 64
    b = 64
    extraction_step_value = b
    imageData_1 = nib.load(path_n[0] + '/' + imgName).get_fdata()
    imageData_1 = (imageData_1 - imageData_1.min())/(imageData_1.max()-imageData_1.min())
    imageData_2 = nib.load(path_n[1] + '/' + imgName).get_fdata()
    imageData_2 = (imageData_2 - imageData_2.min())/(imageData_2.max()-imageData_2.min())
    if number_modalities==3 :
        #imageData_3 = nib.load(path_n[2] + '/' + imgName).get_fdata()
        imageData_3 = imageData_2 - (imageData_2 - imageData_2[:,:,-1])/2
        imageData_3 = (imageData_3 - imageData_3.min())/(imageData_3.max()-imageData_3.min())

    imageData_g = nib.load(pathg + '/' + imgName).get_fdata()

    imageData_1_new = np.zeros((imageData_1.shape[0] ,imageData_1.shape[1], imageData_1.shape[2]))
    imageData_2_new = np.zeros((imageData_1.shape[0],imageData_1.shape[1], imageData_1.shape[2]))
    if number_modalities == 3:
        imageData_3_new = np.zeros((imageData_1.shape[0],imageData_1.shape[1], imageData_1.shape[2]))

    imageData_g_new = np.zeros((imageData_1.shape[0],imageData_1.shape[1], imageData_1.shape[2]))

    imageData_1_new = imageData_1
    imageData_2_new = imageData_2
    if number_modalities == 3:
        imageData_3_new = imageData_3

    imageData_g_new = imageData_g

    num_classes = len(np.unique(imageData_g))

    if number_modalities == 2:
        imageData = np.stack((imageData_1_new, imageData_2_new, imageData_g_new))

    if number_modalities == 3:
        imageData = np.stack((imageData_1_new, imageData_2_new, imageData_3_new, imageData_g_new))
    img_shape = imageData.shape

    patch_1 = extract_patches(imageData_1_new, patch_shape=(a,a,a), extraction_step=(b,b,b))
    patch_2 = extract_patches(imageData_2_new, patch_shape=(a,a,a), extraction_step=(b,b,b))
    if number_modalities == 3:
        patch_3 = extract_patches(imageData_3_new, patch_shape=(a,a,a), extraction_step=(b,b,b))
    patch_g = extract_patches(imageData_g_new, patch_shape=(a,a,a), extraction_step=(b,b,b))

    if number_modalities==2 :
        return patch_1, patch_2, patch_g, img_shape

    if number_modalities==3 :
        return patch_1, patch_2, patch_3, patch_g, img_shape