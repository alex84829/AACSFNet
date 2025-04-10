import os
import numpy as np
import PIL.Image as Image
import random
import math

#D:\UNBC-McMaster\Frame_Labels\PSPI\042-ll042\ll042t1aaaff    ll042t1aaaff001_facs.txt
#D:\UNBC-McMaster\cropped_Images\042-ll042\ll042t1aaaff

def aam_normlization(aam):
    x = aam[30][0]
    y = aam[30][1]
    distance = []

    for i in range(66):
        dis = (aam[i][0] - x) ** 2 + (aam[i][1] - y) ** 2
        dis = math.sqrt(dis)

        distance.append(dis)
    return np.array(distance, dtype='float16')


def read_data(image_path,label_path, aam_path):
    print('read data!!!')
    Train = image_path.split('/')[-1].split('_')[0]

    batch_data = []
    batch_aam = []
    batch_label = []
    count = 0
    different = 0
    
    key_list = ['warp_transpose','warp_gaussian_003','warp_bright','warp_dark_015','warp_contrast16']
    pass_list = ['warp_gaussian_003']

    for line_image in open(image_path, mode='r'):
        if line_image.split('/')[7] in pass_list:
            continue

        fileSequence = line_image.split('/',8)[-1].strip()
        labelSequence = label_path + '/' + fileSequence
        aamSequence = aam_path + '/' + fileSequence
        img = sorted(os.listdir(line_image.strip()))
        aam = sorted(os.listdir(aamSequence))
        label = sorted(os.listdir(labelSequence))

        zero_count = 0
        for i in range(16, len(img)):
            img_datas = []
            for j in range(i - 16, i):
                #img_data = Image.open(os.path.join(line_image.strip(), img[j].strip())).convert('L')
                img_data = Image.open(os.path.join(line_image.strip(), img[j].strip())).convert('RGB')
                img_data = np.array(img_data.resize((112, 112), Image.ANTIALIAS))
                #print(img_data.dtype)
                img_data = np.reshape(img_data, (112, 112, 3))
                img_datas.append(img_data)

            if img[i - 1].split('.')[0] == label[i - 1].split('_')[0]:
                #f1 = open(os.path.join(labelSequence, label[i - 1].strip()), 'r')
                fixed_label = np.loadtxt(os.path.join(labelSequence, label[i - 1].strip()))
                fixed_label = np.float16(fixed_label)
            else:
                different = different + 1
                break

            if img[i - 1].split('.')[0] == aam[i - 1].split('_')[0]:
                array = np.loadtxt(os.path.join(aamSequence, aam[i - 1].strip()))
                array = np.float16(array)
                arr = np.array(array)
                res = aam_normlization(arr)
            else:
                different = different + 1
                break

            if fixed_label == 0.0:
                zero_count = zero_count + 1

            if fixed_label == 0.0 and zero_count > 20 and Train == 'train':
                continue
            if (line_image.split('/')[7] in key_list) and (fixed_label == 0.0):
                continue

            batch_data.append(np.asarray(img_datas))
            batch_aam.append(np.asarray(res))
            batch_label.append(fixed_label)

        count = count + 1
        print('Completed {} sequences!!!'.format(count))
    print(different)
    return batch_data,batch_aam,batch_label

for id in range(3,25):
    print(id)
    #labelTxt = '/media/dvc614/8eabcc17-2e55-4ec1-8a68-428a16020df8/dvc614/Huang/UNBC/UNBC_code/labeltxt.txt'
    labelPath = '/home/server/serverData/Huang/UNBC/PSPI'
    AAM_path = '/home/server/serverData/Huang/UNBC/AAM_landmarks'

    train_path = './txtFile/warp_txt/train_'+str(id)+'.txt'
    val_path = './txtFile/warp_txt/val_'+str(id)+'.txt'
    test_data,test_aam,test_label = read_data(val_path,labelPath,AAM_path)
    train_data,train_aam,train_label = read_data(train_path,labelPath, AAM_path)

    cc = list(zip(train_data,train_aam,train_label))
    random.shuffle(cc)
    train_data[:], train_aam[:], train_label[:] = zip(*cc)
    cc = list(zip(test_data,test_aam,test_label))
    random.shuffle(cc)
    test_data[:], test_aam[:], test_label[:] = zip(*cc)

    data_dict = {}
    data_dict['train_data'] = train_data
    data_dict['train_aam'] = train_aam
    data_dict['train_label'] = train_label
    data_dict['test_data'] = test_data
    data_dict['test_aam'] = test_aam
    data_dict['test_label'] = test_label
    np.save('./data/aam_112_RGB_'+str(id)+'.npy', data_dict)