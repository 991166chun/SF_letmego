# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:08:53 2019

@author: Chun
"""
import sys
import numpy as np
import xml.etree.ElementTree as ET
import os
import csv
from pascal_voc_io import PascalVocReader as PR
from pascal_voc_io import PascalVocWriter as PW
import shutil


class label_operator():

    def __init__(self, path, im_path, anno_path):

        self.path = path
        self.anno_path = anno_path
        self.im_path = im_path
        lis = []
        if path:
            imgs = os.listdir(im_path)
            imglis = [img.split('.')[0] for img in imgs]
            xmls = os.listdir(anno_path)
            xmllis = [xml.split('.')[0] for xml in xmls]
            for xml in xmllis:
                if xml in imglis:
                    lis.append(xml)
            self.lis = lis

    def calculate_label(self, lis=None):

        if lis == None:
            lis = self.lis

        label_count = {}

        for file in lis:
            xml_name = self.anno_path + '/' + file + '.xml'
            tree = ET.parse(xml_name)
            root = tree.getroot()

            for i in range(len(root.findall('object'))):

                label_name = root[6+i][0].text

                try:
                    label_count[label_name] += 1
                except:
                    label_count[label_name] = 1

        self.label_count = label_count

        self.all_label = []
        for key in label_count.keys():
            self.all_label.append(key)

        return label_count

    def calculate_images(self, lis=None):

        if lis == None:
            lis = self.lis

        img_count = {}

        for file in lis:

            xml_name = self.anno_path + '/' + file + '.xml'

            # if os.path.exists(xml_name):
            f = open(xml_name, "r", encoding="utf-8")
            _str1 = f.read()
            # else:
            #     lis.remove(file)
            #     continue

            for k in self.label_count.keys():
                if k in _str1:
                    try:
                        img_count[k] += 1
                    except:
                        img_count[k] = 1
            f.close()

        return img_count, lis

    def read_txt(self, trainfile, testfile):
        '''
            read txt file to list
        '''
        f1 = open(trainfile, 'r')
        trainlis = f1.read().splitlines()
        f1.close()

        f2 = open(testfile, 'r')
        testlis = f2.read().splitlines()
        f2.close()

        self.trainlis = trainlis
        self.testlis = testlis

        self.trainfile = trainfile
        self.testfile = testfile

        return trainlis, testlis

    def slice_data(self, ratio, trainfile, testfile):
        '''
            random slice all files to 2 list by the ratio 
            e.g. ratio=0.2  -> train : test = 4:1 
        '''
        select, other = self.random_select(ratio)

        self.trainlis = other
        self.testlis = select

        self.trainfile = trainfile
        self.testfile = testfile

        self.make_txt(other, select)

    def make_txt(self, trains, tests, trainfile=None, testfile=None):
        '''
            write trainval/test list to the txt file
        '''

        if trainfile == None:
            trainfile = self.trainfile
            testfile = self.testfile

        f1 = open(testfile, 'w')
        for test in tests:

            f1.write(test)
            f1.write('\n')

        f1.close()

        f2 = open(trainfile, 'w')

        for train in trains:
            f2.write(train)
            f2.write('\n')

        f2.close()

        self.trainfile = trainfile
        self.testfile = testfile

    def print_count(self, countlist, classes=None, ratio=False):
        '''
            print the label count
        '''
        length = len(countlist)

        if classes == None:
            classes = self.all_label

        for key in classes:
            print(key + ':' + ' '*(16-len(key)), end='')
            for i in range(length):
                try:
                    print(str(countlist[i][key]) + '\t', end='')
                except:
                    print('0 \t', end='')
            if ratio:
                print('%.2f' % (countlist[-1][key]/countlist[-2][key]))
            else:
                print(' ')

    def write_count_csv(self, csvfile, countlist, classes=None):
        '''
            write the label count to csv 
        '''
        length = len(countlist)

        if classes == None:
            classes = self.all_label

        with open(csvfile, 'w', newline='') as f:

            writer = csv.writer(f)

            for key in classes:
                row = []
                row.append(key)
                for i in range(length):
                    try:
                        row.append(countlist[i][key])
                    except:
                        row.append(0)
                writer.writerow(row)

    def rename_label(self, old_label, new_label):
        '''
        lab.rename_label('others', 'other')

        '''
        path = self.anno_path
        files = os.listdir(path)

        for file in files:
            f = open(path+"/"+file, "r", encoding="utf-8")
            str1 = f.read()
            str1 = str1.replace(old_label, new_label)
            f.close()
            f1 = open(path+"/"+file, "w", encoding="utf-8")
            f1.write(str1)
            f1.close()

    def switch_label(self, name, trainlis=None, testlis=None):
        '''
         changing file from training set to testing set or reverse
        '''

        if trainlis == None:
            trainlis = self.trainlis
            testlis = self.testlis

        if name in trainlis:

            trainlis.remove(name)
            testlis.append(name)
            print('change file %s from training to testing set' % name)

        elif name in testlis:

            testlis.remove(name)
            trainlis.append(name)
            print('change file %s from testing to training set' % name)

        else:
            print('name is not in list')

        self.make_txt(trainlis, testlis)

    def find_label(self, label, lis=None):
        """
        return a list of images that contain specific label

        """
        img_list = []

        if lis == None:
            lis = self.lis

        for file in lis:

            xml_name = self.anno_path + '/' + file + '.xml'
            f = open(xml_name, "r", encoding="utf-8")
            _str1 = f.read()

            if label in _str1:
                img_list.append(file)
            f.close()

        return img_list

    def random_select(self, ratio):

        files = self.lis
        test_num = int(len(files) * ratio)
        idx = np.random.randint(0, len(files), test_num)
        keep = np.zeros(len(files)).astype(bool)
        keep[idx] = True

        select = np.array(files)[keep].tolist()
        other = np.array(files)[~keep].tolist()

        return select, other

    def del_label(self, label_del):

        flist = os.listdir(self.anno_path)
        for filename in flist:
            filepath = self.anno_path + '/' + filename
            pr = PR(filepath)
            shapes = pr.getShapes()
            sl = pr.getSources()

            localpath = os.path.join(self.im_path, sl['filename'])

            pw = PW(sl['folder'],
                    sl['filename'],
                    sl['size'],
                    databaseSrc='Unknown',
                    localImgPath=localpath)
            pw.verified = True

            for shape in shapes:
                if shape[0] != label_del:
                    label = shape[0]
                    difficult = 0
                    bndbox = shape[1]
                    pw.addBndBox(bndbox[0][0], bndbox[0][1],
                                 bndbox[2][0], bndbox[2][1], label, difficult)

            pw.save(filepath)

    def check_label(self):

        flist = os.listdir(self.anno_path)

        for filename in flist:
            filepath = self.anno_path + '/' + filename
            pr = PR(filepath)
            shapes = pr.getShapes()
            sl = pr.getSources()

            if len(shapes) == 0:
                print(filepath)

    def label_filter(self, labels, new_dir):
        flist = []
        for lb in labels:
            stemp = self.find_label(lb)
            for fname in stemp:
                if (fname in flist) == False:
                    flist.append(fname)

        for filename in flist:
            filepath = self.anno_path + '/' + filename + '.xml'
            pr = PR(filepath)
            shapes = pr.getShapes()
            sl = pr.getSources()
            localpath = os.path.join(self.im_path, sl['filename'])

            pw = PW(sl['folder'],
                    sl['filename'],
                    sl['size'],
                    databaseSrc='Unknown',
                    localImgPath=localpath)
            pw.verified = True

            for shape in shapes:
                if shape[0] in labels:
                    label = shape[0]
                    difficult = 0
                    bndbox = shape[1]
                    pw.addBndBox(bndbox[0][0], bndbox[0][1],
                                 bndbox[2][0], bndbox[2][1], label, difficult)

            pw.save(new_dir + '/' + filename + '.xml')

    def copy_image_dir(self, cop_dir, im_list, ori_dir=None):
        '''
            usage: 
                lis2 = lab.find_label('mosquito_late')
                lab.copy_image_dir('./tempan', lis2)
        '''

        if ori_dir == None:
            ori_dir = self.im_path

        for img in im_list:
            ori_file = ori_dir + '/' + img + '.jpg'
            cop_file = cop_dir + '/' + img + '.jpg'

            shutil.copyfile(ori_file, cop_file)

    def refresh_all_xml(self):

        pass

        # TODO:
        # 過濾label
        return None

    def resort_txt(self, file):

        f1 = open(file, 'r')
        lis = f1.read().splitlines()
        f1.close()

        lis = np.array(list(map(int, lis)))

        lis = np.sort(lis).tolist()

        with open(file, 'w') as f1:

            for num in lis:
                nid = '%04d' % num

                xml = os.path.join(self.anno_path, nid + '.xml')
                img = os.path.join(self.im_path, nid + '.jpg')

                if not os.path.exists(xml):
                    print('xml file %s is not exist' % nid)
                    continue
                if not os.path.exists(img):
                    print('xml file %s is not exist' % nid)
                    continue

                f1.write(nid)
                f1.write('\n')

            f1.close()

        return lis

    '''
        
        f1.write(test)
        f1.write('\n')
    
    f1.close()
    '''


if __name__ == '__main__':

    im_path = './images'
    anno_path = './data0504/Annotations_nn'
    # trainfile = 'data1109/trainval.txt'
    # testfile = 'data1109/test.txt'
    classes = ['brownblight', 'blister', 'algal',  'fungi_early',
               'miner',   'thrips',
               'mosquito_early', 'mosquito_late',
               'moth', 'tortrix',   'flushworm',
               'roller',   'other']

    lab = label_operator(True, im_path, anno_path)

    all_list = lab.lis

    label_count = lab.calculate_label()
    img_count, _ = lab.calculate_images()
    lab.print_count([label_count, img_count])

# -----------------------------------------------------------
    # Count train - test
    testfile = './data0504/ImageSets/Main/test.txt'
    trainfile = './data0504/ImageSets/Main/trainval.txt'
    csvfile = './data0504/count.csv'

    # lab.slice_data(0.2, trainfile, testfile)
    trainlis, testlis = lab.read_txt(trainfile, testfile)

    train_count, trainlis = lab.calculate_images(trainlis)
    test_count, testlis = lab.calculate_images(testlis)
    # lab.make_txt(trainlis,testlis)
    lab.print_count([label_count, img_count, train_count,
                     test_count], classes, ratio=True)
    lab.write_count_csv(
        csvfile, [label_count, img_count, train_count, test_count], classes)

# -----------------------------------------------------------
    # filt labels
#    new_dir = 'data1109/roller_only'
#    filt_label = ['roller','roll_leaf','moth', 'tortrix', 'flushworm']
#    lab.label_filter(filt_label, new_dir)

    print('There are %d classes' % len(lab.all_label))
