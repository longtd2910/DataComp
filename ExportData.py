from math import exp, sqrt
from typing import Tuple
from pandas.core.frame import DataFrame
import xlsxwriter
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
from shutil import copy
from pathlib import Path
import pandas as pd
import numpy as np
import unittest


class ExcelWorker:
    DEFAULT_CHAR_WIDTH = 1/7
    PARAM = ['Image', 'Corrected Label', 'Predicted Label', 'Accuracy', 'Loss', "IOU", 'Split']

    def __init__(self, name: str):
        self.workbook = xlsxwriter.Workbook(name)
        self.worksheet = self.workbook.add_worksheet()
        my_format = self.workbook.add_format()
        my_format.set_align('vcenter')
        my_format.set_align('center')
        my_format.set_text_wrap(1)
        self.worksheet.set_column('A:XFD', None, my_format)
        for i in range(len(self.PARAM)):
            self.worksheet.write(0, i, self.PARAM[i])

    def write_list(self, row, col, lst: list) -> str:
        value = ''
        if(len(lst) != 0):
            max_len = len(max([str(a) for a in lst], key=len))
            for i in range(len(lst)):
                value += str(i+1) + ". " + str(lst[i])
                if i != len(lst) - 1:
                    value += '\n'
            self.worksheet.set_column(col, col, (max_len + 10) if (max_len + 10) > 20 else 20)
            self.worksheet.write(row, col, value)
        return value

    def add_row(self, row: int, image: str, correct: list, pred: list, acc: list, loss: list, iou: list, split: str):
        # Get the image size
        w_pixel, h_pixel = Image.open(image).size

        # Calculate column size to fit image
        scale = round(300/h_pixel, 2)
        w_unit = self.DEFAULT_CHAR_WIDTH * w_pixel * scale
        # Init row set
        self.worksheet.set_row_pixels(row, 300 + 20)
        self.worksheet.set_column(0, 0, w_unit)
        self.worksheet.set_column_pixels(0, 0, w_pixel * scale + 20)

        # Write data
        # Image
        self.worksheet.insert_image(row, 0, image, {'object_position': 1, 'x_scale': scale, 'y_scale': scale})
        # Correct label
        self.write_list(row, 1, correct)
        # Predicted label
        self.write_list(row, 2, pred)
        # Accuracy
        self.write_list(row, 3, acc)
        # Loss
        self.worksheet.set_column(4,4, 16)
        self.worksheet.write(row, 4, loss)
        # IOU
        self.write_list(row, 5, iou)
        # Split
        self.worksheet.set_column(6,6, 16)
        self.worksheet.write(row, 6, split)

    def dispose(self):
        self.workbook.close()


class Util:
    def second_max(lst: list):
        lst.pop(lst.index(max(lst)))
        return max(lst)

    def second_min(lst: list):
        lst.pop(lst.index(min(lst)))
        return min(lst)

    def calc_iou(expect: list, pred: list):
        # Define points of bounding box expect(ABCD), pred(MNPQ)
        xC = expect[0] - expect[2] / 2
        yC = expect[1] - expect[3] / 2
        xA = xC + expect[2]
        yA = yC + expect[3]

        xP = pred[0] - pred[2] / 2
        yP = pred[1] - pred[3] / 2
        xM = xP + pred[2]
        yM = yP + pred[3]

        yB = yC
        xD = xC

        yN = yP
        xQ = xP

        # Intersect
        width_inter = min([xA, xM]) - max([xD, xQ])
        height_inter = min([yA, yM]) - max([yB, yN])
        intersect = width_inter * height_inter

        # Union
        area_ABCD = abs(yA-yB) * abs(xA-xD)
        area_MNPQ = abs(yM-yN) * abs(xM-xQ)
        union = area_ABCD + area_MNPQ - intersect

        return intersect / union

    def list_translate(list: list):
        result = ''
        for i in range(len(list)):
            result += str(list[i]) + ('<br>' if (i != len(list) - 1) else '')
        return result




class DataExport:
    '''Export trainned data into Excel spreadsheet with attributes: Image, Expected Label, Predicted Label, IOU, Accuracy, Loss, Split'''

    # DATA PREPARATION
    # Prepare data folder by following structure with source:

    # Steps to reproduce:
    # Run tweaked val.py with attributes: batch-size=1 (Add loss calculation and image processing order)
    # Run detect.py with attributes: --save-txt --hide-conf --hide-labels

    # Required files explanation:
    # val_loss.txt: Exported from tweaked val.py (Loss calculation [box, obj, cls])
    # img_order.txt: Exported from tweaked val.py ()
    # train, val, public_test: Images placed in root folder with 2 sub-folder containing matching label of predict and expect set.
    # Folders and sub-folder, 3 (img_order.txt, val_loss.txt) can be empty but not null (1)

    # TODO: Prepare auto-pipeline for ExportData.py by tweaking val.py and detect.py
    # TODO: Handling file exceptions in call() to minimize preparation (1)

    # Folder structure
    # \__Targeting
    #  \__public_test
    #     \__expect
    #     \__predict
    #     \__images
    #     \__img_order.txt
    #     \__val_loss.txt

    LABEL = ['no_mask', 'mask', 'incorrect_mask']
    def __init__(self, path: str) -> None:
        self.path = path
        self.image_path = '{}/public_test/images'

    def get_split_path(self, split: str) -> str:
        return '{}/{}'.format(self.path, split)

    def get_images_name(self, split: str) -> list:
        images = []
        try:
            with open(self.get_split_path(split) + '/img_order.txt') as img_order:
                for line in img_order.readlines():
                    data = line[line.rindex('/') + 1:]
                    data = data[:data.rindex('.')]
                    images.append(data)
            return images
        except:
            print('img_order.txt not exist in ' + self.get_split_path(split))

    def get_val_loss(self, split: str) -> list:
        val_loss = []
        try:
            with open(self.get_split_path(split) + '/val_loss.txt') as val_loss_file:
                for line in val_loss_file.readlines():
                    val_loss.append([round(float(i), 4) for i in line.split()])
            return val_loss
        except:
            print('val_loss.txt not exist in ' + self.get_split_path(split))

    def get_prediction(self, split: str, image: str) -> Tuple:
        prediction_label = []
        prediction_bbox = []
        with open(self.get_split_path(split) + '/predict/' + image + '.txt') as prediction:
            for line in prediction.readlines():
                data = line[1:]
                prediction_label.append(self.LABEL[int(line[0])])
                prediction_bbox.append(list(map(float, data.split())))
        return prediction_label, prediction_bbox

    def get_expectation(self, split: str, image: str) -> Tuple:
        expect_label = []
        expect_bbox = []
        with open(self.get_split_path(split) + '/expect/' + image + '.txt') as prediction:
            for line in prediction.readlines():
                data = line[1:]
                expect_label.append(self.LABEL[int(line[0])])
                expect_bbox.append(list(map(float, data.split())))
        return expect_label, expect_bbox

    def get_accuracy(self, expect: list, pred: list) -> list:
        return [int(ex == pd) for ex, pd in zip(expect, pred)]

    def get_iou(self, expect: list, pred: list) -> list:
        return [round(Util.calc_iou(ex, pr), 4) for ex, pr in zip(expect, pred)]

    def __call__(self) -> None:
        excel = ExcelWorker('test.xlsx')
        row = 1
        for spl in self.SPLIT_PATCH:
            val_loss = self.get_val_loss(spl)
            images = self.get_images_name(spl)
            for i in range(len(images)):
                expect_label, expect_bbox = self.get_expectation(spl, images[i])
                pred_label, pred_bbox = self.get_prediction(spl, images[i])
                print(val_loss[i])
                excel.add_row(
                    row,
                    '{}/{}.jpg'.format(self.get_split_path(spl), images[i]),
                    expect_label,
                    pred_label,
                    self.get_accuracy(expect_label, pred_label),
                    '\n'.join([str(j) for j in val_loss[i]]) + '\nTotal: ' + str(round(sum(val_loss[i]), 6)),
                    self.get_iou(expect_bbox, pred_bbox),
                    spl
                )
                row += 1
        excel.dispose()


class TestExportData(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.exporter = DataExport('test')

    def test_get_images_name(self):
        expect = ['10.35.17.101_01_20210709175146264_MOTION_DETECTION']
        output = self.exporter.get_images_name('public_test')
        self.assertEqual(expect, output)

    def test_get_val_loss(self):
        expect = [[0.035293575, 0.0059761214, 0.00043651924]]
        output = self.exporter.get_val_loss('public_test')
        self.assertEqual(expect, output)

    def test_get_prediction(self):
        expect_1, expect_2 = ['mask'], [[0.0710938, 0.542361, 0.0375, 0.0791667]]
        output_1, output_2 = self.exporter.get_prediction('public_test', self.exporter.get_images_name('public_test')[0])
        self.assertEqual(expect_1, output_1)
        self.assertEqual(expect_2, output_2)

    def test_get_expectation(self):
        expect_1, expect_2 = ['mask'], [[0.0722656, 0.534722, 0.0289063, 0.0833333]]
        output_1, output_2 = self.exporter.get_expectation('public_test', self.exporter.get_images_name('public_test')[0])
        self.assertEqual(expect_1, output_1)
        self.assertEqual(expect_2, output_2)

    def test_get_accuracy(self):
        expect = [1]
        pred = self.exporter.get_prediction('public_test', self.exporter.get_images_name('public_test')[0])[0]
        exp = self.exporter.get_expectation('public_test', self.exporter.get_images_name('public_test')[0])[0]
        output = self.exporter.get_accuracy(pred, exp)
        self.assertEqual(expect, output)

    def test_get_iou(self):
        expect = [0.6548]
        pred_bb = self.exporter.get_prediction('public_test', self.exporter.get_images_name('public_test')[0])[1]
        ex_bb = self.exporter.get_expectation('public_test', self.exporter.get_images_name('public_test')[0])[1]
        output = self.exporter.get_iou(ex_bb, pred_bb)
        self.assertEqual(expect, output)


if __name__ == '__main__':
    #unittest.main(verbosity=2)
    exporter = DataExport('content/DataComp/ExportData')
    exporter()
