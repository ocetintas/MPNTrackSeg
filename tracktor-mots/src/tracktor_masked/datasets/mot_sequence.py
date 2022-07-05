import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


import cv2
import pycocotools.mask as rletools

#from ..config import cfg
from mot_neural_solver.path_cfg import DATA_PATH
from torchvision.transforms import ToTensor



class MOT17Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(DATA_PATH, 'MOT17Det')
        self._label_dir = osp.join(DATA_PATH, 'MOT16Labels')
        self._raw_label_dir = osp.join(DATA_PATH, 'MOT16-det-dpm-raw')
        self._mot17_label_dir = osp.join(DATA_PATH, 'MOT17Labels')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))


        ########################
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            self.seq_path = osp.join(self._mot17_label_dir, 'train', seq_name + '-' + self._dets[:-2])

        else:
            self.seq_path = osp.join(self._mot17_label_dir, 'test', seq_name+ '-' + self._dets[:-2])
        ########################


        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
            label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
            mot17_label_path = osp.join(self._mot17_label_dir, 'train')
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)
            label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
            mot17_label_path = osp.join(self._mot17_label_dir, 'test')
        raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        det_file = self.get_det_file(label_path, raw_label_path, mot17_label_path)

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        for i in range(1,seqLength+1):
            im_path = osp.join(imDir,"{:06d}.jpg".format(i))

            sample = {'gt':boxes[i],
                      'im_path':im_path,
                      'vis':visibility[i],
                      'dets':dets[i],}

            total.append(sample)

        return total, no_gt

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        if self._dets == "DPM":
            det_file = osp.join(label_path, 'det', 'det.txt')
        elif self._dets == "DPM_RAW16":
            det_file = osp.join(raw_label_path, 'det', 'det-dpm-raw.txt')
        elif "17" in self._seq_name:
            det_file = osp.join(
                mot17_label_path,
                f"{self._seq_name}-{self._dets[:-2]}",
                'det',
                'det.txt')
        else:
            det_file = ""
        return det_file

    def __str__(self):
        return f"{self._seq_name}-{self._dets[:-2]}"

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        print("[*] Writing to: {}".format(output_dir))

        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        #if "17" in self._dets:
        #    file = osp.join(output_dir, 'MOT17-'+self._seq_name[6:8]+"-"+self._dets[:-2]+'.txt')
        #else:
        #    file = osp.join(output_dir, 'MOT16-'+self._seq_name[6:8]+'.txt')

        with open(output_dir, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])


class MOT19Sequence(MOT17Sequence):

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(DATA_PATH, 'MOT19')
        self._mot17_label_dir = osp.join(DATA_PATH, 'MOT19')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(DATA_PATH, 'MOT16Labels')
        self._raw_label_dir = osp.join(DATA_PATH, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        # FRCNN detections
        if "MOT19" in self._seq_name:
            det_file = osp.join(mot17_label_path, self._seq_name, 'det', 'det.txt')
        else:
            det_file = ""
        return det_file

    def write_results(self, all_tracks, output_dir):
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        #file = osp.join(output_dir, f'{self._seq_name}.txt')

        print("[*] Writing to: {}".format(output_dir))

        with open(output_dir, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


class MOT17LOWFPSSequence(MOT17Sequence):

    def __init__(self, split, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(DATA_PATH, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')
        self._mot17_label_dir = osp.join(DATA_PATH, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(DATA_PATH, 'MOT16Labels')
        self._raw_label_dir = osp.join(DATA_PATH, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = Compose([ToTensor(), Normalize(normalize_mean,
                                                         normalize_std)])

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

class MOTS20Sequence(MOT17Sequence):
    def __init__(self, seq_name=None, dets='', vis_threshold=1.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):

        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mots_dir = osp.join(DATA_PATH, 'MOTS20')
        self._train_folders = os.listdir(os.path.join(self._mots_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mots_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['dets_mask'] = data['dets_mask']
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['gt_mask'] = data['gt_mask']

        return sample

    def __str__(self):
        return f"{self._seq_name}"

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mots_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self._mots_dir, 'test', seq_name)

        self.seq_path = seq_path

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        train = []
        val = []

        boxes = {}
        masks = {}
        visibility = {}
        dets = {}
        dets_masks = {}

        for i in range(1, seqLength + 1):
            boxes[i] = {}
            masks[i] = {}
            visibility[i] = {}
            dets[i] = []
            dets_masks[i] = []

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=' ')
                for row in reader:

                    # row  = frame, objectid, classid, height, width, rlemask
                    if int(row[1]) != 10000:
                        # mask
                        rle_mask = {'size': [int(row[3]), int(row[4])], 'counts': row[5].encode(encoding='UTF-8')}
                        #mask = rletools.decode(rle_mask)

                        # bounding box
                        box = rletools.toBbox(rle_mask)  # x,y,h,w
                        bb = np.array([box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1],
                                      dtype=np.float32)  # x1,y1,x2,y2

                        # if (bb[2] > rle_mask['size'][1] or bb[3] > rle_mask['size'][0]):
                        #     print('BBOX OUTSIDE FOUND IN GT!!!')

                        boxes[int(row[0])][int(row[1])] = bb
                        masks[int(row[0])][int(row[1])] = rle_mask

        else:
            no_gt = True

        det_file = osp.join(seq_path, 'det', 'det.txt')
        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=' ')
                for row in reader:
                    # row = frame bb_left bb_top bb_width bb_height conf label img_height img_width rle
                    if int(row[6]) == 2:
                        rle_mask = {'size': [int(row[7]), int(row[8])], 'counts': row[9].encode(encoding='UTF-8')}

                        x1 = float(row[1]) - 1
                        y1 = float(row[2]) - 1
                        x2 = x1 + float(row[3]) - 1
                        y2 = y1 + float(row[4]) - 1

                        score = float(row[5])

                        # bounding box
                        bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                        dets[int(float(row[0]))].append(bb)
                        dets_masks[int(float(row[0]))].append(rle_mask)

                        # if bb[2] > rle_mask['size'][1] or bb[3] > rle_mask['size'][0]:
                        #     print('BBOX OUTSIDE FOUND IN DET!!!')

        for i in range(1, seqLength + 1):
            im_path = osp.join(imDir, "{:06d}.jpg".format(i))

            sample = {
                'gt': boxes[i],
                'gt_mask': masks[i],
                'im_path': im_path,
                'dets': dets[i],
                'dets_mask': dets_masks[i]
            }

            total.append(sample)

        return total, no_gt

    def write_results(self, all_tracks, output_dir):

        """Write the tracks in the format of MOTS20 submission
        all_tracks: dictionary with 1 dictionary for every track with {..., i:[bbox, mask, score ] at key track_num
        Each file contains these lines:
        <time_frame> <id> <class_id> <img_height> <img_width> <rle>
        # TODO: change the names of files to submit
        Files to submit:
        ?
        """

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"
        print("[*] Writing to: {}".format(output_dir))

        # writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
        #
        # with open(output_dir, "w") as of:
        #
        #     writer = csv.writer(of, delimiter=' ')
        #
        #     for i, track in all_tracks.items():
        #
        #         for frame, segm in track.items():
        #             # determine image dimensions from mask
        #             mask = segm[1]
        #
        #             class_id = 2  # we have only pedestrians
        #             ped_id = class_id * 1000 + i + 1  # mots notation
        #
        #             writer.writerow(
        #                 # <time_frame> <id> <class_id> <img_height> <img_width> <rle>
        #                 [frame + 1, ped_id, class_id, mask['size'][0], mask['size'][1],
        #                  mask['counts'].decode(encoding='UTF-8')])

        with open(output_dir, "w") as of:
            writer = csv.writer(of, delimiter=' ')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]

                    class_id = 2  # we have only pedestrians
                    ped_id = class_id * 1000 + i + 1  # mots notation
                    conf = 1

                    writer.writerow([frame+1, ped_id, x1+1, y1+1, x2-x1+1, y2-y1+1, conf, class_id])
