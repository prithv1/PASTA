# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import random
import glob
import imagesize
import functools
import os.path as osp
from pathlib import Path

import cityscapesscripts.helpers.labels as CSLabels
import xmltodict
import xml.etree.ElementTree as ET

import mmcv
import numpy as np


SEED = 42
def collect_files(img_dir, gt_dir, subsample=None):
    files = []
    for img_file in Path(img_dir).glob("*.jpg"):
                
        gt_file = Path(gt_dir) / img_file.with_suffix(".xml").name
        segm_file = ""
        if gt_file.exists():
            files.append([img_file, gt_file, segm_file])
            
    print(f'Loaded {len(files)}')

    if subsample is not None:
        random.seed(SEED)
        n = int(subsample * len(files))
        random.shuffle(files)
        files = files[:n]

    return files


def collect_annotations(files, perturbor, nproc=1):
    print('Loading annotation images')

    f = functools.partial(load_img_info, perturbor=perturbor)
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            f, files, nproc=nproc)
    else:
        images = mmcv.track_progress(f, files)

    return images


def load_img_info(files, perturbor):

    img_file, gt_file, segm_file = files
    
    img_w, img_h = imagesize.get(img_file)
    
    
    # Read boxes
    tree = ET.parse(gt_file)
    xml_data = tree.getroot()
    xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')

    data_dict = dict(xmltodict.parse(xmlstr))

    bboxes = data_dict['annotation']['object']
    if not isinstance(bboxes, list):
        bboxes = [bboxes]

    new_bboxes = []
    category_ids = []
    for bbox in bboxes:
        display_class_name = bbox['name']
        category_id = 26
        x_top_left = int(bbox['bndbox']['xmin'])
        y_top_left = int(bbox['bndbox']['ymin'])
        box_w = int(bbox['bndbox']['xmax']) - int(bbox['bndbox']['xmin'])
        box_h = int(bbox['bndbox']['ymax']) - int(bbox['bndbox']['ymin'])
        
        box = [x_top_left, y_top_left, box_w, box_h]
        new_bboxes.append(box)
        category_ids.append(category_id)

        
    anno_info = []
    for bbox, category_id in zip(new_bboxes, category_ids):

        anno = dict(
            iscrowd=0,
            category_id=category_id,
            bbox=bbox,
            area=bbox[2] * bbox[3], 
            segmentation=dict(size=[0,0], counts="")
        )
        anno_info.append(anno)
    
    
    img_info = dict(
        # remove img_prefix for filename
        file_name=img_file.name,
        segm_file=segm_file, 
        height=img_h,
        width=img_w,
        anno_info=anno_info,
    )
    
    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    for label in CSLabels.labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to COCO format')
    parser.add_argument('--sim10k_path', help='sim10k data path')
    parser.add_argument('--img-dir', default='VOC2012/JPEGImages', type=str)
    parser.add_argument('--gt-dir', default='VOC2012/Annotations', type=str)
    parser.add_argument('--subsample', default=1., type=float)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--drop", type=str, default="no", help="[param]-[criterion(small,truncated,occluded)]")
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    sim10k_path = args.sim10k_path
    out_dir = args.out_dir if args.out_dir else sim10k_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(sim10k_path, args.img_dir)
    gt_dir = osp.join(sim10k_path, args.gt_dir)

    set_name = dict(
        train='voc2012_annotations.json')

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It took {}s to convert Sim10k annotation'):

            
            files = collect_files(img_dir, gt_dir, subsample=args.subsample)
            image_infos = collect_annotations(files, perturbor=None, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))

            
if __name__ == '__main__':
    main()