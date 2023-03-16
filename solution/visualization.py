import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from utils import get_data


def viz(ground_truth):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    # paths = glob.glob('../data/images/*')
    paths = glob.glob('UDautonomous/data/images/*')

    # mapping to access data faster
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt

    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}

    f, ax = plt.subplots(4, 5, figsize=(20, 10))
    for i in range(20):
        x = i % 4
        y = i % 5

        filename = os.path.basename(paths[i])
        img = Image.open(paths[i])
        ax[x, y].imshow(img)

        bboxes = gtdic[filename]['boxes']
        classes = gtdic[filename]['classes']
        for cl, bb in zip(classes, bboxes):
            y1, x1, y2, x2 = bb
            rec = Rectangle((x1, y1), x2- x1, y2-y1, facecolor='none', 
                            edgecolor=colormap[cl])
            ax[x, y].add_patch(rec)
        ax[x ,y].axis('off')
    plt.tight_layout()
    plt.show()

import numpy as np    
def viz_compare(ground_truth, predictions):
    """
    create a grid visualization of images with color coded bboxes
    args:
    - ground_truth [list[dict]]: ground truth data
    """
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    path = 'UDautonomous/data/images/'
    
    # paths = glob.glob('../data/images/*')
    # paths = glob.glob('UDautonomous/data/images/*')

    # mapping to access data faster
    gtdic = {}
    for gt in ground_truth:
        gtdic[gt['filename']] = gt        
    pdic = {}
    for p in predictions:
        pdic[p['filename']] = p

    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}

    f, ax = plt.subplots(1, 2, figsize=(20, 10))    
    for i in range(2):
        # x = i % 1
        y = i % 2
        
        img = Image.open(path+filename)
        ax[y].imshow(img)

        if i ==0:
            gt_bboxes = gtdic[filename]['boxes']
            gt_classes = gtdic[filename]['classes']
            for cl, bb in zip(gt_classes, gt_bboxes):
                y1, x1, y2, x2 = bb
                rec = Rectangle((x1, y1), x2- x1, y2-y1, facecolor='none', 
                                edgecolor=colormap[cl])
                # ax[x, y].add_patch(rec)                    
                ax[y].add_patch(rec)                    
        else: 
            p_bboxes = pdic[filename]['boxes']
            p_classes = pdic[filename]['classes']
            for cl, bb in zip(p_classes, p_bboxes):
                y1, x1, y2, x2 = bb
                rec = Rectangle((x1, y1), x2- x1, y2-y1, facecolor='none', 
                                edgecolor=colormap[cl])
                # ax[x, y].add_patch(rec)
                ax[y].add_patch(rec)
        # ax[x ,y].axis('off')        
        ax[y].axis('off')        
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    viz(ground_truth)   # multi-ground truth
    viz_compare(ground_truth, predictions)   # comparing a pair of ground truth and prediction
    