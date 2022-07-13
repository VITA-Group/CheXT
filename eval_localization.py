
from email.policy import default
from importlib.resources import path
import sys

from dataset import ChestXRay14
from utils import *
from collections import OrderedDict, defaultdict

import argparse

import cv2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.loss import FocalLoss, NTXentLoss
from models.radiomic import extract_radiomic_features
from models.radiomic_mlp import RaMLP
from models.chext import CheXT
from models.vision_transformer import vit_small

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion',
               'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

def get_IoU(truth_coords, pred_coords):
    # coords of intersection rectangle
    x1 = max(truth_coords[0], pred_coords[0])
    y1 = max(truth_coords[1], pred_coords[1])
    x2 = min(truth_coords[2], pred_coords[2])
    y2 = min(truth_coords[3], pred_coords[3])
    # area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # area of prediction and truth rectangles
    boxTruthArea = (truth_coords[2] - truth_coords[0] + 1) * (truth_coords[3] - truth_coords[1] + 1)
    boxPredArea = (pred_coords[2] - pred_coords[0] + 1) * (pred_coords[3] - pred_coords[1] + 1)
    # intersection over union 
    iou = interArea / float(boxTruthArea + boxPredArea - interArea)
    return iou

def generate_mask(attention):
    blur = cv2.GaussianBlur(np.uint8(255 * attention),(5,5),0)
    T, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, 1, 2)

    list_BB = []
    for item in range(len(contours)):
        cnt = contours[item]
        x,y,w,h = cv2.boundingRect(cnt)

        list_BB.append([x, y, x+w, y+h])
    return list_BB

def iou_eval(image_id, disease_label, attention, gt_box, args):
    max_iou = 0
    index = 0
    list_BB = generate_mask(attention)
    for i, pred_box in enumerate(list_BB):
        iou = get_IoU(gt_box, pred_box)
        if iou > max_iou:
            index = i
            max_iou = iou
    best_pred_bb = list_BB[index]
    img = cv2.imread(os.path.join(args.image_path, image_id))
    img = cv2.resize(img, (224, 224))
    cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), color=(0, 0, 255), thickness=2)
    cv2.rectangle(img, (int(best_pred_bb[0]), int(best_pred_bb[1])), (int(best_pred_bb[2]), int(best_pred_bb[3])), color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(args.output_path, image_id + '_' + disease_label + '_pred.png'), img)

    return max_iou

def eval(model, args):
    model.eval()
    bbox_df = pd.read_csv(args.bbox_csv_path)
    iou_results = defaultdict(list)
    for index, row in bbox_df.iterrows():
        image_id = row['Image Index']
        disease_label = row['Finding Label']
        xmin, ymin, xmax, ymax = row['Bbox [x'], row['y'], row['Bbox [x']+row['w'], row['y']+row['h]']
        gt_box = [xmin, ymin, xmax, ymax]
        img = cv2.imread(os.path.join(args.image_path, image_id), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (224, 224))
        img = np.stack([img, img, img], axis=-1)
        img = transform(img)
        _, attn = model.ViT(img.unsqueeze(0).to(args.device))
        nh = attn.shape[1]

        attentions = attn[:, :, 0, 1:].reshape(1, nh, 14, 14)
        attentions = F.interpolate(attentions, scale_factor=16, mode="nearest").cpu().detach().numpy()
        attentions = attentions.mean(axis=0)
        if args.threshold != 0:
            percentiles = np.percentile(attentions, args.threshold, axis=0)
            attentions = [np.clip(attentions[i], a_min=percentiles[i], a_max=None) for i in range(1)]

        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        attentions = np.array(list(map(normalize, attentions)))

        attention = attentions[0]
        iou_score = iou_eval(image_id, disease_label, attention, gt_box, args)
        iou_results[disease_label].append(iou_score)

    return iou_results
        

def main(args):

    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create model and load weights
    model = vit_small(patch_size=16, num_classes=0)
    ramlp = RaMLP()

    model = CheXT(model, ramlp, 2, 8, [384, 107])

    state_dict = torch.load(args.model_weight_path)

    msg = model.load_state_dict(state_dict, strict=False)
    # print(
    #     "Pretrained weights found at {} and loaded with msg: {}".format(
    #         args.weights, msg
    #     )
    # )
    model = model.to(args.device)

    IoU_results = eval(model, args)
    return IoU_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight_path', type=str, default='/home/yh9442/yan/chext/results/chext_aug_pretr_lr-0.0001_bs-128_ls-0.1_wd1e-5_drp-fc/chkpt_epoch-31.pt')
    parser.add_argument('--image_path', type=str, default='/home/yh9442/yan/CXR14/images/')
    parser.add_argument('--bbox_csv_path', type=str, default='/home/yh9442/yan/chext_yolov5/predictions/BBox_List_2017.csv')
    parser.add_argument('--output_path', type=str, default='/home/yh9442/yan/chext/visualization/')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--device', type=str, default='cuda:7')
    args = parser.parse_args()

    if args.output_path != '':
        if os.path.isdir(args.output_path):
            shutil.rmtree(args.output_path)
        os.mkdir(args.output_path)

    IoU_results = main(args)

    print(IoU_results)

    # for key, value in IoU_results.items():
    #     print(key)
    #     print(value)