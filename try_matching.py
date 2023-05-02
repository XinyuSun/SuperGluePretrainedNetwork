from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import time
import numpy as np

from superglue.models.matching import Matching
from superglue.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    opt = parser.parse_args()
    print(opt)

    device = 'cuda'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    
    target_image1 = cv2.imread("1.png", 0)
    source_image1 = cv2.imread("2.png", 0)
    target_image2 = cv2.imread("3.png", 0)
    source_image2 = cv2.imread("4.png", 0)
    target_image1 = cv2.resize(target_image1, (128, 128))
    source_image1 = cv2.resize(source_image1, (128, 128))
    target_image2 = cv2.resize(target_image2, (128, 128))
    source_image2 = cv2.resize(source_image2, (128, 128))
    # image = cv2.resize(image, (w_new, h_new),
    #                            interpolation=self.interp)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    target_tensor1 = frame2tensor(target_image1, device)
    source_tensor1 = frame2tensor(source_image1, device)
    target_tensor2 = frame2tensor(target_image2, device)
    source_tensor2 = frame2tensor(source_image2, device)
    
    # ts = time.time()
    # target_kp = matching.superpoint({'image': target_tensor})
    # target_kp = {k+'0': target_kp[k] for k in keys}
    # target_kp["image0"] = target_tensor
    
    # source_kp = matching.superpoint({'image': source_tensor})
    # source_kp = {k+'1': source_kp[k] for k in keys}
    # source_kp["image1"] = source_tensor
    
    # tu = time.time() - ts
    # print(f"time used to extract superpoint: {tu}s")

    
    # ts = time.time()
    # pred = {**target_kp, **source_kp, **matching({**target_kp, **source_kp})}
    # tu = time.time() - ts
    
    # target_tensor = frame2tensor(target_image, device)
    target_tensor = torch.cat([target_tensor1, target_tensor2], dim=0)
    # source_tensor = frame2tensor(target_image, device)
    source_tensor = torch.cat([source_tensor1, source_tensor2], dim=0)
    
    ts = time.time()
    pred = matching({'image0': target_tensor, 'image1': source_tensor})
    tu = time.time() - ts
    
    print(f"time used to extract superpoint and run matching: {tu}s")
    
    kpts0 = pred['keypoints0'][1].cpu().numpy()
    kpts1 = pred['keypoints1'][1].cpu().numpy()
    matches = pred['matches0'][1].cpu().numpy()
    confidence = pred['matching_scores0'][1].cpu().numpy()
    
    valid = np.bitwise_and(matches > -1, confidence > 0.8)
    sorted = np.argsort(confidence[valid])[::-1][:16]
    mkpts0 = kpts0[valid][sorted]
    mkpts1 = kpts1[matches[valid][sorted]]
    color = cm.jet(confidence[valid])
    
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(1, 2),
    ]
    
    out = make_matching_plot_fast(
        target_image2, source_image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
    
    cv2.imwrite("match.png", out)