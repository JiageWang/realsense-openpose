import time
from multiprocessing import Process

import numpy as np
import torch
import cv2

from config import *
from utils import normalize, pad_width
from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules.keypoints import extract_keypoints, group_keypoints
from .modules.load_state import load_state
from .modules.pose import Pose, track_poses


class OpenposeLight(object):
    STRIDE = OPENPOSE_STRIDE
    SMOOTH = OPENPOSE_SMOOTH
    HEIGHT_SIZE = OPENPOSE_HEIGHT_SIZE
    UPSAMPLE_RATIO = OPENPOSE_UPSAMPLE_RATIO
    IMG_MEAN = OPENPOSE_IMG_MEAN
    IMG_SCALE = OPENPOSE_IMG_SCALE
    PAD_VALUE = OPENPOSE_PAD_VALUE
    THRESHOLD = OPENPOSE_THRESHOLD
    NUM_KEYPOINTS = Pose.num_kpts

    def __init__(self, checkpoints_path):
        self.previous_poses = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(checkpoints_path)

    def load_model(self, checkpoints_path):
        model = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoints_path, map_location='cpu')
        load_state(model, checkpoint)
        return model.to(self.device).eval()

    def predict(self, img):
        tensor_img, scale, pad = self.preprocess(img)
        heatmaps, pafs = self.forward_model(tensor_img)
        current_poses = self.postprocess(heatmaps, pafs, scale, pad)
        # track
        track_poses(self.previous_poses, current_poses, smooth=self.SMOOTH)
        self.previous_poses = current_poses
        return current_poses

    def render(self, img, poses=None):
        if not poses:
            poses = self.predict(img)
        self.draw_poses(img, poses)

    @classmethod
    def draw_poses(cls, img, poses, dists=None):
        for i, pose in enumerate(poses):
            if pose.confidence < cls.THRESHOLD:
                continue
            pose.draw(img)
            # # 正常
            # rect_color = (0, 255, 0)
            # # 未佩戴安全帽
            # if not pose.has_helmet:
            #     rect_color = (0, 255, 255)
            # # 未佩戴安全帽进入roi
            # if dists and dists[i] > 0 and not pose.has_helmet:
            #     rect_color = (0, 0, 255)
            # # rect_color = (0, 0, 255) if dists and dists[i] > 0 else (255, 0, 0)
            # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
            #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), rect_color, thickness=1)
            # # cv2.rectangle(img, (pose.head_bbox[0], pose.head_bbox[1]),
            # #               (pose.head_bbox[0] + pose.head_bbox[2], pose.head_bbox[1] + pose.head_bbox[3]), rect_color,
            # #               thickness=1)
        return img

    def preprocess(self, img):
        height, width, _ = img.shape
        scale = self.HEIGHT_SIZE / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, self.IMG_MEAN, self.IMG_SCALE)
        min_dims = [self.HEIGHT_SIZE, max(scaled_img.shape[1], self.HEIGHT_SIZE)]
        padded_img, pad = pad_width(scaled_img, self.STRIDE, self.PAD_VALUE, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        tensor_img = tensor_img.to(self.device)
        return tensor_img, scale, pad

    def forward_model(self, tensor_img):
        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.UPSAMPLE_RATIO, fy=self.UPSAMPLE_RATIO,
                              interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.UPSAMPLE_RATIO, fy=self.UPSAMPLE_RATIO, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs

    def postprocess(self, heatmaps, pafs, scale, pad):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.NUM_KEYPOINTS):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.STRIDE / self.UPSAMPLE_RATIO - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.STRIDE / self.UPSAMPLE_RATIO - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.NUM_KEYPOINTS, 2), dtype=np.int32) * -1
            for kpt_id in range(self.NUM_KEYPOINTS):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        return current_poses


class OpenposeProcess(Process):
    def __init__(self, q_imgs, q_poses, checkpoints_path):
        super(OpenposeProcess, self).__init__()
        self.q_imgs = q_imgs
        self.q_poses = q_poses
        self.checkpoints_path = checkpoints_path

    def run(self):
        openpose = OpenposeLight(self.checkpoints_path)
        while True:
            # get阻塞直到有输入
            start_time = time.time()
            poses = openpose.predict(self.q_imgs.get())
            self.q_poses.put(poses)
            # print("model: openpose, pid: {}, process time: {}".format(os.getpid(), time.time() - start_time))
