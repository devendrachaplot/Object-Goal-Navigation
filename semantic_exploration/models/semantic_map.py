import numpy as np
import third_party.semantic_exploration.envs.utils.depth_utils as du
import torch
import torch.nn as nn
from torch.nn import functional as F


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, config):
        super(Semantic_Mapping, self).__init__()

        self.device = config.DEVICE
        self.screen_h = config.FRAME_HEIGHT
        self.screen_w = config.FRAME_WIDTH
        self.resolution = config.MAP_RESOLUTION
        self.z_resolution = config.MAP_RESOLUTION
        self.map_size_cm = config.MAP_SIZE_CM // config.GLOBAL_DOWNSCALING
        self.n_channels = 3
        self.vision_range = config.VISION_RANGE
        self.dropout = 0.5
        self.fov = config.HFOV
        self.du_scale = config.DU_SCALE
        self.cat_pred_threshold = config.CAT_PRED_THRESHOLD
        self.exp_pred_threshold = config.EXP_PRED_THRESHOLD
        self.map_pred_threshold = config.MAP_PRED_THRESHOLD
        self.num_sem_categories = config.NUM_SEM_CATEGORIES

        self.max_height = int(180 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = config.CAMERA_HEIGHT * 100.0
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov
        )

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = (
            torch.zeros(
                config.NUM_PROCESSES,
                1 + self.num_sem_categories,
                vr,
                vr,
                self.max_height - self.min_height,
            )
            .float()
            .to(self.device)
        )
        self.feat = (
            torch.ones(
                config.NUM_PROCESSES,
                1 + self.num_sem_categories,
                self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            )
            .float()
            .to(self.device)
        )

    def forward(self, obs, pose_obs, maps_last, poses_last):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale
        )

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device
        )

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device
        )

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - vision_range // 2.0) / vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (
            (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.0) / (max_h - min_h) * 2.0
        )
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]).view(
            bs, c - 4, h // self.du_scale * w // self.du_scale
        )

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(
            self.init_grid * 0.0, self.feat, XYZ_cm_std
        ).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(
            bs,
            c,
            self.map_size_cm // self.resolution,
            self.map_size_cm // self.resolution,
        ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range

        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold, min=0.0, max=1.0
        )

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * torch.sin(
                pose[:, 2] / 57.29577951308232
            ) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * torch.cos(
                pose[:, 2] / 57.29577951308232
            ) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = -(
            st_pose[:, :2] * 100.0 / self.resolution
            - self.map_size_cm // (self.resolution * 2)
        ) / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = self.get_grid(st_pose, agent_view.size(), self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # Remove people in the last map if found new people
        if translated[:, 19, :, :].sum() > 0.99:
            maps_last[:, 19, :, :] = 0

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        if np.sum(np.array(map_pred)[0, 1, :, :]) == 0:
            import pdb

            pdb.set_trace()

        return fp_map_pred, map_pred, pose_pred, current_poses

    @staticmethod
    def get_grid(pose, grid_size, device):
        """
        Input:
            `pose` FloatTensor(bs, 3)
            `grid_size` 4-tuple (bs, _, grid_h, grid_w)
            `device` torch.device (cpu or gpu)
        Output:
            `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
            `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

        """
        pose = pose.float()
        x = pose[:, 0]
        y = pose[:, 1]
        t = pose[:, 2]

        t = t * np.pi / 180.0
        cos_t = t.cos()
        sin_t = t.sin()

        theta11 = torch.stack(
            [cos_t, -sin_t, torch.zeros(cos_t.shape).float().to(device)], 1
        )
        theta12 = torch.stack(
            [sin_t, cos_t, torch.zeros(cos_t.shape).float().to(device)], 1
        )
        theta1 = torch.stack([theta11, theta12], 1)

        theta21 = torch.stack(
            [torch.ones(x.shape).to(device), -torch.zeros(x.shape).to(device), x], 1
        )
        theta22 = torch.stack(
            [torch.zeros(x.shape).to(device), torch.ones(x.shape).to(device), y], 1
        )
        theta2 = torch.stack([theta21, theta22], 1)

        rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
        trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

        return rot_grid, trans_grid
