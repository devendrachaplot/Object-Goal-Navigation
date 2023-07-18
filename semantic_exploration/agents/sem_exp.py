# -*- coding: utf-8 -*-
import math
import os

import third_party.semantic_exploration.agents.utils.visualization as vu
import cv2
import third_party.semantic_exploration.envs.utils.pose as pu
from third_party.semantic_exploration.envs.utils.fmm_planner import FMMPlanner
import numpy as np
import skimage.morphology
from third_party.semantic_exploration.agents.utils.detic_semantic_prediction import SemanticPredDetic
from third_party.semantic_exploration.agents.utils.owlvit_semantic_prediction import SemanticPredOwlvit
from third_party.semantic_exploration.agents.utils.semantic_prediction import SemanticPredMaskRCNN
from third_party.semantic_exploration.constants import color_palette
from PIL import Image
from torchvision import transforms


class Sem_Exp_Env_Agent:
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, config, rank=1):
        self.config = config
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH),
                    interpolation=Image.NEAREST,
                ),
            ]
        )

        if self.config.DETECTION_MODEL == "detectron2":
            self.sem_pred = SemanticPredMaskRCNN(self.config)
        elif self.config.DETECTION_MODEL == "detic":
            self.sem_pred = SemanticPredDetic(self.config)
        elif self.config.DETECTION_MODEL == "owlvit":
            self.sem_pred = SemanticPredOwlvit(self.config)
        else:
            raise NotImplementedError

        # initializations for planning:
        self.selem = skimage.morphology.disk(self.config.OBS_DILATION_SELEM_RADIUS)
        self.obs = None
        self.info = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        if self.config.PLANNER == "frontier":
            self.start_obs_dilation_selem_radius = self.config.OBS_DILATION_SELEM_RADIUS
            self.goal_dilation_selem_radius = self.config.GOAL_DILATION_SELEM_RADIUS
            self.min_obs_dilation_selem_radius = (
                self.config.MIN_OBS_DILATION_SELEM_RADIUS
            )
            self.agent_cell_radius = self.config.AGENT_CELL_RADIUS
            self.goal_tolerance = self.config.GOAL_TOLERANCE
            self.continuous_angle_tolerance = self.config.CONTINUOUS_ANGLE_TOLERANCE
            self.curr_obs_dilation_selem_radius = None
            self.obs_dilation_selem = None

        if self.config.VISUALIZE:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            semantic_exploration_dir = os.path.join(os.path.dirname(this_dir))
            self.legend = cv2.imread(semantic_exploration_dir+"/docs/legend.png")
            self.vis_image = None
            self.rgb_vis = None
            self.depth_vis = None
        self.goal_name = None
        self.timestep = 0
        self.rank = rank
        self.episode_no = 0
        self.cur_stg = None

    def reset(self, obs_size, goal_name):
        self.info = None
        self.obs_shape = obs_size
        self.goal_name = goal_name

        # Episode initializations
        map_shape = (
            self.config.MAP_SIZE_CM // self.config.MAP_RESOLUTION,
            self.config.MAP_SIZE_CM // self.config.MAP_RESOLUTION,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [
            self.config.MAP_SIZE_CM / 100.0 / 2.0,
            self.config.MAP_SIZE_CM / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None

        if self.config.PLANNER == "frontier":
            self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
            self.obs_dilation_selem = skimage.morphology.disk(
                self.curr_obs_dilation_selem_radius
            )

        if self.config.VISUALIZE:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        self.timestep = 0

    def update_vis_image_goal(self, goal_name):
        self.goal_name = goal_name
        if self.config.VISUALIZE:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

    def plan_act_and_preprocess(self, planner_inputs, info):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        self.info = info
        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            return np.zeros(self.obs.shape), 0.0, False, self.info

        action = self._plan(planner_inputs)

        if self.config.VISUALIZE:
            self._visualize(planner_inputs)

        self.timestep += 1

        if action >= 0:
            # act
            action = {"action": action}
            obs = self.info["state"]
            self.last_action = action["action"]
            self.obs = obs
            self.info = info
            self.info["action"] = action

            return obs, 0.0, False, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            self.info["action"] = -1
            return np.zeros(self.obs_shape), 0.0, False, self.info

    def _reach_goal_if_in_map(self, goal_map, found_goal):
        height = goal_map.shape[0]
        width = goal_map.shape[1]
        init_goal_map = np.zeros((height, width))
        if found_goal:
            init_goal_map = goal_map
        return init_goal_map

    def _explore_otherwise(self, exp_pred, goal_map, found_goal):
        """Explore closest unexplored region otherwise."""
        # Select unexplored area
        frontier_map = exp_pred == 0
        self.dilate_explored_kernel = skimage.morphology.disk(10)
        # Dilate explored area
        frontier_map = 1 - skimage.morphology.binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        self.select_border_kernel = skimage.morphology.disk(1)
        # Select the frontier
        frontier_map = (
            skimage.morphology.binary_dilation(frontier_map, self.select_border_kernel)
            - frontier_map
        )

        if not found_goal:
            goal_map = frontier_map

        return goal_map

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """

        self.last_loc = self.curr_loc

        # Get Map prediction (obstacle)
        map_pred = np.rint(planner_inputs["map_pred"])
        if self.config.PLANNER == "frontier":
            goal = self._reach_goal_if_in_map(
                planner_inputs["goal"], planner_inputs["found_goal"]
            )
            goal = self._explore_otherwise(
                planner_inputs["exp_pred"], goal, planner_inputs["found_goal"]
            )
        else:
            goal = planner_inputs["goal"]

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / self.config.MAP_RESOLUTION - gx1),
            int(c * 100.0 / self.config.MAP_RESOLUTION - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if self.config.VISUALIZE:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [
                int(r * 100.0 / self.config.MAP_RESOLUTION - gx1),
                int(c * 100.0 / self.config.MAP_RESOLUTION - gy1),
            ]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
                last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
            )

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self.config.COLLISION_THRESHOLD:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * (
                            (i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1))
                        )
                        wy = y1 + 0.05 * (
                            (i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1))
                        )
                        r, c = wy, wx
                        r, c = int(r * 100 / self.config.MAP_RESOLUTION), int(
                            c * 100 / self.config.MAP_RESOLUTION
                        )
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, replan, stop = self._get_stg(
            map_pred, start, np.copy(goal), planning_window
        )

        # We were not able to find a path to the high-level goal
        if replan and self.config.PLANNER == "frontier":
            # Clean collision map
            self.collision_map *= 0

            # Reduce obstacle dilation
            if self.curr_obs_dilation_selem_radius > 1:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )

        # Deterministic Local Policy
        if stop and planner_inputs["found_goal"] == 1:
            if self._get_distance_to_obstacle() <= 0.2:
                action = 0
            else:
                action = 1
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.config.TURN_ANGLE / 2.0:
                # Right
                action = 3
            elif relative_angle < -self.config.TURN_ANGLE / 2.0:
                # Left
                action = 2
            else:
                # Forward
                action = 1

        self.cur_stg = stg

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        if self.config.PLANNER == "frontier":
            obstacles = grid[x1:x2, y1:y2]
            # Dilate obstacles
            dilated_obstacles = cv2.dilate(
                obstacles, self.obs_dilation_selem, iterations=1
            )
            traversible = 1 - dilated_obstacles
        else:
            traversible = (
                skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem)
                != True  # noqa
            )
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible, step_size=self.config.PLANNER_STEP_SIZE)
        # Set the goal size
        selem = skimage.morphology.disk(self.config.GOAL_DILATION_SELEM_RADIUS)
        goal = skimage.morphology.binary_dilation(goal, selem) != True  # noqa
        goal = 1 - goal * 1.0
        planner.set_multi_goal(goal)


        if self.config.VISUALIZE:
            dump_dir = "{}/dump/{}/".format(self.config.DUMP_LOCATION, self.config.EXP_NAME)
            ep_dir = "{}/episodes/thread_{}/eps_{}/".format(
                dump_dir, self.rank, self.episode_no
            )
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)
            r, c = traversible.shape
            dist_vis = np.zeros((r, c * 3))
            dist_vis[:, :c] = np.flipud(traversible)
            dist_vis[:, c : 2 * c] = np.flipud(goal)
            dist_vis[:, 2 * c :] = np.flipud(planner.fmm_dist / planner.fmm_dist.max())

            fn = "{}/episodes/thread_{}/eps_{}/frontier-{}-{}-Vis-{}.png".format(
                dump_dir,
                self.rank,
                self.episode_no,
                self.rank,
                self.episode_no,
                self.timestep,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.3
            color = (0, 0, 255)  # BGR
            thickness = 1
            dist_vis = cv2.cvtColor((255.0 * dist_vis).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            dist_vis = cv2.putText(dist_vis, "trav. (w: trav.; b: can't tarv.)", (2, 25), font, fontScale, color, thickness, cv2.LINE_AA)
            dist_vis = cv2.putText(dist_vis, "goal (w: goal; b: non-goal)", (c+2,25), font, fontScale, color, thickness, cv2.LINE_AA)
            dist_vis = cv2.putText(dist_vis, "trav.+goal (w: non-goal target; b: goal target)", (2*c+2,25), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(fn, dist_vis.astype(np.uint8))
            cv2.waitKey(1)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # Add the replan flag
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), use_seg=use_seg)
        self.depth_vis = depth
        depth = self._preprocess_depth(
            depth, self.config.MIN_DEPTH, self.config.MAX_DEPTH
        )

        ds = (
            self.config.ENV_FRAME_WIDTH // self.config.FRAME_WIDTH
        )  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2 :: ds, ds // 2 :: ds]
            sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred

    def _get_distance_to_obstacle(self):
        """"Return the distance between the obstacle and the robot."""
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        dist = pu.get_l2_distance(x1, x2, y1, y2)
        return dist


    def _visualize(self, inputs):
        dump_dir = "{}/dump/{}/".format(self.config.DUMP_LOCATION, self.config.EXP_NAME)
        ep_dir = "{}/episodes/thread_{}/eps_{}/".format(
            dump_dir, self.rank, self.episode_no
        )
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs["map_pred"]
        exp_pred = inputs["exp_pred"]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs["pose_pred"]

        goal = inputs["goal"]
        goal[int(self.cur_stg[0]), int(self.cur_stg[1])] = 1
        sem_map = inputs["sem_map_pred"]

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == self.config.NUM_SEM_CATEGORIES + 4  # 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(self.goal_dilation_selem_radius)
        goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True  # noqa

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.0) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(
            sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        self.depth_vis = cv2.cvtColor((255.0 * self.depth_vis).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.vis_image[
            50 : 50 + self.config.ENV_FRAME_HEIGHT,
            15 : 15 + self.config.ENV_FRAME_WIDTH,
        ] = self.rgb_vis # depth_vis or rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100.0 / self.config.MAP_RESOLUTION - gy1)
            * 480
            / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100.0 / self.config.MAP_RESOLUTION + gx1)
            * 480
            / map_pred.shape[1],
            np.deg2rad(-start_o),
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (
            int(color_palette[11] * 255),
            int(color_palette[10] * 255),
            int(color_palette[9] * 255),
        )
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if self.config.VISUALIZE:
            fn = "{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png".format(
                dump_dir,
                self.rank,
                self.episode_no,
                self.rank,
                self.episode_no,
                self.timestep,
            )
            cv2.imwrite(fn, self.vis_image)
