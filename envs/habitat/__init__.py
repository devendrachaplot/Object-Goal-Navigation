# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset

from agents.sem_exp import Sem_Exp_Env_Agent
from .objectgoal_env import ObjectGoal_Env

from .utils.vector_env import VectorEnv


def make_env_fn(args, config_env, rank):
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if args.agent == "sem_exp":
        env = Sem_Exp_Env_Agent(args=args, rank=rank,
                                config_env=config_env,
                                dataset=dataset
                                )
    else:
        env = ObjectGoal_Env(args=args, rank=rank,
                             config_env=config_env,
                             dataset=dataset
                             )

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".glb.json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext) + 4]
            scenes.append(scene)
    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.DATASET.DATA_PATH = \
        basic_config.DATASET.DATA_PATH.replace("v1", args.version)
    basic_config.DATASET.EPISODES_DIR = \
        basic_config.DATASET.EPISODES_DIR.replace("v1", args.version)
    basic_config.freeze()

    scenes = basic_config.DATASET.CONTENT_SCENES
    if "*" in basic_config.DATASET.CONTENT_SCENES:
        content_dir = os.path.join(basic_config.DATASET.EPISODES_DIR.format(
            split=args.split), "content")
        scenes = _get_scenes_from_folder(content_dir)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=["envs/habitat/configs/"
                                           + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]
            print("Thread {}: {}".format(i, config_env.DATASET.CONTENT_SCENES))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")
        # agent_sensors.append("SEMANTIC_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        # Reseting episodes manually, setting high max episode length in sim
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 10000000
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = args.min_depth
        config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.max_depth
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        # config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
        # config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
        # config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
        # config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
        #     [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = args.turn_angle
        config_env.DATASET.SPLIT = args.split
        config_env.DATASET.DATA_PATH = \
            config_env.DATASET.DATA_PATH.replace("v1", args.version)
        config_env.DATASET.EPISODES_DIR = \
            config_env.DATASET.EPISODES_DIR.replace("v1", args.version)

        config_env.freeze()
        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )

    return envs
