import os
import torch
import numpy as np

from envs import make_vec_envs
from arguments import get_args

os.environ["OMP_NUM_THREADS"] = "1"

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    num_episodes = int(args.num_eval_episodes)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    for ep_num in range(num_episodes):
        for step in range(args.max_episode_length):
            action = torch.randint(0, 3, (args.num_processes,))
            obs, rew, done, infos = envs.step(action)

            if done:
                break

    print("Test successfully completed")


if __name__ == "__main__":
    main()
