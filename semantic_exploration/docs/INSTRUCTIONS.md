# Instructions

## Training
For training the SemExp model on the Object Goal Navigation task:
```
python main.py
```

### Specifying number of threads
The code runs multiple parallel threads for training. Each thread loads a scene on a GPU. The code automatically decides the total number of threads and number of threads on each GPU based on the available GPUs.

If you would like to not use the auto gpu config, you need to specify the following:
```
--auto_gpu_config 0
-n, --num_processes NUM_PROCESSES
--num_processes_per_gpu NUM_PROCESSES_PER_GPU
--num_processes_on_first_gpu NUM_PROCESSES_ON_FIRST_GPU 
```
`NUM_PROCESSES_PER_GPU` will depend on your GPU memory, 6 works well for 16GB GPUs.
`NUM_PROCESSES_ON_FIRST_GPU` specifies the number of processes on the first GPU in addition to the SemExp model, 1 works well for 16GB GPUs.
`NUM_PROCESSES` depends on the number of GPUs used for training and `NUM_PROCESSES_PER_GPU` such that 
```
NUM_PROCESSES <= min(NUM_PROCESSES_PER_GPU * number of GPUs + NUM_PROCESSES_ON_FIRST_GPU, 25)
```
The Gibson training set consists of 25 scenes.

For example, for training the model on 5 GPUs with 16GB memory per GPU:
```
python main.py --auto_gpu_config 0 -n 25 --num_processes_per_gpu 6 --num_processes_on_first_gpu 1 --sim_gpu_id 1 
```
Here, `sim_gpu_id = 1` specifies simulator threads to run from GPUs 1 onwards.
Each GPU from 1 to 4 will run 6 threads each, and GPU 0 will run 1 thread and
the SemExp model.

### Specifying log location, periodic model dumps
```
python main.py -d saved/ --exp_name exp1 --save_periodic 500000
```
The above will save the best model files and training log at `saved/models/exp1/` and save all models periodically every 500000 steps at `saved/dump/exp1/`. Each module will be saved in a separate file. 

### Hyper-parameters
Most of the default hyper-parameters should work fine. Some hyperparameters are set for training with 25 threads, which might need to be tuned when using fewer threads. Fewer threads lead to a smaller batch size so the learning rate might need to be tuned using `--lr`.

## Downloading pre-trained models
```
mkdir pretrained_models;
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=171ZA7XNu5vi3XLpuKs8DuGGZrYyuSjL0' -O pretrained_models/sem_exp.pth
```

## Evaluation

The following are instructions to evaluate the model on the Gibson val set.

For evaluating the pre-trained model:
```
python main.py --split val --eval 1 --load pretrained_models/sem_exp.pth
```

The pre-trained model should get 0.657 Success, 0.339 SPL and 1.474 DTG.

### Manual GPU config

If you would like to not use the auto GPU config, specify the number of threads for evaluation using `--num_processes` and the number of evaluation episodes per thread using `--num_eval_episodes`.
The Gibson val set consists of 5 scenes and 200 episodes per scene. Thus, we need to use 5 threads for evaluation, and 200 episodes per thread. Split 5 scenes on GPUs based on your GPU memory sizes. The code requires `0.8 + 0.4 * num_scenes (GB)` GPU memory on the first GPU for the model and around 2.6GB memory per scene.

For example, if you have 1 GPU with 16GB memory:
```
python main.py --split val --eval 1 --auto_gpu_config 0 \
-n 5 --num_eval_episodes 200 --num_processes_on_first_gpu 5 \
--load pretrained_models/sem_exp.pth
```
or if you have 2 GPUs with 12GB memory each:
```
python main.py --split val --eval 1 --auto_gpu_config 0 \
-n 5 --num_eval_episodes 200 --num_processes_on_first_gpu 1 \
--num_processes_per_gpu 4 --sim_gpu_id 1 \
--load pretrained_models/sem_exp.pth
```

### Visualization and printing images
For visualizing the agent observations and predicted map and pose, add `-v 1` as an argument to the above command. This will require a display to be attached to the system.

To visualize on headless systems (without display), use `--print_images 1 -d results/ --exp_name exp1`. This will save the visualization images in `results/dump/exp1/episodes/`.

Both `-v 1` and `--print_images 1` can be used together to visualize and print images at the same time. 


## Notes

- Training the model for 10 million frames with 25 threads takes around 2.5 days on an Nvidia DGX-1 system using 5 16GB GPUs, but the model provides good performance even with only 1 million frames (~6 hrs) of training.

- Evaluating the model on the val set for 1000 episodes with 5 threads takes around 2.5 hrs on an Nvidia DGX-1 system.

- The code does not contain the Denoising Network described in our [paper](https://arxiv.org/pdf/2007.00643.pdf).
This is because of the following reasons:
  - Training the Denoising Network requires downloading the original Gibson dataset (non-Habitat format), 3DSceneGraph dataset, and building Habitat format semantic scenes using both the datasets.
  - Training the Denoising Network requires building and cleaning top-down maps which makes training much slower.
  - The first-person semantic annotations for Gibson are not perfectly accurate, they do not align with the depth sensor. This results in Denoising Network only providing a marginal performance improvement.


## Tips
To silence the habitat sim log add the following to your `~/.bashrc` (Linux) or `~/.bash_profile` (Mac) 
```
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
```