# Docker and Singularity Instructions:
We provide experimental [docker](https://www.docker.com/) and [singularity](https://sylabs.io/) images with all the dependencies installed.

Before setting up the docker, pull the code using:
```
git clone https://github.com/devendrachaplot/Object-Goal-Navigation/
cd Object-Goal-Navigation/;
```
Download and set up the scene and episode datasets as described [here](README.md#setup).

For docker, either build docker image using the provided [Dockerfile](./Dockerfile):
```
docker build -t devendrachaplot/habitat:sem_exp .
```
Or pull docker image from dockerhub:
```
docker pull devendrachaplot/habitat:sem_exp
```

After building or pulling the docker image, run the docker using:
```
docker run -v $(pwd)/:/code -v $(pwd)/data:/code/data --runtime=nvidia -it devendrachaplot/habitat:sem_exp
```

Inside the docker, check the habitat compatibility with your system:
```
cd /habitat-api/
python examples/benchmark.py
```

To run the SemExp model inside the docker, `cd /code/` and run the same commands as described in [INSTRUCTIONS](./INSTRUCTIONS.md).

For pulling the singularity image:
```
singularity pull docker://devendrachaplot/habitat:sem_exp
```