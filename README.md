# Code for the Paper: *Sparser, Better, Deeper, Stronger: Improving Sparse Training with Exact Orthogonal Initialization*

## Project Structure

`data_handling` - code for downloading & loading the data   
`e2e_tests` - contains some tests of initialization schemes (EOI, AI)  
`models` - the models  
`orthogonal` - contains the logic for generating random, sparse, orthogonal matrices using Givens rotations  
`sao` - code of the SAO (Sparsity-Aware Orthogonal Initialization).    
`sparselearning` - the logic behind sparse training (main mask logic in `core.py`)  
`specs` - contains all the experiment settings which we refer to in our paper.  

`main.py` - contains the main logic for running experiments  
`trainer.py` - contains the logic for training   

## Running Experiments

Prepare a `conda` environment for runing the experiments. You need to create the environment using the `environment.yml` which contains the necessary dependencies in appropriate versions. This may take a few minutes to complete.

```
conda env create -f environment.yml
```

After the environment is created, activate it by running:

```
conda activate ortho-sparse
```

Inside the environment, you need to additionally install `wandb`.

```
pip install wandb
```

Also please set the environment variables needed for wandb usage.

```
export WANDB_ENTITY_NAME=<your_entity_name>
export WANDB_PROJECT_NAME=<your_project_name>
export WANDB_API_KEY=<your_API_key>
```

This is a necessary step, even if you won't be using wandb, although if you're not planning to use wandb, you can set these environment variables to some dummy values. Usage of `wandb` is indicated by the flag `--use_wandb` in our project.

Also, you can login to `wandb` with:

```
wandb login
```

The next step is preparing the datasets. Depending on which experiments you wish to run, you need to define an appropriate environment variable which stores the path to the dataset.

```
export MNIST_PYTORCH=<path_to_mnist>
export CIFAR10_PYTORCH=<path_to_cifar10>
export TINY_IMAGENET=<path_to_tiny_imagenet>
```

In the case of MNIST and CIFAR10, the datasets should be downloaded automatically when you run an experiment so only exporting the paths should be sufficient, but in the case of TINY_IMAGENET, you need to download the dataset yourself and place in the directory pointed by the TINY_IMAGENET variable.

### Running a single experiment

You can do it by running `python main.py ...` with an appropriate collection of flags which describe the experiment you want to run. You can investigate the `main.py` file to find the flags and their descriptions. There are also some flags defined in `sparselearning/core.py`. An example is given below:

```
python main.py --use_wandb false --model cifar_resnet_32 --data cifar10 --batch_size 64 --epochs 100 --sparse true --sparse_init snip_direct_EI --density 0.05 --seed 13 --lr 0.1 --activation relu --more_nonzeros true
```

### Running a batch of experiments

You can run experiments described by the spec files located in the `specs` folder. These `specs` folder contains a complete collection of the experiments which we describe in the paper. For this you will need to install `mrunner` in your environment.

```
pip install git+https://gitlab.com/awarelab/mrunner.git
```

Now you can simply select a spec from the `specs` folder and run:

```
python mrun.py --ex <path_to_spec>
```

It is also possible to create your own specs. In the specs, the flag `use_wandb` in `base_config` is set to `False`. To use `wandb` when running a spec simply change this value to `True`.

## Acknowledgements

The project contains some external repositories:

The contents of folder `sao` come from the [public SAO repository](https://github.com/kiaraesguerra/SAO). We do not claim ownership of this code.  

The contents of folder `sparselearning` are based on the [public sparse_learning](https://github.com/TimDettmers/sparse_learning) repository (MIT License), 
and the [public Random Pruning](https://github.com/VITA-Group/Random_Pruning) repository


## Notes

For clarity, we note that in the code, our sparse initialization scheme "Exact Orthogonal Initialization" (EOI) is referred to by a shorter acronym: **EI**.
