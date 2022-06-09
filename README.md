## Kaggle Competition
### [I Was Busy Thinkin' 'Bout Birds](https://www.kaggle.com/competitions/birds22sp/leaderboard)
#

# Installation/Setup (for Windows)

[Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) - command-line to install and run the following programs

[JupyterLab](https://jupyter.org/install) - for `.ipynb` notebooks + training on local machine

[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network) - to use Nvidia GPU for training 

[PyTorch](https://pytorch.org/get-started/locally/) - can check CUDA version using `nvcc -V`

[Google Colab](https://colab.research.google.com/) - decent cloud GPU usage for free

[Downloading ILSVRC 2012](https://reimbar.org/dev/imagenet/) 

- Move validation folders to labeled subfolders with this [shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- (for reference, I ended up not training with ImageNet for the competition)

# Training Model
The dominant model used was pretrained ResNeXt-50 (`resnext50_32x4d`). 

Pretrained model ResNeXt-101 (`resnext101_32x8d`) was also tested with different tradeoffs between image and batch sizes (larger images & smaller batches, vice versa). It failed miserably (18.4% best accuracy) despite a relatively similar loss convergence. Better hyperparameters as seen with training ResNeXt-50 couldn't have been used due to GPU memory constraints.

Initially, the plan was to train the models using ImageNet (ILSVRC 2012 subset), but it was taking way too long to even complete one epoch so the plan was to continue finding the best model and hyperparameters (with memory constraints in mind).

# ResNeXt-50 Model Loss Functions & Submission Results


# Issues
- The bottleneck in improving the model (e.g. larger images, bigger batch sizes) seemed to be GPU memory. This was resolved by using Colab (but training runtimes took much longer).

- After encountering `RuntimeError: CUDA error: out of memory`, restarting runtime helps if modifying hyperparameters, notably batch size, continues to error out (excluding actual memory bottlenecks).

- Often loading checkpoints for training would result in out of memory despite no issues occuring when training without a checkpoint. This was resolved with using parameter `map_location='cpu'` in `torch.load()`.