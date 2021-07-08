# Mapping 2D Skeleton Sequences from Speed Climbing Videos onto a Virtual Reference Wall

**Contains submodules. Be sure to clone with `--recurse-submodules`, or to run `git submodule update --init` after cloning.**

This repository contains the implementation of the pipeline from my diploma thesis ["Mapping 2D Skeleton Sequences from Speed Climbing Videos onto a Virtual Reference Wall"](https://is.muni.cz/th/zp7vz/?lang=en;setlang=en). Consult the thesis for more information about the used techniques. The thesis archive also contains a version of this repo with the pre-trained models included.

## What does it do

This pipeline takes a speed climbing video and finds a series of transformatins, mapping each frame to a reference wall. This way, skeletons obtained through pose estimation can be transformed from frame coordinates to reference wall coordinates.



https://user-images.githubusercontent.com/4580066/124996181-ce662580-e048-11eb-93dc-755bfddc78aa.mp4



## Structure

- `.vscode` -- configuration files for Visual Studio Code
- `.devcontainer` -- files necessary to build a Docker image for the pipeline
- `actions` -- bash scripts exposing most common actions with the pipeline
- `annotations` -- stores TFRecord annotations (auto-generated)
- `config` -- stores configuration files
- `images` -- stores `train` and `test` datasets, consisting of images with COCO XML annotations
- `models` -- stores models (including checkpoints and exported weights)
- `pretrained_models` -- stores pre-trained model weights (a base for fine-tuning)
- `reference` -- stores reference wall data
- `scripts` -- stores Python code

## How to use

### System requirements

GPU with at least 8 GB of VRAM is necessary for training. For inference, it is possible to use a CPU (and it is sometimes faster than using a GPU, for example for high-frequency RAM + AMD Ryzen CPU). For running using Docker, refer to Docker's system requirements.

### Environment

The environment for this pipeline is specific, with many system dependencies (like very specific TensorFlow or CUDA versions), which is usual for machine learning projects. To simplify this process, it is recommended to use [Docker](https://www.docker.com/) and [Visual Studio Code](https://code.visualstudio.com/) (both multi-platform).

1. [Install Docker](https://www.docker.com/get-started) for your system. 
    - _Optional if you want to train on GPU_: CUDA11-compatible GPU drivers need to be installed. CUDA itself is not necessary. For Windows, follow [this guide](https://stackoverflow.com/questions/49589229/is-gpu-pass-through-possible-with-docker-for-windows/66437683#66437683).
1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
    - _Note_: The extension only works on the Microsoft-branded version of Visual Studio Code. Your Linux distribution might contain an OSS "vscodium" version in the repositories, that does not support the extension out of the box.
1. Open this folder (where the `README.md` is) in Visual Studio Code. Visual Studio Code will automatically detect the container and prompt you to run in container. Alternatively, it is possible to click the `><` button in the bottom left corner and open the container from there.
1. Wait for the image to build. This might take time, as a few GB of data need to be downloaded, and then dependencies installed. This is a one-time process.
1. After the process is finished, Visual Studio Code will open the project folder inside a Docker container configured with correct dependencies. You can use the integrated terminal (View > Terminal) to run commands. Just the local folder will be bind-mounted to the container, thus to read data, you have to put them in here.

It is of course possible to not use Visual Studio Code and use Docker directly, or even to not use Docker and install the system dependencies directly in the OS by hand (for overview, read `.devcontainer/Dockerfile`). However, we do believe that following the tutorial above is the easiest way, regardless of reader's familiarity with Docker and/or Visual Studio Code.

## Commands

Run commands from this folder (where the `README.md` is). Replace `<model_name>` with a name of a subfolder of `models`, for example `20k`.

```bash
# Prepare dataset
# (generates annotations/{train,test}.tfrecord from the images folder)
./actions/prepare_dataset

# Train a specified model
# (models/<model_name>/pipeline.config must be created manually)
./actions/train_model <model_name>

# Run TensorBoard to monitor the training progress
./actions/tensorboard <model_name>

# Evaluate a trained model using COCO metrics
# (Note: training and evaluation can be ran at the same time,
# but your VRAM size has to allow that.)
./actions/evaluate_model <model_name>

# Export a trained model to run inference
# (Note: export is done automatically after training)
./actions/export_model <model_name>

# See all the video processing options
./actions/process_video --help

# Process all videos in `test_data` folder, saving to `out` folder,
# generating transformations, evaluation videos, previews, and logs
./actions/process_video -dvpl -o out test_data/*
```

## Help for `./actions/process_video`

```
usage: process_video.py [-h] -o OUT_DIR [-v] [-d] [-p] [-l]
                        [--detection_threshold DETECTION_THRESHOLD]
                        [--cleaning_eps CLEANING_EPS]
                        [--saved_model_dir SAVED_MODEL_DIR]
                        [--min_track_length MIN_TRACK_LENGTH]
                        [--csaps_smoothing CSAPS_SMOOTHING]
                        [--degrees_of_freedom {2,4,6,8}]
                        input_video [input_video ...]

Map a speed climbing video to a reference wall.

positional arguments:
  input_video           video to detect

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out_dir OUT_DIR
                        output directory
  -v, --save_video      enable saving of evaluation video
  -d, --save_data       enable saving of absolute transformations -- for
                        further use, load by
                        numpy.loadtxt('file.txt.gz').reshape((-1,3,3))
  -p, --save_preview    enable saving of evaluation video preview
  -l, --save_log        enable saving of log.jsonl
  --detection_threshold DETECTION_THRESHOLD
                        detection threshold, from interval <0.0, 1.0)
  --cleaning_eps CLEANING_EPS
                        eps for the cleaning step (removing outliers)
  --saved_model_dir SAVED_MODEL_DIR
                        saved model directory (containing saved_model.pb)
  --min_track_length MIN_TRACK_LENGTH
                        remove tracks shorter than this
  --csaps_smoothing CSAPS_SMOOTHING
                        smoothing coefficient for cubic splines
  --degrees_of_freedom {2,4,6,8}
                        degrees of freedom for the final transformation
```
