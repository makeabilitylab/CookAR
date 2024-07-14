# CookAR
~~Banner figs  and introduction here~~
## Setup
To use CookAR, we recommand using Conda. CookAR also depends on [MMDetection toolbox](https://mmdetection.readthedocs.io/en/latest/) and [PyTorch](https://pytorch.org/get-started/locally/). If your GPU supports [CUDA](https://developer.nvidia.com/cuda-toolkit), please install it first.

```
conda create --name=CookAR python=3.8 
conda activate CookAR
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  //change according to your cuda version
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
It is recommended that you first install PyTorch and then MMDetection otherwise it might not be correctly complied with CUDA.

- Once you installed everything, firstly make three folders inside the mmdetection directory namely `./data`, `./checkpoints` and `./work_dir` either manually or using `mkdir` in conda.
- Download pre-trained config and weights files from mmdetection by running `mim download mmdet --config rtmdet-ins_l_8xb32-300e_coco --dest ./checkpoints` and run `python test_install.py` to check see if things are working correctly.  You should see an image with segmentation masks pops out.

## Dataset
~~brief descrption and stats about the dataset~~

## Model fine-tuning & Run on image and video
In this section we provide a brief guideline about how to fine-tune the CookAR models on your customzied datasets and how to run on imgae or video of your choice. Specifically, we break this section into four parts:
1. Download the checkpoints
2. Download and check the dataset
3. Download and edit configuration file
4. Start training
5. Run on image or video

CookAR is initially fine-tuned on RTMDet-Ins-L with frozen backbone stages, which can be found at the [official repo](https://github.com/open-mmlab/mmdetection). You can find a more detailed tutorial on fine-tuning RTMDet related models at [here](https://github.com/makeabilitylab/mmdet-fine-tuning).

### Step 1: Download the checkpoints
- Vanilla CookAR: Use this [link](https://google.com) ~~(**put the trained weights here in a gdrive link as they are generally too big to be held at github, delete when done**)~~ to download our fine-tuned weights.

You can directly use it for your tasks ( jump to step 5 ) or build upon it with your own data.
### Step 2: Download and check the dataset
- CookAR Dataset: Use this [link](https://google.com) ~~(**put dataset zip here in a gdrive or roboflow link if they are too big to be held at github, delete when done**)~~ to download our self-built dataset in **COCO-MMDetection** format.

If you are fine-tuning with your own dataset, make sure it is also in COCO-MMDetection format and it is recommanded to run `coco_classcheck.py` in fine-tuning folder to check the classes contained.
### Step 3: Edit configuration file
In this repo, we provide the config file used in our fine-tuning process. Before start your own training, check and run `config_setup.py` in fine-tuning folder to edit the config file. 

Make sure that the number of classes is correctly modified in reflect of the dataset provided and all classes are listed in the same order shown by `coco_classcheck.py`.

### Step 4: Start training
Run `python tools/train.py PATH/TO/CONFIG`.

### Step 5: Run on image or video
Use the provided scripts `infer_img.py` and `infer_video.py` to run inferences on a single image or video.

## Other sections if necessary
