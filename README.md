<p align="center">

  <h1 align="center">CookAR: Affordance Augmentations in Wearable AR to Support Kitchen Tool Interactions for People with Low Vision</h1>
  <p align="center">
    Jaewook Lee<sup>1</sup>, 
    Andrew D. Tjahjadi<sup>1</sup>,
    Jiho Kim<sup>1</sup>,
    Junpu Yu<sup>1</sup>,
    Minji Park<sup>2</sup>,
    Jiawen Zhang<sup>1</sup>, <br>
    Yang Li<sup>1</sup>,
    Sieun Kim<sup>3</sup>,
    XunMei Liu<sup>1</sup>,
    Jon E. Froehlich<sup>1</sup>,
    Yapeng Tian<sup>4</sup>,
    Yuhang Zhao<sup>5</sup>
    <br><br>
    <sup>1</sup>University of Washington, 
    <sup>2</sup>Sungkyunkwan University, 
    <sup>3</sup>Seoul National University,<br>
    <sup>4</sup>University of Texas at Dallas,
    <sup>5</sup>University of Wisconsin-Madison
    <br>
  </p>
  <h2 align="center">UIST 2024</h2>
  <h3 align="center"><a href="https://github.com/makeabilitylab/CookAR">Code</a> | <a href="https://arxiv.org/abs/2407.13515">Paper </a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/makeabilitylab/CookAR/blob/main/assets/banner.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>CookAR</strong> is a Computer Vision-powered prototype AR system with real-time object affordance augmentations to support safe and efficient interactions with kitchen tools for people with impaired vision abilities (<a href="https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/low-vision">Low Vision</a>). In this repo, we present the exact fine-tuned instance segmentation model for affordance augmentations, along with the first egocentric dataset of kitchen tool affordances collected and annotated by the research team.
</p>
<br>


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
Along with CookAR, we present the very first kitchen tool affordance image dataset, which contains **10,152** images (**8,346** for training, **1193** for validation, and **596** for testing) with **18** categories of objects listed below. Raw images were extracted from [EPIC-KITCHENS video dataset](https://epic-kitchens.github.io/2024).

<p align="center">
  <a href="">
    <img src="https://github.com/makeabilitylab/CookAR/blob/main/assets/ds_fig.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">

### Categories
|                |                |
| -------------- | -------------- |
| Carafe Base    | Carafe Handle  |
| Cup Base       | Cup Handle     |
| Fork Tines     | Fork Handle    |
| Knife Blade    | Knife Handle   |
| Ladle Bowl     | Ladle Handle   |
| Pan Base       | Pan Handle     |
| Scissor Blade  | Scissor Handle |
| Spatula Head   | Spatula Handle |
| Spoon Bowl     | Spoon Handle   |

## Model fine-tuning & Run on image and video
In this section we provide a brief guideline about how to fine-tune the CookAR models on your customzied datasets and how to run on imgae or video of your choice. Specifically, we break this section into four parts:
1. Download the checkpoints
2. Download and check the dataset
3. Download and edit configuration file
4. Start training
5. Run on image or video

CookAR is initially fine-tuned on RTMDet-Ins-L with frozen backbone stages, which can be found at the [official repo](https://github.com/open-mmlab/mmdetection). You can find a more detailed tutorial on fine-tuning RTMDet related models at [here](https://github.com/makeabilitylab/mmdet-fine-tuning).

### Step 1: Download the checkpoints
- Vanilla CookAR: Use this [link](https://drive.google.com/file/d/1gQAB4rDclr2bw2FzfY7mooA6GLqj7F1D/view?usp=drive_link) to download our fine-tuned weights.

You can directly use it for your tasks ( jump to step 3 ) or build upon it with your own data.

### Step 2: Download and check the dataset
- CookAR Dataset: Use this [link](https://drive.google.com/file/d/1kHNFvjYimnKNTcffDigZ8EpPie4u1DTa/view?usp=drive_link) to download our self-built dataset in **COCO-MMDetection** format.

If you are fine-tuning with your own dataset, make sure it is also in COCO-MMDetection format and it is recommanded to run `coco_classcheck.py` in fine-tuning folder to check the classes contained.

### Step 3: Download and edit configuration file
In this repo, we also provide the config file used in our fine-tuning process, which can be found in configs folder. To use the model on your tasks directly, no modification is required and jump to step 5.

Before start your own training, check and run `config_setup.py` in fine-tuning folder to edit the config file. Make sure that the number of classes is correctly modified in reflect of the dataset provided and all classes are listed in the same order shown by `coco_classcheck.py`.

### Step 4: Start training
Simply run `python tools/train.py PATH/TO/CONFIG`.

### Step 5: Run on image or video
Use the provided scripts `infer_img.py` and `infer_video.py` to run inferences on a single image or video.

## Citation
```bibtex
@inproceedings{lee2024cookar,
  author = {Lee, Jaewook and Tjahjadi, Andrew D. and Kim, Jiho and Yu, Junpu and Park, Minji and Zhang, Jiawen and Li, Yang and Kim, Sieun and Liu, XunMei and Froehlich, Jon E. and Tian, Yapeng and Zhao, Yuhang},
  title = {CookAR: Affordance Augmentations in Wearable AR to Support Kitchen Tool Interactions for People with Low Vision},
  year = {2024},
  url = {},
  doi = {},
  booktitle = {Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology},
  articleno = {},
  numpages = {},
  series = {UIST '24}
}
```
