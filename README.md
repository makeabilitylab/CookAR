# CookAR
# Set up
conda blablabala, mmdet blablabla
# Dataset

# Model fine-tuning & Run on image and video
In this section we provide a brief guideline about how to fine-tune the CookAR models on your customzied datasets and how to run on imgae or video of your choice. Specifically, we break this section into four parts:
1. Download the checkpoints
2. Download and check the dataset
3. Download and edit configuration file
4. Start training
5. Run on image or video

CookAR is initially fine-tuned on RTMDet-Ins-L with frozen backbone stages, which can be found at the [official repo](https://github.com/open-mmlab/mmdetection). You can find a more detailed tutorial on fine-tuning RTMDet related models at [here](https://github.com/makeabilitylab/mmdet-fine-tuning).

## Step1: Download the checkpoints
- Vanilla CookAR: Use this [link](https://google.com) (**put the trained weights here in a gdrive link as they are generally too big to be held at github, delete when done**) to download our fine-tuned weights.

You can directly use it for your tasks or build upon it with your own data.
## Step2: Download and check the dataset
- CookAR Dataset: Use this [link](https://google.com) (**put dataset zip here in a gdrive or roboflow link if they are generally too big to be held at github, delete when done**) to download our self-built dataset in **COCO-MMDetection** format.

If you are fine-tuning with your own dataset, make sure it is also in COCO-MMDetection format and it is recommanded to run `coco_classcheck.py` in fine-tuning folder to check the classes contained.
## Step3: Edit configuration file
In this repo, we provide the config file used in our fine-tuning process. Before start your own training, check and run `config_setup.py` in fine-tuning folder to edit the config file. 

Make sure that the number of classes is correctly modified in reflect of the dataset provided and all classes are listed in the same order shown by `coco_classcheck.py`.

## Step4: Start training
Run `python tools/train.py PATH/TO/CONFIG`.

## Step5: Run on image or video
Use the provided scripts `infer_img.py` and `infer_video.py` to run inferences on a single image or video.
