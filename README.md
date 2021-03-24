
## Installation

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Dataset Preparation


**Preparing ADE20K Dataset**. The dataset can be downloaded [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), which is from [MIT Scene Parsing BenchMark](http://sceneparsing.csail.mit.edu/). After unzipping the datgaset, put the jpg image files `ADEChallengeData2016/images/` and png label files `ADEChallengeData2016/annotatoins/` in the same directory. 


## Generating Images Using Pretrained Model


1. Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ/view?usp=sharing), save it in 'checkpoints/'.

ADE20K pretrained models:

0.5_net_G_seg_outpainting.pth: 50% ratio of mask Semantic Segmentation Outpainting model.

0.5_net_G_image_outpainting.pth: 50% ratio of mask  Semantic Image Synthesis model.


0.25_net_G_seg_outpainting.pth: 25% ratio of mask Semantic Segmentation Outpainting model.

0.25_net_G_image_outpainting.pth: 25% ratio of mask  Semantic Image Synthesis model.


2. Outpainting using the pretrained model.

Firstly, please use pre-trained ResNeSt-200~\cite{zhang2020resnest} to generate segmentation map of train data and validation data and put them in \'[path_to /ADEChallengeData2016/]/predict_full/\'.


Semantic Segmentation Outpainting stage:


    ```bash
    python predict_ade20k.py --name=[experiment\'s name] --dataroot=[path_to /ADEChallengeData2016/] --G_checkpoint_name=[name_of_pretrained_model] --use_gpu --batchSize=1 --save_image --ratio=[ratio_of_outpainting_task: 0.5_or_0.25]
    ```
It will generate extended semantic segmentation map for next stage in ../results/[experiment\'s name]/. And change line 47 of data/ade20k_dataset.py to "label_files = [\'../results/[experiment\'s name]/train/ADE_val_%08d.png\' % (j+1) for j in range(2000)]".

Semantic Image Synthesis stage:

    ```bash
    python predict_ade20k_segmentation.py --name=[experiment\'s name] --dataroot=[path_to /ADEChallengeData2016/] --G_checkpoint_name=[name_of_pretrained_model] --use_gpu --batchSize=1 --save_image --ratio=[ratio_of_outpainting_task: 0.5_or_0.25]
    ```
It will generate extended image guided by semantic segmentation map in \'../results/[experiment\'s name]/train/\'



3. The output images are stored at `../results/[experiment\'s name]/gen_new/` by default.


## Code Structure

-  `predict_ade20k.py` : code for Semantic Image Synthesis stage.
-  `predict_ade20k_segmentation.py` : code for 
Semantic Segmentation Outpainting stage.
- `trainers/pix2pix_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `data/`: defines the class for loading images and label maps.



```

## Acknowledgments
This code borrows heavily from SPADE.
