# Towards Scalable Unpaired Virtual Try-On via Patch-Routed Spatially-Adaptive GAN
Official code for NeurIPS 2021 paper "Towards Scalable Unpaired Virtual Try-On via Patch-Routed Spatially-Adaptive GAN"

## Requirements

Create a virtual environment:
```
virtualenv pasta --python=3.7
source pasta/bin/activate
```
Install required packages:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
pip install psutil scipy matplotlib opencv-python scikit-image pycocotools
```

## Data Preparation
Since the copyright of the UPT dataset belongs to the E-commerce website [Zalando](https://www.zalando.co.uk/) and [Zalora](https://www.zalora.com.my/), we only release the image links in this [link](https://drive.google.com/file/d/1GpiwvE318_EOmbLrRR8gADmY2cimvkFF/view?usp=sharing). For more details about the dataset and the crawling scripts, please send email to [xiezhy6@mail2.sysu.edu.cn]().

After downloading the raw RGB image, we run the pose estimator [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and human parser [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) for each image to obtain the 18-points human keypoints and the 19-labels huamn parsing, respectively.

The dataset structure is recommended as:
```
+—UPT_256_192
|   +—UPT_subset1_256_192
|       +-image
|           +- e.g. image1.jpg
|           +- ...
|       +-keypoints
|           +- e.g. image1_keypoints.json
|           +- ...
|       +-parsing
|           +- e.g. image1.png
|           +- ...
|       +-train_pairs_front_list_0508.txt
|       +-test_pairs_front_list_shuffle_0508.txt
|   +—UPT_subset2_256_192
|       +-image
|           +- ...
|       +-keypoints
|           +- ...
|       +-parsing
|           +- ...
|       +-train_pairs_front_list_0508.txt
|       +-test_pairs_front_list_shuffle_0508.txt
|   +— ...
```

By using the raw RGB image, huamn keypoints, and human parsing, we can run the training script and the testing script.

## Running Inference
We provide the [pre-trained model](https://drive.google.com/drive/folders/1CNj5VJawwEjbAnpCRO0XlWdOduF1xDjm?usp=sharing) of PASTA-GAN which is trained by using the full UPT dataset (i.e., our newly collected data, data from Deepfashion dataset, data from MPV dataset).

we provide a simple script to test the pre-trained model provided above on the UPT dataset as follow:
```
CUDA_VISIBLE_DEVICES=0 python3 -W ignore test.py \
    --network /datazy/Codes/PASTA-GAN/PASTA-GAN_fullbody_model/network-snapshot-004000.pkl \
    --outdir /datazy/Datasets/pasta-gan_results/unpaired_results_fulltryonds \
    --dataroot /datazy/Datasets/PASTA_UPT_256 \
    --batchsize 16
```
or you can run the bash script by using the following commend:
```
bash test.sh 1
```
Note that, in the testing script, the parameter `--network` refers to the path of the pre-trained model, the parameter `--outdir` refers to the path of the directory for generated results, the parameter `--dataroot` refers to the path of the data root. Before running the testing script, please make sure these parameters refer to the correct locations.

## Running Inference
### Training the 256x192 PASTA-GAN full body model on the UPT dataset
1. Download the UPT_256_192 training set.
2. Download the VGG model from [VGG_model](https://drive.google.com/file/d/1G3L6rzwSXRALoSq4heqKsboUZk4lth1g/view?usp=sharing), then put "vgg19_conv.pth" and "vgg19-dcbb9e9d" under the directory "checkpoints".
3. Run `bash train.sh 1`.

## Todo
- [x] Release the the pretrained model (256x192) and the inference script.
- [x] Release the training script.
- [ ] Release the pretrained model (512x320).

## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.

