# DiME's official code

This is the codebase for the ACCV 2022 paper [Diffusion Models for Counterfactual Explanations](https://arxiv.org/abs/2203.15636).

## Environment

Through anaconda, install our environment:

```bash
conda env create -f env.yaml
conda activate dime
``` 

## Data preparation

Please download and uncompress the CelebA dataset [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). There is no need for any post-processing. The final folder structure should be:

```
PATH ---- img_align_celeba ---- xxxxxx.jpg
      |
      --- list_attr_celeba.csv
      |
      --- list_eval_partition.csv
```

## Downloading pre-trained models

To use our trained models, you must download them first. Please extract them to the folder `models`. Our code provides the CelebA diffusion model, the classifier under observation, and the trained oracle. Download the VGGFace2 model throught this [github repo](https://github.com/cydonia999/VGGFace2-pytorch). Download the `resnet50_ft` model.

Download Link:

- [Classifier](https://drive.google.com/file/d/1OqjWns4NSu6AiKkOnpUOjUHzA8sQlaOA/view?usp=sharing)
- [Diffusion Model](https://drive.google.com/file/d/17iB1aL4xctDukov-OIDuKqZdQ9YB1ZQz/view?usp=sharing)
- [Oracle](https://drive.google.com/file/d/1Ua9gK1BiUTG4wIkhpBpWyn6B-OCQKKMx/view?usp=sharing)


## Extracting Counterfactual Explanations

To create the counterfactual explanations, please use the main.py script as follows:

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"
DATAPATH=/path/to/dataset
MODELPATH=/path/to/model.pt
CLASSIFIERPATH=/path/to/classifier.pt
ORACLEPATH=/path/to/oracle.pt
OUTPUT_PATH=/path/to/output
EXPNAME=exp/name

# parameters of the sampling
GPU=0
S=60
SEED=4
USE_LOGITS=True
CLASS_SCALES='8,10,15'
LAYER=18
PERC=30
L1=0.05
QUERYLABEL=31
TARGETLABEL=-1
IMAGESIZE=128  # dataset shape

python -W ignore main.py $MODEL_FLAGS $SAMPLE_FLAGS \
  --query_label $QUERYLABEL --target_label $TARGETLABEL \
  --output_path $OUTPUT_PATH --num_batches $NUMBATCHES \
  --start_step $S --dataset 'CelebAMV' \
  --exp_name $EXPNAME --gpu $GPU \
  --model_path $MODELPATH --classifier_scales $CLASS_SCALES \
  --classifier_path $CLASSIFIERPATH --seed $SEED \
  --oracle_path $ORACLEPATH \
  --l1_loss $L1 --use_logits $USE_LOGITS \
  --l_perc $PERC --l_perc_layer $LAYER \
  --save_x_t True --save_z_t True \
  --use_sampling_on_x_t True \
  --save_images True --image_size $IMAGESIZE
```

Given that the sampling process may take much time, we've included a way to split the sampling into multiple processes. To use this feature, include the flag `--num_chunks C`, where `C` is the number of chunks to split the dataset. Then, run `C` times the code using the flag `--chunk c`, where `c` is the chunk to generate the evaluation (hence, `c \in {0, 1, ..., C - 1}`).

The results will be stored `OUTPUT_PATH`. This folder has the following structure:

```
OUTPUT_PATH ----- Original ---- Correct
              |             |
              |             --- Incorrect
              |
              |
              |
              --- Results ---- EXPNAME ---- (I/C)C ---- (I/C)CF ---- CF
                                                                 |
                                                                 --- Info
                                                                 |
                                                                 --- Noise
                                                                 |
                                                                 --- SM
```

We found this structure useful to experiment since we can change only the `EXPNAME` to refer to another experiment without changing the original images. The folder `Original` contains the correctly classified (misclassified) images in `Correct` (`Incorrect`). We resume the structure of the counterfactuals explanations (`Results/EXPNAME`) as: `(I/C)C`: (In/correct) classification. `(I/C)CF`: (In/correct) counterfactual. `CF`: counterfactual images. `Info`: Useful information per instance. `Noise`: Noisy instance at timestep $\tau$ of the input data. `SM`: Difference between the input and its counterfactual. All files in all folders will have the same identifier.


## Evaluation

We provide our evaluation protocol scripts to assess the performance of our method. All our evaluation codes use the folder structure presented before. Please look at the --help function flag for more information about their inputs.
- FVA: `compute_FVA.py`.
- MNAC: `compute_MNAC.py`.
- $\sigma_L$: `compute_LPIPS.py`. Computes the variability metric.
- CD: `compute_CD.py`. Computes our proposed metric, Correlation Difference.
- FID: `compute-fid.sh`. The first input is the OUTPUT_PATH and the second one the EXPNAME.


## Training the DDPM model from scratch

We provided a bash script to train the DDPM to generate the counterfactual explanations: `celeba-train.sh`. Nevertheless, the syntax to run the code base is:

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 15 --lr 1e-4 --save_interval 30000 --weight_decay 0.05 --dropout 0.0"
mpiexec -n N python celeba-train-diffusion.py $TRAIN_FLAGS \
                                              $MODEL_FLAGS \
                                              --output_path OUTPUT_FOLDER \
                                              --gpus GPUS
```

## Citation

If you found useful our code, please cite our work.

```
@inproceedings{Jeanneret_2022_ACCV,
    author    = {Jeanneret, Guillaume and Simon, Lo\"ic and Fr\'ed\'eric Jurie},
    title     = {Diffusion Models for Counterfactual Explanations},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022}
}
``` 

## Code Base

We based our repository on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
