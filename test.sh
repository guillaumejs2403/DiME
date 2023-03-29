MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 50 --timestep_respacing 200"
DATAPATH=/path/to/celeba
OUTPUT_PATH=/path/to/results
EXPNAME=expname

# parameters of the sampling
GPU=2
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
  --output_path $OUTPUT_PATH \
  --start_step $S --dataset 'CelebA' \
  --exp_name $EXPNAME --gpu $GPU \
  --classifier_scales $CLASS_SCALES \
  --seed $SEED --data_dir $DATAPATH\
  --l1_loss $L1 --use_logits $USE_LOGITS \
  --l_perc $PERC --l_perc_layer $LAYER \
  --save_x_t True --save_z_t True \
  --use_sampling_on_x_t True --num_batches 1 \
  --save_images True --image_size $IMAGESIZE
