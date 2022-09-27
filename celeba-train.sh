TRAIN_FLAGS="--batch_size 15 --lr 1e-4 --save_interval 10000 --weight_decay 0.05 --dropout 0.0"
MODEL_FLAGS="--image_size 128 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

echo "Training diffusor"
export NCCL_P2P_DISABLE=1
mpiexec -n 4 python celeba-train-diffusion.py $TRAIN_FLAGS $MODEL_FLAGS \
                                              --output_path '/data/chercheurs/jeanner211/RESULTS/DCF-CelebA/ddpm-128-unconditional' \
                                              --gpus '0,1,2,3'
