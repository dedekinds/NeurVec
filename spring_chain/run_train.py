import os

#euler
os.system('python train_poly_segments.py \
    --train_dir ../data/spring_chain/euler_trainset.npy\
    --test_dir ../data/spring_chain/validationset.npy\
    --gpu_id 0\
    --ckpt neurvec_spring_euler\
    --epoch 500\
    --model_name orgNN\
    --lr 0.001\
    --T_train 2\
    --optim adam\
    --train_coarse 1 \
    --batch_size 8096\
    --solver euler\
    --dt 0.2')

#rk4
os.system('python train_poly_segments.py \
    --train_dir ../data/spring_chain/rk4_trainset.npy\
    --test_dir ../data/spring_chain/validationset.npy\
    --gpu_id 0\
    --ckpt neurvec_spring_rk4\
    --epoch 500\
    --model_name orgNN\
    --lr 0.001\
    --T_train 2\
    --optim adam\
    --train_coarse 1 \
    --batch_size 16384\
    --solver rk4\
    --dt 0.2')
