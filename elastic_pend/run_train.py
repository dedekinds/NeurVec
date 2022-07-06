import os
os.system('python train_poly_segments.py --train_dir ../data/elastic_pend/trainset.npy \
                            --test_dir ../data/elastic_pend/validationset.npy \
                            --gpu_id 0 \
                            --ckpt neurvec_pend \
                            --epoch 500 \
                            --model_name orgNN \
                            --lr 0.001 \
                            --T_train 2 \
                            --optim adam \
                            --train_coarse 1 \
                            --batch_size 16384 \
                            --dt 0.1')
