import os
os.system('python check_train_hamilton.py --ckpt ../../ckpts/spring_chain/Euler \
                            --solver euler\
                            --dt 0.2')
os.system('python check_train_hamilton.py --ckpt ../../ckpts/spring_chain/Runge_Kutta4 \
                            --solver rk4\
                            --dt 0.2')