#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Tensorboard
pkill tensorboard
rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add baseline package to path
export PYTHONPATH=$DIR/thirdparty/multiagent-particle-envs:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment 1
python3 main.py \
--env-name "comm_cbaa" \
--batch-size 128 \
--ep-max-timesteps 20 \
--total-ep-count 8000 \
--seed 1 \
--n-student 6 \
--n-task 100 \
--maxTask 50 \
--student-noise-type "gauss" \
--gauss-std 0.1 \
--student-train-type "centralized" \
--discount 0.95 \
--prefix ""

