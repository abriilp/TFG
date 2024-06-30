#!/bin/bash

#SBATCH -n 4 # Number of CPU cores. Maximum 10
#SBATCH -p dcc # Partition to submit to
#SBATCH --mem 8192 # 8GB memory 
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Number of requested GPU (1). Max 8


#############################################################
### training ###

# for single GPU
#python train.py -opt=options/train/ir-sde.yml
python train.py -opt=options/train/refusion.yml

# for multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/train/ir-sde.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py -opt=options/train/refusion.yml --launcher pytorch

#############################################################

### testing ###
# python test.py -opt=options/test/ir-sde.yml
# python test.py -opt=options/test/refusion.yml

#############################################################