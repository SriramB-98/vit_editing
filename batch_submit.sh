#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=vit_editing                                   # sets the job name
#SBATCH --output=vit_editing.out.%j                              # indicates a file to redirect STDOUT to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=vit_editing.out.%j                               # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=10:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=medium                                           # set QOS, this will determine what resources can be requested
#SBATCH --nodes=4                                              # how many nodes you will require; default is 1 node
#SBATCH --ntasks=4                                              # request 4 cpu cores be reserved for your node total
#SBATCH --ntasks-per-node=1                                    
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --partition=tron
#SBATCH --account=nexus


srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32gb bash -c "hostname; python identify_causal_neurons.py --lr 5e-2 --epochs 70 --weight_init -2 --lamb 0.5 --gamma 1 --beta 1 --hook_names blocks.9 --batch_size 256" &
srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32gb bash -c "hostname; python identify_causal_neurons.py --lr 5e-2 --epochs 70 --weight_init -2 --lamb 0.5 --gamma 1 --beta 1 --hook_names blocks.8 --batch_size 256" &
srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32gb bash -c "hostname; python identify_causal_neurons.py --lr 5e-2 --epochs 70 --weight_init -2 --lamb 0.5 --gamma 1 --beta 1 --hook_names blocks.2 --batch_size 256" &
srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32gb bash -c "hostname; python identify_causal_neurons.py --lr 5e-2 --epochs 70 --weight_init -2 --lamb 0.5 --gamma 1 --beta 1 --hook_names blocks.1 --batch_size 256" &
wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
