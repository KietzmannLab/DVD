#!/bin/bash
#! How much wallclock time will be required?
#SBATCH --time=47:59:00 # Run time
#! Name of the job:
#SBTACH --job-name H100-test
#! Specify the number of GPUs per node (between 1 and n; must be n if nodes>1).
#SBATCH --gres=gpu

#! How much wmemory will be required?
#SBATCH --mem 150G #600G # #!TODO cannot be too large, the whole cluster only has 2T --> analysis ecoset, we need to be large
#! How many cpus will be required?
#SBATCH -c 30 #TODO
#! Which partition? klab-gpu klab-l40s gpu
#SBATCH -p gpu
#! How many whole nodes should be allocated?
#SBATCH --nodes 1  # Number of reaquested nodes 
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1 #* 2 for 2 gpu
##SBATCH --ntasks-per-node=1          # Number of tasks per node (e.g., GPUs per node)

#! error and output file?
#SBATCH --error=./logs/slurm_logs/h100_multi_error.o%j
#SBATCH --output=./logs/slurm_logs/h100_multi_output.o%j
#! What types of email messages do you wish to receive?
#SBATCH --mail-user=lzejin@uni-osnabrueck.de
##SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

#! new add by zejin
# Set environment variables for NCCL and PyTorch
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO


# add by myself
spack unload #unload all the previously loaded packages and load the required packages below
echo $CONDA_PREFIX # to print the path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ #Set the path for Nvidia libs
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX #XLA path specification
echo $LD_LIBRARY_PATH # to print the path

# add by myself 16th Dec
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 
## term handler the function is executed once the job gets the TERM signal
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"
conda activate bias

# ABOVE: the file starts with any options for the job scheduler, same as for submission of a single job

# this file is called from the array submission script, so we know that the first argument it gets is the config file
# that holds parameters for each job in the array. Retrieve it here
file=$1
echo "Config file: $file"

# every job submitted from an array has access to its index, retrieve it here
# get slurm array index
index=$SLURM_ARRAY_TASK_ID
echo "Index: $index"

# get index row from file. This gives us the parameters we need for this job
line=$(sed "${index}q;d" $file)
echo "Running with opt. $line"  # show the options in the terminal / logs for debugging

# the rest of this file is the same as for submission of a single job
# source ~/setup_H100.sh

# grab config from list of configs and submit
# ~/miniconda3/envs/avalanche-h100/bin/python analysis_activation_trajectory_single_run.py $line  # pass the options to the python interpreter, we can access them in the script with argparse / hydra etc. as always

#! normal
python main.py $line  # pass the options to the python interpreter, we can access them in the script with argparse / hydra etc. as always

#! SSL
# python main_simclr.py $line  # pass the options to the python interpreter, we can access them in the script with argparse / hydra etc. as always