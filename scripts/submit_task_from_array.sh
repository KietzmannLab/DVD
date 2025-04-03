#!/bin/bash
#! How much wallclock time will be required?
#SBATCH --time=47:59:00 # Run time
#! Name of the job:
#SBTACH --job-name H100-test
#! Specify the number of GPUs per node (between 1 and n; must be n if nodes>1).
#SBATCH --gres=gpu:2
#! How much wmemory will be required?
#SBATCH --mem 200G #600G # #!TODO cannot be too large, the whole cluster only has 2T --> analysis ecoset, we need to be large
#! How many cpus will be required?
#SBATCH -c 20 #TODO
#! Which partition? klab-gpu klab-l40s gpu
#SBATCH -p gpu
#! How many whole nodes should be allocated?
#SBATCH --nodes 1  # Number of reaquested nodes 
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1 
##SBATCH --ntasks-per-node=1          # Number of tasks per node (e.g., GPUs per node)

#! error and output file?
#SBATCH --error=./logs/slurm_logs_v2/h100_multi_error.o%j
#SBATCH --output=./logs/slurm_logs_v2/h100_multi_output.o%j
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
conda activate h100



# 1) The first script argument is the name of the config file we wrote
file=$1
echo "Config file: $file"

# 2) The Slurm array index for this job
index=$SLURM_ARRAY_TASK_ID
echo "Index: $index"

# 3) Extract the line corresponding to this array index
line=$(sed "${index}q;d" "$file")
echo "Running command: $line"

# 4) Evaluate that entire line
eval "$line"

#! SSL
# python main_simclr.py $line  # pass the options to the python interpreter, we can access them in the script with argparse / hydra etc. as always