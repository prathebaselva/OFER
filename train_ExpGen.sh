#!/bin/sh
#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=2-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python trainExpGen.py --cfg './src/configs/config_flameparamdiffusion_exp.yml' 
