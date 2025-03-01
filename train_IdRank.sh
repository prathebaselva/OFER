#!/bin/sh
#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=3-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1

#module load gcc/7.1.0
module load cuda11/11.2.1

python trainIdRank.py --cfg './src/configs/config_flameparamrank_flame23.yml' --toseed 0 
