#!/bin/sh
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH -e test.err
#SBATCH --mem=5G
#SBATCH --array=1-20

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=0-10:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1


#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1


python test.py --numcheckpoint 3 --cfg './src/configs/config_flameparamdiffusion_flame20.yml' --checkpoint1 'checkpoint/model_idrank.tar' --checkpoint2 'checkpoint/model_idgen_flame20.tar' --checkpoint3 'checkpoint/model_expgen_flame20.tar' --filename 'data/PICKPIK/validation/'${SLURM_ARRAY_TASK_ID}'.txt' --imagepath 'data/PICKPIK/' --outputpath 'output'
