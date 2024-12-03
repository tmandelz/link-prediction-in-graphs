#!/bin/bash
set -e
USER_HOME="tmandelz"
BASE_PATH="/mnt/nas05/data01/cluster-user-homes/${USER_HOME}"

if [ -f ~/slurmenv.sh ]; then
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] sourcing ~/slurmenv.sh"
  source ~/slurmenv.sh
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] slurmenv.sh sourced"
fi

# DOWNLOAD image
mkdir -p "${BASE_PATH}/images"
if [ ! -f "${BASE_PATH}/images/link-prediction-in-graphs.sif" ]; then
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] pulling image link-prediction-in-graphs started"
  srun --pty --job-name="image" -p performance singularity pull --no-cache --name "${BASE_PATH}/images/link-prediction-in-graphs.sif" docker://cr.gitlab.fhnw.ch/thomasoliver.mandelz/link-prediction-in-graphs:latest
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] image link-prediction-in-graphs pulled"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="link-prediction-gnn"
#SBATCH --cpus-per-task=1
#SBATCH -p performance
#SBATCH -t 12:00:00
#SBATCH --gpus=1
#SBATCH --out="${BASE_PATH}/logs/link-prediction-in-graphs/%j.log"
#SBATCH --exclude=node15,sdas2,gpu22a,gpu22b

# git repo bind into docker image
singularity exec --nv --bind "${BASE_PATH}/dev/link-prediction-in-graphs:/work/project" --pwd /work/project "${BASE_PATH}/images/link-prediction-in-graphs.sif" python modelling/gnn/gnn.py $@
EOF
