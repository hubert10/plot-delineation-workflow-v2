#!/bin/sh
# The job name: you can choose whatever you want for this.
#SBATCH --job-name=cpu_job_predictions_debi_tiguet

# Your email address and the events for which you want to receive email
# notification (NONE, BEGIN, END, FAIL, ALL).
#SBATCH --mail-user=h.kanyamahanga@cgiar.org
#SBATCH --mail-type=ALL

# The compute configuration for the job. For a job that uses GPUs, the
# partition must be set to "gpu". This example script requests access
# to a single GPU, 16 CPUs, and 30 GB of RAM for a single PyTorch task.
#SBATCH --nodes=4  # Run all processes on 4 nodes
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 # Number of CPU cores per task
#SBATCH --mem=128gb        # Total memory limit

# Specifies how long the job will be allowed to run in HH:MM:SS.
#SBATCH --time=05:05:05

# The log file for all job output. Note the special string "%j", which
# represents the job number.
#SBATCH --output=cpu_job_predictions_debi_tiguet_%j.out

# Prints the working directory, name of the assigned node, and
# date/time at the top of the output log.
pwd; hostname; date

module load tensorflow/2.4.1

# This should be the command you would use if you were running your TensorFlow application from the terminal.
python debi_tiguet_predictions.py

date

sbatch --account=hoogenboom --qos=hoogenboom cpu_job_predictions_debi_tiguet.sh
