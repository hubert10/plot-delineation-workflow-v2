#!/bin/sh
# The job name: you can choose whatever you want for this.
#SBATCH --job-name=cpu_job_tile_4096_4096

# Your email address and the events for which you want to receive email
# notification (NONE, BEGIN, END, FAIL, ALL).
#SBATCH --mail-user=h.kanyamahanga@cgiar.org
#SBATCH --mail-type=ALL

# The compute configuration for the job. For a job that uses GPUs, the
# partition must be set to "gpu". This example script requests access
# to a 32 CPUs, and 1000 GB of RAM for a single PyTorch task.
#SBATCH --nodes=1  # Run all processes on a single node
#SBATCH --ntasks=1  # Run only task
#SBATCH --cpus-per-task=32 # Number of CPU cores per task
#SBATCH --mem=500gb        # Total memory limit

# Specifies how long the job will be allowed to run in HH:MM:SS.
#SBATCH --time=7:00:00 # Expected run time for this job task

# The log file for all job output. Note the special string "%j", which
# represents the job number.
#SBATCH --output=logs/cpu_job_tile_4096_4096_%j.out

# Prints the working directory, name of the assigned node, and
# date/time at the top of the output log.
pwd; hostname; date

module load tensorflow/2.4.1

# This should be the command you would use if you were running your TensorFlow application from the terminal.
python make_predictions.py

date