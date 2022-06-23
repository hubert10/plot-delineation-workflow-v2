###### Tutorials on sbatch GPU on Hipergator #####

#1. Account and QOS(quality of service) limits under SLURM
# Link: https://help.rc.ufl.edu/doc/Account_and_QOS_limits_under_SLURM
# QOSes can be thought of as pools of computational (CPU cores),
# memory (RAM), maximum run time (time limit) resources

#2. Get your groud by running: id
uid=12182(kanyamahanga.h) gid=3686(hoogenboom) groups=3686(hoogenboom)
# The user kanyamahanga.h is part hoogenboom group
# and the share the same resources

#3. QOS Resource Limits
# CPU cores, Memory (RAM), GPU accelerators, software licenses, etc.
# are referred to as Trackable Resources (TRES) by the scheduler. 
# The TRES available in a given QOS are determined by the group's investments and the QOS configuration.

#$ showQos hoogenboom
hoogenboom           hoogenboom qos                 cpu=256,gres/gpu=0,mem=2000G                       
256 

We can see that the hoogenboom investment QOS has a pool of 256 CPU cores, 2000GB of RAM, and no GPUs. This pool of resources 
is shared among all members of the hoogenboom group.

#3 To show the status of any SLURM account as well
# as the overall usage of HiPerGator resources, use the following command from the ufrc environment module

#$ slurmInfo hoogenboom

hoogenboom           hoogenboom qos                 cpu=256,gres/gpu=0,mem=2000G                       256 

#3 Check running jobs and check its status
squeue

sstat -j 18684648
#5 Get your job status
squeue -u kanyamahanga.h

kanyamahanga.h@login2 ~]$ squeue -u kanyamahanga.h
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          18684648 hpg-defau spawner- kanyamah  R 1-05:19:48      1 c0702a-s2
[kanyamahanga.h@login2 ~]$ 

#6 Slurm commands link
# https://help.rc.ufl.edu/doc/SLURM_Commands
# https://help.rc.ufl.edu/doc/Sample_SLURM_Scripts#Basic.2C_Single-Threaded_Job

 sstat -j 18684648.batch -o maxrss
