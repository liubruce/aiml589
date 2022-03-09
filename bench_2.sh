#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd /vol/grid-solar/sgeusers/liuguoq
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific
# directory that should have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
#
# Mail me at the b(eginning) and e(nd) of the job
#
#$ -M brucelgq.liu@gmail.com
#$ -m be
#
if [ -d /local/tmp/liuguoq/$JOB_ID.$SGE_TASK_ID ]; then
        cd /local/tmp/liuguoq/$JOB_ID.$SGE_TASK_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP liuguoq "
        ls -la /local/tmp/liuguoq
        echo "Exiting"
        exit 1
fi
#
# Now we are in the job-specific directory so now can do something useful
#
# Stdout from programs and shell echos will go into the file
#    scriptname.o$JOB_ID
#  so we'll put a few things in there to help us see what went on
#
#
# Copy the input file to the local directory
#
bash
export HOME=/vol/grid-solar/sgeusers/liuguoq
export PATH=$HOME/anaconda3/envs/mujoco/bin:$HOME/miniconda3/bin/:$PATH
#conda activate mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
cp -r /vol/grid-solar/sgeusers/liuguoq/aiml589 .
echo ==WHATS THERE HAVING COPIED STUFF OVER AS INPUT==
ls -la
#
# Note that we need the full path to this utility, as it is not on the PATH
#
algorithmName=${1}'.py'
echo $algorithmName
problemName=$2
pyName="run.py"
cd ./aiml589
#/vol/grid-solar/sgeusers/liuguoq/anaconda3/envs/mujoco/bin/python run.py specs/td3_on_mujoco.py
# ./bench_run.sh
python $pyName $algorithmName $problemName $SGE_TASK_ID
#
echo ==AND NOW, HAVING DONE SOMTHING USEFUL AND CREATED SOME OUTPUT==
#ls -la
#
# Now we move the output to a place to pick it up from later
#  (really should check that directory exists too, but this is just a test)
#
mkdir -p /vol/grid-solar/sgeusers/liuguoq/results/$1/$problemName/$JOB_ID.$SGE_TASK_ID
cp -r ./out/ /vol/grid-solar/sgeusers/liuguoq/results/$1/$problemName/$JOB_ID.$SGE_TASK_ID
rm -r ./out
#
echo "Ran through OK"

