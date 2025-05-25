esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=jobarray --function=all --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm
esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=jobarray --function=merge --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm
esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=run --function=merge --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm

esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=camels_cv --system=slurm
