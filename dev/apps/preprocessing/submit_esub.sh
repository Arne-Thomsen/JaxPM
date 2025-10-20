# Euler

esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=jobarray --function=all --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm
esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=jobarray --function=merge --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm
esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=run --function=merge --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm

esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=camels_cv --system=slurm

esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py \
    --mode=jobarray --function=all --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm \
    --camels_cv_dir="/cluster/work/refregier/athomsen/flatiron/CAMELS/Sims/SIMBA/CV" --build_from_scratch

esub /cluster/home/athomsen/flatiron/repos/JaxPM/dev/hpm/preprocessing.py \
    --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=camels_cv --system=slurm \
    --camels_cv_dir="/cluster/work/refregier/athomsen/flatiron/CAMELS/Sims/SIMBA/CV" --build_from_scratch

# Perlmutter ##########################################################################################################

# debug
esub /global/homes/a/athomsen/flatiron/JaxPM/dev/apps/preprocessing/run_preprocessing.py \
    --mode=run --function=main --n_jobs=1 --tasks="3" --job_name=camels_cv --system=slurm \
    --camels_cv_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/Sims/SIMBA/CV" \
    --out_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/h5/SIMBA/CV" \
    --parts_per_dim=16 --mesh_per_dim=16 --build_from_scratch \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub /global/homes/a/athomsen/flatiron/JaxPM/dev/apps/preprocessing/run_preprocessing.py \
    --mode=jobarray --function=all --n_jobs=1 --tasks="4" --job_name=camels_cv --system=slurm \
    --camels_cv_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/Sims/SIMBA/CV" \
    --out_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/h5/SIMBA/CV" \
    --parts_per_dim=128 --mesh_per_dim=128 --build_from_scratch \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# production
esub /global/homes/a/athomsen/flatiron/JaxPM/dev/apps/preprocessing/run_preprocessing.py \
    --mode=jobarray --function=all --n_jobs=27 --tasks="0>27" --job_name=camels_cv --system=slurm \
    --camels_cv_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/Sims/SIMBA/CV" \
    --out_dir="/pscratch/sd/a/athomsen/flatiron/CAMELS/h5/SIMBA/CV" \
    --parts_per_dim=128 --mesh_per_dim=128 --build_from_scratch \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
