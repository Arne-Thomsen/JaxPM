import os, argparse, h5py

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORMS_NAME"] = "cpu"

from tqdm import tqdm
from jaxpm import camels, data


def resources(args):
    resources = {
        "main_time": 4,
        "main_memory": 4096,
        "main_n_cores": 8,
        "main_scratch": 0,
        "merge_time": 4,
        "merge_n_cores": 8,
        "merge_scratch": 0,
    }

    return resources


def setup(args):
    description = "This is an example script that can be used by esub"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument(
        "--camels_cv_dir",
        type=str,
        default="/cluster/scratch/athomsen/CV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--parts_per_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--mesh_per_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--build_from_scratch",
        action="store_true",
    )
    args, _ = parser.parse_known_args(args)

    if args.out_dir is None:
        args.out_dir = args.camels_cv_dir

    return args


def main(indices, args):
    args = setup(args)

    # each index corresponds to one CV cosmology
    for index in indices:
        CV = os.path.join(args.camels_cv_dir, f"CV_{index}")

        camels_dict = camels.load_CV_snapshots(
            CV,
            args.mesh_per_dim,
            args.parts_per_dim,
            return_hydro=True,
            force_h5=args.build_from_scratch,
        )

        X_particle, X_field, Y_particle, Y_field = data.get_offline_regression_data(
            camels_dict,
            include_scale=True,
            standardize_input=False,
            standardize_label=False,
        )

        index_file = f"hpm_parts={args.parts_per_dim},mesh={args.mesh_per_dim}.h5"
        with h5py.File(os.path.join(CV, index_file), "w") as f:
            f.create_dataset("X_particle", data=X_particle)
            f.create_dataset("X_field", data=X_field)
            f.create_dataset("Y_particle", data=Y_particle)
            f.create_dataset("Y_field", data=Y_field)

        yield index


def merge(indices, args):
    args = setup(args)

    index_file = f"hpm_parts={args.parts_per_dim},mesh={args.mesh_per_dim}.h5"

    merge_file = f"merged_hpm_parts={args.parts_per_dim},mesh={args.mesh_per_dim}.h5"
    out_file = os.path.join(args.out_dir, merge_file)

    with h5py.File(out_file, "w") as f_out:
        # initialized datasets with the shape of the first index
        with h5py.File(os.path.join(args.camels_cv_dir, f"CV_{indices[0]}", index_file), "r") as f_in:
            f_out.create_dataset(
                "X_particle", shape=(len(indices),) + f_in["X_particle"].shape, dtype=f_in["X_particle"].dtype
            )
            f_out.create_dataset("X_field", shape=(len(indices),) + f_in["X_field"].shape, dtype=f_in["X_field"].dtype)
            f_out.create_dataset(
                "Y_particle", shape=(len(indices),) + f_in["Y_particle"].shape, dtype=f_in["Y_particle"].dtype
            )
            f_out.create_dataset("Y_field", shape=(len(indices),) + f_in["Y_field"].shape, dtype=f_in["Y_field"].dtype)

        # now fill the datasets with the data from each index
        for index in tqdm(indices):
            HPM = os.path.join(args.camels_cv_dir, f"CV_{index}", index_file)

            with h5py.File(HPM, "r") as f_in:
                f_out["X_particle"][index] = f_in["X_particle"][:]
                f_out["X_field"][index] = f_in["X_field"][:]
                f_out["Y_particle"][index] = f_in["Y_particle"][:]
                f_out["Y_field"][index] = f_in["Y_field"][:]

    print(f"Saved merged file to {out_file}")
