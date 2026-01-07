cv_dir = "/pscratch/sd/a/athomsen/flatiron/CAMELS/Sims/SIMBA/CV"
i0 = 0
i1 = -1

all_ref_pk = []
all_pm_pk = []
all_hpm_pk = []
all_egd_pk = []

all_res_pk = []
all_res_cross = []
for i in tqdm(range(3)):
    sim_file = os.path.join(cv_dir, f"CV_{i}", f"parts={parts_per_dim},mesh={mesh_per_dim}.h5")

    # ref
    with h5py.File(sim_file) as f:
        i_snap = (i0, i1)
        
        scales = f["scales"][:]
        dm_poss = f["dm_poss"][i_snap, ...]
        dm_vels = f["dm_vels"][i_snap, ...]
        gas_poss = f["gas_poss"][i_snap, ...]
        gas_vels = f["gas_vels"][i_snap, ...]

    ref_pos = gas_poss[1]
    ref_pk = get_pk(ref_pos)
    all_ref_pk.append(ref_pk)
        
    y0 = (dm_poss[0], dm_vels[0], gas_poss[0], gas_vels[0])
    t0 = scales[i0]
    tsave = [scales[i1]]

    # pm
    pm_res = solve_ode(
        y0, 
        t0, 
        tsave, 
        pressure_model=None, 
        training=False, 
        nt=1, 
        tstep=scales
    )
    pm_pk = get_pk(pm_res[2])
    pm_x = get_cross(
    

    # hpm
    hpm_res = solve_ode(
        y0, 
        t0, 
        tsave, 
        pressure_model=pressure_model, 
        training=False, 
        nt=1, 
        tstep=scales
    )

    ref_pos = gas_poss[1]
    res_pos = res[2]
    
    k, ref_pk = get_pk(ref_pos)
    _, res_pk = get_pk(res_pos)
    _, res_cross = get_cross(res_pos, ref_pos)

    
    all_ref_pk.append(ref_pk)
    all_res_pk.append(res_pk)
    all_res_cross.append(res_cross)


all_ref_pk = jnp.stack(all_ref_pk)
all_res_pk = jnp.stack(all_res_pk)
all_res_cross = jnp.stack(all_res_cross)