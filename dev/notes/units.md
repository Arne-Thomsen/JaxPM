$\def\vswift{v_\mathrm{SWIFT}}$
$\def\vgadget{v_\mathrm{GADGET}}$
$\def\vpeculiar{v_\mathrm{p}}$

# velocity

- coordinates:
    - physical: $r$
    - comoving: $x_c$: $r = ax_c$
- velocities:
    - comoving: $\dot x_c = \frac{\dot r}{a} - \frac{\dot a r}{a^2}$
    - peculiar: $\vpeculiar = a \dot x_c$
    - GADGET: $\vgadget = \sqrt{a}\dot x_c$
    - SWIFT: $\vswift = a^2\dot x_c = a \vpeculiar$
    - Conversion
        - Setting $H_0 := 1$ in the integration operators implies time units of $1/H_0$ instead of $s$
        - $\frac{km}{s} \frac{\text{mesh size}}{\text{box size}} =\frac{km}{s} \frac{h}{Mpc} = \frac{1}{100} \, 100 \frac{km \, h}{s \, Mpc} = \frac{1}{100} H_0$ with $H_0 = 100 h \frac{km}{s \, Mpc}$

```
gas_vel = data["PartType0/Velocities"][:]  # v_gadget (sqrt(a) km/s)
gas_vel *= np.sqrt(scale_factor)  # -> v_peculiar (a km/s)
gas_vel *= scale_factor  # -> v_swift (a^2 km/s)
gas_vel *= mesh_per_dim / box_size  # -> pm length (a^2 km/s h/Mpc), where [mesh_per_dim] = int, [box_size] = Mpc/h
gas_vel /= 100  # -> pm velocity (a^2 H_0)
```

# pressure
- density: $\rho_c = a^3 \rho$
- pressure: $P_c = a^{3 \gamma} P$

```
# density (comoving)
rho_gas = cic_paint(np.zeros([mesh_per_dim] * 3), gas_pos, gas_mass)
gas_rho = cic_read(rho_gas, gas_pos)  # dm_mass/pm_len^3

# internal energy (physical)
gas_U = data["PartType0/InternalEnergy"][:]  # (km/s)^2
gas_U *= (mesh_per_dim * scale_factor / (box_size * 100)) ** 2  # rescale like the velocity, pm_vel^2

# pressure (physical)
gamma = 5.0 / 3.0
gas_P = (gamma - 1.0) * gas_U * (gas_rho / scale_factor**3)  #  dm_mass*pm_vel^2/pm_len^3

# pressure (comoving)
gas_P *= scale_factor ** (3 * gamma)

```

### references
- https://github.com/tilmantroester/hydrox/blob/main/notes/equations_of_motion.md
- https://www.tng-project.org/data/docs/specifications/#parttype1
- https://camels.readthedocs.io/en/latest/snapshots.html?highlight=velocity#initial-conditions
- SWIFT: https://arxiv.org/pdf/2305.13380
- Gadget: https://wwwmpa.mpa-garching.mpg.de/gadget/gadget1-paper.pdf