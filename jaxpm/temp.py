# Now let's compute 2pt stats
_, pk_egd = power_spectrum(
    compensate_cic(tot_delta_egd), boxsize=np.array([25.0] * 3), kmin=np.pi / 25.0, dk=2 * np.pi / 25.0
)

_, xpk_egd = cross_correlation_coefficients(
    compensate_cic(tot_delta_hydro),
    compensate_cic(tot_delta_egd),
    boxsize=np.array([25.0] * 3),
    kmin=np.pi / 25.0,
    dk=2 * np.pi / 25.0,
)

fig, ax = plt.subplots(figsize=[13, 5], ncols=2)

ax[0].axhline(1.0, color="black")
ax[0].semilogx(k, (pk_dmo / pk_hydro), label="DMO")
ax[0].semilogx(k, (pk_egd / pk_hydro), label="DMO+EGD")
ax[0].set(
    xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",
    ylabel=r"$ P^{DMO}(k) \ / \ P^{Hydro}(k)$",
    xlim=(0.1, 15),
    ylim=(0.0, 1.5),
)
ax[0].grid(True)
ax[0].legend()

ax[1].axhline(1.0, color="black")
ax[1].semilogx(k, xpk / (jnp.sqrt(pk_dmo) * jnp.sqrt(pk_hydro)), label="DMO")
ax[1].semilogx(k, xpk_egd / (jnp.sqrt(pk_egd) * jnp.sqrt(pk_hydro)), label="DMO+EGD")
ax[1].set(
    xlabel=r"$k$ [$h \ \mathrm{Mpc}^{-1}$]",
    ylabel=r"$ P_{cross}(k) \ / \sqrt{ P^{DMO}(k) \ P^{Hydro}(k)}$",
    xlim=(0.1, 15),
    ylim=(0.95, 1.05),
)
ax[1].grid()
ax[1].legend()
