# Catalogue pédagogique des fonctions Shamrock

### Pour écrire un setup et pour ajouter un terme physique

> Ce catalogue est tiré directement du code (état de la branche courante). Chaque entrée
> renvoie au fichier source (`chemin:ligne`) où la fonction est définie : c'est **la**
> référence en cas de doute, les signatures pouvant évoluer.

---

## Sommaire

- [0. Carte mentale : comment Shamrock est organisé](#0-carte-mentale--comment-shamrock-est-organisé)
- **Partie I — Écrire un setup (Python)**
  - [1. Squelette canonique d'un setup](#1-squelette-canonique-dun-setup)
  - [2. Système, contexte, unités](#2-système-contexte-unités)
  - [3. Choisir un modèle](#3-choisir-un-modèle)
  - [4. Configurer le solveur SPH (`cfg.*`)](#4-configurer-le-solveur-sph-cfg)
  - [5. Scheduler, boîte, masse des particules](#5-scheduler-boîte-masse-des-particules)
  - [6. Générer les particules (`SPHSetup`)](#6-générer-les-particules-sphsetup)
  - [7. Initialiser / modifier des champs](#7-initialiser--modifier-des-champs)
  - [8. Faire avancer la simulation](#8-faire-avancer-la-simulation)
  - [9. Entrées/sorties et reprise](#9-entréessorties-et-reprise)
  - [10. Récupérer, analyser et visualiser les données](#10-récupérer-analyser-et-visualiser-les-données)
  - [11. Orchestrer une simulation : `SimulationRunner` & co.](#11-orchestrer-une-simulation--simulationrunner--co)
  - [12. Le modèle Godunov / Ramses (grille AMR)](#12-le-modèle-godunov--ramses-grille-amr)
  - [13. Les modèles GSPH et Zeus](#13-les-modèles-gsph-et-zeus)
- **Partie II — Ajouter un terme physique (C++)**
  - [14. Architecture interne d'un modèle](#14-architecture-interne-dun-modèle)
  - [15. Anatomie d'un pas de temps SPH](#15-anatomie-dun-pas-de-temps-sph)
  - [16. Briques C++ de base](#16-briques-c-de-base)
  - [17. Recettes : ajouter de la physique pas à pas](#17-recettes--ajouter-de-la-physique-pas-à-pas)
  - [18. Règles du jeu (déterminisme, conventions, build, tests)](#18-règles-du-jeu)
- [Annexe A — Champs SPH disponibles](#annexe-a--champs-sph-disponibles)
- [Annexe B — Où chercher quoi](#annexe-b--où-chercher-quoi)

---

## 0. Carte mentale : comment Shamrock est organisé

```text
 Script Python (setup)                       ← Partie I
   │  shamrock.get_Model_SPH(...), cfg.set_*, setup.make_generator_*, model.evolve_until...
   ▼
 Bindings pybind11  (src/shammodels/<modèle>/src/py*Model.cpp)
   │  traduisent chaque méthode Python en appel C++
   ▼
 Model<Tvec, Kernel>        façade « utilisateur » (Model.hpp)
   └── Solver<Tvec, Kernel>                 ← Partie II
         ├── SolverConfig   : TOUTE la configuration physique (variants EOS, AV, MHD, dust, forces…)
         ├── SolverStorage  : champs temporaires (pression, omega, voisins, ghosts…)
         ├── SolverGraph    : graphe de nœuds (INode) reliés par des arêtes (edges) typées
         └── modules/       : un module/nœud = une opération physique (EOS, forces, CFL…)
               │
               ▼
         noyaux GPU : sham::kernel_call / sham::distributed_data_kernel_call (SYCL)
               │
               ▼
 PatchScheduler (ctx.sched) : découpage du domaine en patches, distribution MPI,
                              chaque patch = PatchDataLayer = ensemble de PatchDataField
```

Trois idées à retenir :

1. **Tout ce qui est physique est d'abord une option de `SolverConfig`** (souvent un
   `std::variant`). Le script Python ne fait que remplir cette config (`cfg.set_*`), puis
   `model.set_solver_config(cfg)`.
2. **La config détermine le *layout*** : la liste des champs portés par chaque particule
   (`SolverConfig::set_layout`, voir [annexe A](#annexe-a--champs-sph-disponibles)). Ajouter une
   variable évoluée = ajouter un champ au layout.
3. **Un terme physique = un nœud** (`INode`) qui lit des arêtes d'entrée et écrit des arêtes
   de sortie, et dont le cœur est un noyau GPU appelé patch par patch.

---

# Partie I — Écrire un setup (Python)

## 1. Squelette canonique d'un setup

Ordre **obligatoire** des étapes (dérivé de `examples/sph/run_sod.py` et
`examples/sph/run_sph_basic_disc.py`) :

```python
import shamrock

# (1) Initialisation du système (seulement si lancé avec `python`, pas avec ./shamrock)
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")  # "<device alt>:<device calcul>" (ids de --smi)

# (2) Contexte + modèle
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

# (3) Configuration physique
cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(1.4)
cfg.print_status()
model.set_solver_config(cfg)  # ← AVANT init_scheduler (le layout en dépend)

# (4) Scheduler puis boîte
model.init_scheduler(int(1e8), 1)  # split / merge des patches
model.resize_simulation_box((-1, -0.5, -0.5), (1, 0.5, 0.5))

# (5) Génération des particules
setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, (-1, -0.5, -0.5), (1, 0.5, 0.5))
setup.apply_setup(gen)

# (6) Champs initiaux + masse des particules
model.set_value_in_a_box("uint", "f64", u0, (-1, -0.5, -0.5), (1, 0.5, 0.5))
model.set_particle_mass(model.total_mass_to_part_mass(totmass))

# (7) Pas de temps
model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)
model.timestep()  # 1er pas : converge h, initialise l'intégrateur
model.evolve_until(t_end)
```

**Pièges classiques**

| Piège | Conséquence | Remède |
|---|---|---|
| `set_solver_config` après `init_scheduler` | le layout (champs) ne contient pas les champs de la physique demandée | toujours config → `set_solver_config` → `init_scheduler` |
| `add_sink` avant `init_scheduler` | exception explicite (`Model.hpp:191`) | appeler après `init_scheduler` |
| masse de particule non fixée | `gpart_mass = 0` → densités nulles | `set_particle_mass` (model ou cfg) |
| EOS localement isotherme sans `cs_profile` dans `make_generator_disc_mc` | exception | fournir `cs_profile` **ou** `cs_field` |
| `ctx.collect_data()` sur une grosse simu MPI | tout est rapatrié sur le rang 0 → OOM | utiliser `compute_field`/rendus, ou des callbacks par patch |

---

## 2. Système, contexte, unités

Les bindings C++ s'enregistrent via `ON_PYTHON_INIT` dans le module embarqué `pyshamrock`,
réexporté par le paquet Python `src/pylib/shamrock/`. Sous-modules : `sys`, `phys`, `phys.eos`,
`math`, `math.sphkernel`, `comm`, `model_sph`, `model_gsph`, `model_zeus`, `model_ramses`…

> `import shamrock` n'importe que `shamrock.matplotlib` et `shamrock.utils`. Il faut importer
> explicitement `shamrock.utils.analysis`, `shamrock.utils.SimulationRunner`,
> `shamrock.utils.numba_helper`, `shamrock.utils.DustMRNDistribution`.

### 2.1 Système (`src/shampylib/src/pyShamsys.cpp`, `pyNodeInstance.hpp`)

| Fonction | Arguments | Rôle |
|---|---|---|
| `shamrock.sys.init(sycl_cfg)` | `"alt:calcul"`, ex. `"0:0"` ; ou `"auto:<motif>"` | démarre SYCL + MPI. **Ne pas appeler** si le script est lancé par `./shamrock --rscript` (d'où le test `is_initialized()`) |
| `shamrock.sys.is_initialized()` | | |
| `shamrock.sys.world_rank()`, `world_size()`, `mpi_barrier()` | | MPI (ex. ne tracer que sur le rang 0) |
| `shamrock.sys.get_compute_device_properties()` | | dict : vendor, backend, nom, mémoire… |
| `shamrock.change_loglevel(n)` | int | verbosité (1 dans les exemples) |
| `shamrock.enable_experimental_features()` | | **obligatoire** pour les options marquées expérimentales : poussière SPH (`dust != None`), MHD non idéale, auto-gravité/nouvelle AMR/scalaires passifs Ramses |
| `shamrock.get_wtime()`, `get_wtime_sync()` | | temps mural (`_sync` : collectif MPI) |
| `shamrock.version_string()`, `print_git_info()`, `print_compile_arg()` | | infos de build |
| `shamrock.dump_profiling(prefix)`, `dump_profiling_chrome(prefix)` | | profiling (si compilé avec `SHAMROCK_USE_PROFILING`) |

### 2.2 Contexte (`shamrock.Context`, `src/shampylib/src/pyShamrockCtx.cpp`)

| Méthode | Rôle |
|---|---|
| `Context()` | conteneur des données (scheduler, layout) |
| `pdata_layout_new()` | réinitialise le layout des champs — toujours avant `get_Model_*` |
| `pdata_layout_add_field(fname, nvar, type_str)`, `pdata_layout_print()` | bas niveau (le modèle s'en charge) |
| `collect_data()` | dict numpy de tous les champs, rassemblé sur le rang 0 |
| `set_coord_domain_bound(min, max)` | = `model.resize_simulation_box` |
| `get_patch_list_global()`, `dump_status()`, `scheduler_step(...)` | inspection / debug du découpage en patches |

### 2.3 Unités et constantes (`src/shampylib/src/pyUnits.cpp`)

```python
si = shamrock.UnitSystem()  # SI
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(), unit_length=sicte.au(), unit_mass=sicte.sol_mass()
)  # kw-only, valeurs en SI
ucte = shamrock.Constants(codeu)
G = ucte.G()  # G en unités code
cfg.set_units(codeu)
```

| Objet / méthode | Rôle |
|---|---|
| `UnitSystem(*, unit_time=1, unit_length=1, unit_mass=1, unit_current=1, unit_temperature=1, unit_qte=1, unit_lumint=1)` | système d'unités (valeurs SI) |
| `codeu.get(name, power=1, pref="None")` | valeur de l'unité `name` en unités code (ex. `get("m", power=-3, pref="c")` = cm⁻³) |
| `codeu.to(name, power=1, pref="None")` | conversion inverse |
| `to_json()` / `from_json(d)` | sérialisation |
| `Constants(units).<cte>(power=1)` | constantes : `c, h, e, k, Na, G, hbar, mu_0, Z_0, epsilon_0, ke, kb, sigma` ; masses `proton_mass, electron_mass, earth_mass, jupiter_mass, sol_mass, planck_mass, dalton` ; longueurs `au, light_year, parsec, solar_radius, earth_radius, planck_length` ; durées `second, minute, hour, day, year` |

Noms d'unités (`src/shamunits/include/shamunits/unit_table.hpp`) : `s, m, kg, A, K, mol, cd, Hz,
N, Pa, J, W, C, V, F, ohm, S, Wb, T, H, …, mn, hr, dy, yr, au, ly, pc, rsol, rearth, sol_mass, eV, erg`.

### 2.4 Physique utilitaire (`shamrock.phys`, `src/shampylib/src/pyShamphys.cpp`)

| Fonction (kw-only) | Rôle |
|---|---|
| `keplerian_speed(G=, M=, R=)` ou `(M=, R=, units=)` | vitesse képlérienne |
| `hill_radius(R=, m=, M=)`, `schwarzschild_radius(M=, G=, c=)` | rayons caractéristiques |
| `get_binary_pair(m1=, m2=, a=, e=, nu=, G=)` | positions/vitesses d'une binaire (anomalie vraie `nu`) |
| `get_binary_rotated(..., roll=, pitch=, yaw=)` | idem, orbite tournée (→ `add_sink`) |
| `rotate_point(point=, roll=, pitch=, yaw=)` | |
| `free_fall_time(rho, G)` (positionnel) | |
| `epstein_stopping_time(rho_grain=, s_grain=, rho=, cs=, gamma=, f=1.0)` | temps d'arrêt d'Epstein |
| `epstein_supersonic_correction(delta_v=, cs=)` | |
| `HydroSoundwave(cs=, k=, rho_tilde=, v_tilde=)` → `.get_omega()`, `.get_value(t, x)` | onde sonore analytique |
| `SodTube(gamma=, rho_1=, P_1=, rho_5=, P_5=)` → `.get_value(t, x) -> (rho, vx, P)` | tube de Sod exact |
| `SedovTaylor()` → `.get_value(x)` | |
| `phys.eos.eos_Machida06(...)`, `phys.eos.eos_Fermi(mu_e=, rho=)` | EOS appelables depuis Python (tests) |

### 2.5 Mathématiques (`shamrock.math`)

| Fonction | Rôle |
|---|---|
| `get_ideal_hcp_box(dr, bmin, bmax)` | ajuste la boîte sur un nombre entier de mailles HCP → `(bmin, bmax)` |
| `get_periodic_hcp_box(dr, imin, imax)` | boîte HCP périodique à partir de coordonnées entières de réseau |
| `get_ideal_fcc_box`, `get_periodic_fcc_box` | équivalents FCC |
| `Ray_f64_3(origin, direction)`, `RingRay_f64_3(center, radius, e_x, e_y)` | rayons pour `render_column_integ` / `render_azymuthal_integ` |
| `AABB_f64_3(min, max)` | boîtes alignées |
| `math.sphkernel.<K>_W3d(r, h)`, `<K>_dW3d`, `<K>_f(q)`, `<K>_Rkern()`… | noyaux SPH en Python (tracés, tests) |
| `derivative_*`, `euler_ode`, `least_squares` | petits outils numériques |

### 2.6 Algèbre sur champs distribués, histogrammes, Phantom

| Fonction | Rôle |
|---|---|
| `shamrock.map_fields_f64(func, **fields)` / `map_fields_f64_3` | applique `func(size, **arrays)` patch par patch à des `Field` (issus de `model.compute_field`) → nouveau `Field` ; `.collect_data()` pour récupérer |
| `shamrock.compute_histogram(bin_edges, x_field, y_field, do_average=False)` | profils radiaux, etc. (distribué) |
| `compute_histogram_convolve_x(...)`, `compute_histogram_2d(...)` | |
| `shamrock.NeighCacheStrategy.{SingleStage, TwoStage, …}` | enum pour `cfg.set_neigh_cache_strategy` |
| `shamrock.load_phantom_dump(fname)` → `PhantomDump` (`save_dump`, `print_state`, `read_header_float/int`) | lire un dump Phantom |
| `shamrock.utils.phantom.load_simulation(folder, dump_file_name=, in_file_name=)` → `(ctx, model, in_params)` | reprendre une simulation Phantom complète |

---

## 3. Choisir un modèle

| Fonction Python | Modèle | Arguments |
|---|---|---|
| `shamrock.get_Model_SPH(context, vector_type, sph_kernel)` | SPH (style Phantom) | `vector_type="f64_3"` ; `sph_kernel ∈ {"M4","M6","M8","C2","C4","C6"}` (`pySPHModel.cpp`, fin de fichier) |
| `shamrock.get_Model_Ramses(...)` | Godunov AMR (grille) | voir [§12](#12-le-modèle-godunov--ramses-grille-amr) |
| `shamrock.get_Model_GSPH(...)` | SPH Godunov (solveur de Riemann) | voir [§13](#13-les-modèles-gsph-et-zeus) |
| Zeus | différences finies | voir [§13](#13-les-modèles-gsph-et-zeus) |

**Noyaux SPH** (`src/shammath/include/shammath/sphkernels.hpp`) :

| Noyau | Famille | Rayon de support `Rkern` | `hfact` par défaut |
|---|---|---|---|
| `M4` | B-spline cubique | 2 | 1.2 |
| `M6` | B-spline quintique | 3 | 1.0 |
| `M8` | B-spline septique | 4 | 1.0 |
| `C2` | Wendland C2 | 2 | 1.4 |
| `C4` | Wendland C4 | 2 | 1.6 |
| `C6` | Wendland C6 | 2 | 2.2 |

(D'autres noyaux — M5, M7, M9, M10, M4DH, M4Shift, TGauss… — existent dans `shammath` et
dans `shamrock.math.sphkernel` pour l'exploration, mais le modèle SPH n'est instancié que pour
les six ci-dessus.)

`model.get_hfact()` renvoie `hfact` ; la densité SPH s'en déduit : `rho = m * (hfact/h)**3`
(c'est `model.rho_h(...)`, C++ `shamrock::sph::rho_h`).

---

## 4. Configurer le solveur SPH (`cfg.*`)

On obtient une config par défaut avec `cfg = model.gen_default_config()`, on la modifie, puis
`model.set_solver_config(cfg)`. `model.get_current_config()` relit la config active,
`cfg.print_status()` l'affiche (JSON). La config est (dé)sérialisable en JSON
(`shammodels::common::add_json_defs`).

Toutes ces méthodes sont définies dans `src/shammodels/sph/src/pySPHModel.cpp:77-547`, et
implémentées dans `src/shammodels/sph/include/shammodels/sph/SolverConfig.hpp`.

> Notation : `kw` = arguments *keyword-only* (il faut écrire `nom=valeur`).

### 4.1 Équation d'état (EOS)

| Méthode | Arguments | Physique | Champs ajoutés |
|---|---|---|---|
| `set_eos_isothermal(cs)` | `cs` | P = cs² ρ | — |
| `set_eos_adiabatic(gamma)` | `gamma` | P = (γ−1) ρ u | `uint`, `duint` |
| `set_eos_polytropic(K, gamma)` | `K, gamma` | P = K ρ^γ | — |
| `set_eos_locally_isothermal()` | — | cs(r) lu dans le champ `soundspeed` (fixé au setup) | `soundspeed` |
| `set_eos_locally_isothermalLP07(cs0=, q=, r0=)` | kw | cs = cs0 (r/r0)^(−q), Lodato & Price 2007 | `soundspeed` |
| `set_eos_locally_isothermalFA2014(h_over_r=)` | kw | Farris+ 2014 (binaires) | `soundspeed` |
| `set_eos_locally_isothermalFA2014_extended(cs0=, q=, r0=, n_sinks=)` | kw | version généralisée à n sinks | `soundspeed` |
| `set_eos_fermi(mu_e=)` | kw | gaz de Fermi dégénéré | — |

Formules : `src/shamphys/include/shamphys/eos.hpp` (`EOS_Isothermal`, `EOS_Adiabatic`,
`EOS_Polytropic`, `EOS_LocallyIsothermal`, `EOS_Machida06`, `EOS_Fermi`).

### 4.2 Viscosité artificielle

| Méthode (toutes kw sauf `None`) | Paramètres | Usage |
|---|---|---|
| `set_artif_viscosity_None()` | — | pas de viscosité (tests) |
| `set_artif_viscosity_Constant(alpha_u, alpha_AV, beta_AV)` | α conductivité, α visqueux, β | AV constante |
| `set_artif_viscosity_VaryingMM97(alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV)` | | Morris & Monaghan 97 (α par particule → champ `alpha_AV`, `divv`) |
| `set_artif_viscosity_VaryingCD10(alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV)` | | Cullen & Dehnen 2010 (champs `alpha_AV`, `divv`, `dtdivv`, `curlv`) — **défaut recommandé pour les chocs** |
| `set_artif_viscosity_ConstantDisc(alpha_AV, alpha_u, beta_AV)` | | viscosité « disque » : `alpha_AV` ≈ α_SS × 10 (cf. `alpha_AV = 1e-3/0.08` dans `run_sph_basic_disc.py`) |

### 4.3 Conditions aux limites

| Méthode | Arguments | Remarque |
|---|---|---|
| `set_boundary_free()` | — | défaut ; les particules peuvent sortir de la boîte |
| `set_boundary_periodic()` | — | boîte périodique (dimensions données par `resize_simulation_box`) |
| `set_boundary_shearing_periodic(shear_base, shear_dir, speed)` | `i32_3, i32_3, float` | shearing box, ex. `((1,0,0),(0,1,0), q*Omega*Lx)` (`run_sph_shear_test.py:72`) |

### 4.4 MHD

| Méthode | Arguments | Champs |
|---|---|---|
| `set_IdealMHD(sigma_mhd=, alpha_B=1.0)` | kw | `B/rho`, `dB/rho`, `psi/ch`, `dpsi/ch`, `divB`, `curlB`… |
| `set_NonIdealMHD(sigma_mhd=, etaO=, etaH=, etaAD=, alpha_B=1.0)` | kw ; résistivités Ohm, Hall, ambipolaire | + `J` |

La MHD non idéale exige `shamrock.enable_experimental_features()` (`SolverConfig.hpp:693`).
Initialiser B via `model.set_field_value_lambda_f64_3("B/rho", f)` (`run_orzag_tang.py:158`),
ou `model.push_particle_mhd(...)`.

### 4.5 Poussière

| Méthode | Arguments | Physique |
|---|---|---|
| `set_dust_mode_none()` | — | défaut |
| `set_dust_mode_monofluid_tva(nvar=, pure_diffusion_mode=False, C_1_fluid=0.1, C_drift=1.0, cfl_density_threshold=eps, ensure_s_j_positivity=True, smooth_s_positivity_limiter=False, dust_corrected_av=False, clamp_dust_frac=None)` | kw ; `nvar` = nb d'espèces | monofluide « terminal velocity approximation » ; variable évoluée `s_j = sqrt(ρ ε_j)` |
| `set_dust_mode_monofluid_complete(ndust=)` | kw | monofluide complet (`epsilon`, `deltav`) |
| `set_dust_drag_constant([ts_1, ..., ts_n])` | liste de temps d'arrêt | t_s constant par espèce |
| `set_dust_drag_epstein(gamma, grain_sizes, grain_densities)` | | traînée d'Epstein |
| `set_dust_evol_coala_coag(rhodust_eps, vfrag_threshold, massgrid, tabflux_coag, tau_frac=10)` | `massgrid` de taille nbins+1, `tabflux_coag` numpy (nbins,nbins,nbins) | coagulation COALA (voir docstring `pySPHModel.cpp:430`) |
| `set_dust_ballabio_ts_limiter(enabled)` | bool | limiteur de t_s de Ballabio+ |

Tout mode de poussière autre que `none` exige `shamrock.enable_experimental_features()`
(`SolverConfig.hpp:380`).
Initialiser la poussière TVA : `model.set_field_value_lambda_f64("s_j", lambda r: sqrt(rho*eps), ivar)`
(`run_dustywave_tva.py:96`) ; injection en cours de route : `overwrite_field_value_f64("s_j", f, k)`
(`run_dustysettle_tva.py:925`).

### 4.6 Auto-gravité

| Méthode | Arguments (kw) | Méthode numérique |
|---|---|---|
| `set_self_gravity_none()` | — | défaut |
| `set_self_gravity_direct(reference_mode=False)` | | somme directe O(N²) — référence |
| `set_self_gravity_mm(order=, opening_angle=, reduction_level=3)` | | multipôles (arbre) |
| `set_self_gravity_fmm(order=, opening_angle=, reduction_level=3)` | | Fast Multipole Method |
| `set_self_gravity_sfmm(order=, opening_angle=, leaf_lowering=True, reduction_level=3)` | | FMM « symétrique » |
| `set_softening_plummer(epsilon=)` / `set_softening_none()` | | adoucissement de Plummer |

### 4.7 Forces externes (potentiels analytiques)

Toutes s'**ajoutent** à une liste (`ExtForceConfig::ext_forces`) : on peut en cumuler plusieurs.

| Méthode | Arguments | Force |
|---|---|---|
| `add_ext_force_point_mass(central_mass, Racc, central_pos=(0,0,0))` | `central_pos` kw | gravité newtonienne d'une masse ponctuelle (+ accrétion dans `Racc`) |
| `add_ext_force_paczynski_wiita(central_mass, central_pos, Racc)` | positionnels | potentiel pseudo-newtonien PW |
| `add_ext_force_lense_thirring(central_mass=, Racc=, a_spin=, dir_spin=, central_pos=)` | kw ; `dir_spin` unitaire | précession de Lense-Thirring |
| `add_ext_force_shearing_box(Omega_0=, eta=, q=)` | kw | forces de marée + Coriolis (Stone 2010) |
| `add_ext_force_vertical_disc_potential(central_mass=, R0=)` | kw | a_z = −GMz / (R0² √(R0²+z²)) |
| `add_ext_force_velocity_dissipation(eta=)` | kw | a = −η v (relaxation / amortissement) |

> `add_ext_force_velocity_dissipation` est l'exemple le plus simple d'ajout de physique : il
> sert de fil rouge à la [recette 17.A](#17a-un-terme-de-force-local-sans-voisins).

### 4.8 Unités, CFL, masse

| Méthode | Arguments | Rôle |
|---|---|---|
| `set_units(unit_system)` / `get_units()` | `shamrock.UnitSystem` | système d'unités code (G, c, μ0 en découlent : `get_constant_G()`…) |
| `set_cfl_cour(x)` | défaut 0.3 | C_cour (condition de Courant) |
| `set_cfl_force(x)` | défaut 0.25 | C_force (condition sur l'accélération) |
| `set_eta_sink(x)` | défaut 0.05 | pas de temps des sinks |
| `set_cfl_mult_stiffness(x)` | défaut 2 | raideur de la remontée du multiplicateur CFL |
| `set_show_cfl_detail(bool)` | | affiche quel critère limite dt |
| `set_particle_mass(m)` | | masse de particule (aussi faisable sur `model`) |
| `add_kill_sphere(center=, radius=)` | kw | supprime les particules **hors** de la sphère (`ParticleKillingConfig`) |

### 4.9 Longueur de lissage, arbre, performances, diagnostics

| Méthode | Rôle |
|---|---|
| `set_smoothing_length_density_based()` | itération h–ρ standard (défaut) |
| `set_smoothing_length_density_based_neigh_lim(max_neigh_count)` | idem avec plafond de voisins (défaut 500) |
| `set_smoothing_length_density_based_bisecting_nr()` | Newton-Raphson + bissection (robuste) |
| `set_tree_reduction_level(n)` | niveau de réduction de l'arbre (défaut 3) |
| `set_neigh_cache_strategy(NeighCacheStrategy.TwoStage / .SingleStage)` | stratégie de construction du cache de voisins |
| `set_enable_particle_reordering(bool)`, `set_particle_reordering_step_freq(n)` | réordonnancement Morton périodique |
| `set_scheduler_config(split_load_value=, merge_load_value=)` | critères de découpe/fusion des patches |
| `set_particle_tracking(bool)` | ajoute le champ `part_id` (suivi des particules) |
| `set_save_dt_to_fields(bool)` | ajoute `dt_part` (dt par particule) |
| `use_luminosity(bool)` | calcule `luminosity` |
| `compute_GW(bool)` | émission d'ondes gravitationnelles |
| `set_show_neigh_stats`, `set_show_ghost_stats`, `set_show_ghost_zone_graph`, `set_filter_empty_patch_gz` | diagnostics / optimisation |

### 4.10 Partir d'un dump Phantom

| Méthode | Rôle |
|---|---|
| `model.gen_config_from_phantom_dump(dump, bypass_error=False)` | construit une `cfg` équivalente à la config Phantom |
| `model.init_from_phantom_dump(dump, hpart_fact_load=1.0)` | charge les particules |
| `model.make_phantom_dump()` → objet avec `.save_dump(path)` | écrire au format Phantom |

Voir `examples/sph/run_start_sph_from_phantom_dump.py`, `run_phantom_run_emulation.py`.

---

## 5. Scheduler, boîte, masse des particules

| Méthode (`model.`) | Arguments | Rôle / quand l'appeler |
|---|---|---|
| `init_scheduler(split, merge)` | `u32, u32` | **crée le PatchScheduler et le layout des champs** à partir de la config. Un patch est découpé au-delà de `split` particules, fusionné sous `merge`. `int(1e8), 1` = quasi un seul patch (OK sur 1 GPU) |
| `init()` | — | variante sans arguments (lit `cfg.scheduler_conf`) |
| `resize_simulation_box(bmin, bmax)` | 2 tuples | boîte de simulation (= domaine périodique si BC périodiques) |
| `get_box_dim_fcc_3d(dr, nx, ny, nz)` | | dimensions d'une boîte **périodique** compatible avec un réseau HCP de nx×ny×nz particules |
| `get_box_dim_true_fcc_3d(dr, nx, ny, nz)` | `ny` pair, `nz` multiple de 3 | idem pour un vrai FCC |
| `shamrock.math.get_ideal_hcp_box(dr, bmin, bmax)` | | ajuste une boîte pour qu'elle soit périodique en HCP |
| `total_mass_to_part_mass(M)` | | M / N_total (à appeler **après** génération) |
| `set_particle_mass(m)` / `get_particle_mass()` | | masse unique des particules SPH |
| `get_total_part_count()` | | N total (tous rangs MPI) |
| `get_hfact()` / `rho_h(m, h)` | | relation h ↔ ρ |
| `get_patch_transform()` | | transformation coordonnées ↔ grille de patches |

Idiome « boîte périodique à N particules » (`run_sod.py`) :

```python
(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
dr = 1 / xs  # normalise la longueur x à 1
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)
model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
```

---

## 6. Générer les particules (`SPHSetup`)

`setup = model.get_setup()` renvoie un objet qui fabrique un **arbre de nœuds de setup** :
des *générateurs* (feuilles), des *modificateurs* (transforment un parent), des *combineurs*
(fusionnent deux parents). Rien n'est créé tant qu'on n'appelle pas `setup.apply_setup(noeud)`.
L'arbre est visualisable : `shamrock.utils.plot.DotGraph(noeud.get_dot())`.

Source : `pySPHModel.cpp:590-939`, implémentations dans
`src/shammodels/sph/include/shammodels/sph/modules/setup/`.

### 6.1 Générateurs

| Méthode | Arguments | Résultat |
|---|---|---|
| `make_generator_lattice_hcp(dr, box_min, box_max, discontinuous=True, init_h_factor=2^(5/6))` | `dr` = demi-distance entre voisins | réseau hexagonal compact (le plus utilisé) |
| `make_generator_lattice_fcc(dr, box_min, box_max, discontinuous=True, init_h_factor=…)` | nb de couches en z multiple de 3 | vrai FCC (empilement ABC) |
| `make_generator_lattice_cubic(dr, box_min, box_max)` | | réseau cubique simple |
| `make_generator_disc_mc(part_mass=, disc_mass=, r_in=, r_out=, sigma_profile=, H_profile=, rot_profile=None, cs_profile=None, velocity_field=None, cs_field=None, random_seed=, init_h_factor=0.8)` | kw ; profils = fonctions Python | disque Monte-Carlo : r tiré selon Σ(r), z gaussien d'échelle H(r). **Exactement un** de `rot_profile(r)` / `velocity_field((x,y,z))` ; si EOS localement isotherme, **exactement un** de `cs_profile(r)` / `cs_field((x,y,z))` |
| `make_generator_from_context(other_ctx)` | un autre `shamrock.Context` | recopie les particules d'une autre simulation (upscale, restart modifié : `run_init_sim_from_other.py`) |

`discontinuous=True` : le réseau est « coupé » aux bords de la boîte (utile pour juxtaposer
deux réseaux de pas différents, ex. tube de Sod).

### 6.2 Modificateurs et combineurs

| Méthode | Arguments (kw) | Effet |
|---|---|---|
| `make_combiner_add(gen1, gen2)` | positionnels | union de deux ensembles de particules |
| `make_modifier_offset(parent=, offset_position=, offset_velocity=)` | | translation en position et vitesse |
| `make_modifier_filter(parent=, filter=)` | `filter((x,y,z)) -> bool` | ne garde que les particules où `filter` est vrai (sphère, cylindre…) |
| `make_modifier_warp_disc(parent=, Rwarp=, Hwarp=, inclination=, posangle=0.)` | | gauchit un disque (warp) |
| `make_modifier_custom_warp(parent=, inc_profile=, psi_profile=, k_profile=)` | fonctions de r | warp à profils arbitraires (`run_sph_custom_warp_profile.py`) |
| `make_modifier_split_part(parent=, n_split=, seed=, h_scaling=0.6)` | | divise chaque particule en `n_split` (raffinement) |

### 6.3 `apply_setup`

```python
setup.apply_setup(
    noeud,
    part_reordering=True,
    gen_step=None,
    insert_step=None,
    msg_count_limit=None,
    rank_comm_size_limit=None,
    max_msg_size=None,
    do_setup_log=False,
    use_new_setup=True,
    speculative_balancing=False,
)
```

Génère les particules par paquets et les distribue sur les patches/rangs MPI.
`insert_step` (nb de particules insérées par paquet) est souvent mis à la valeur de split du
scheduler pour les gros setups (`run_dustywave_tva.py:91`). Les autres options règlent la
communication MPI pendant le setup (laisser par défaut au début).

### 6.4 Anciennes méthodes (toujours exposées)

`add_cube_fcc_3d(dr, bmin, bmax)`, `add_cube_hcp_3d(...)`, `add_cube_hcp_3d_v2(...)`,
`add_disc_3d(center, central_mass, Npart, r_in, r_out, disc_mass, p, H_r_in, q)`,
`add_disc_3d_keplerian(...)`, `add_big_disc_3d(..., seed)`,
`push_particle(pos_list, h_list, u_list)`, `push_particle_mhd(pos, h, u, B_on_rho, psi_on_ch)`.
`push_particle` est l'échappatoire universelle : on calcule les positions en numpy et on les
pousse directement.

---

## 7. Initialiser / modifier des champs

Les noms de champs sont ceux du layout ([annexe A](#annexe-a--champs-sph-disponibles)).
`field_type` vaut `"f64"` (scalaire) ou `"f64_3"` (vecteur). Pour un champ multi-variables
(`nvar>1`, ex. `s_j` avec plusieurs espèces), `ivar`/`offset` choisit la composante.

| Méthode (`model.`) | Signature | Usage |
|---|---|---|
| `set_value_in_a_box(field_name, field_type, value, box_min, box_max, ivar=0)` | `ivar` kw | valeur constante dans un pavé (états gauche/droite d'un Riemann) |
| `set_value_in_sphere(field_name, field_type, value, center, radius)` | | valeur constante dans une sphère (Sedov…) |
| `set_field_value_lambda_f64(field_name, pos_to_val, offset=0)` | `pos_to_val((x,y,z)) -> float` | champ scalaire analytique f(r) |
| `set_field_value_lambda_f64_3(field_name, pos_to_val, offset=0)` | `pos_to_val((x,y,z)) -> (vx,vy,vz)` | champ vectoriel analytique (vitesse, B/ρ) |
| `overwrite_field_value_f64(field_name, compute, offset)` | `compute(patchdata) -> list/array` de taille N_patch | **le plus général** : `patchdata["xyz"]`, `patchdata["hpart"]`… donnent accès à tous les champs du patch (numpy) ; utile quand la valeur dépend de ρ ou d'autres champs |
| `overwrite_field_value_f64_3(...)` | idem vectoriel | |
| `add_kernel_value(field_name, field_type, value, center, h_ker)` | | ajoute `value × W(r−center, h_ker)` (dépôt lissé d'énergie, ex. Sedov) |
| `remap_positions(f)` | `f((x,y,z)) -> (x',y',z')` | déforme les positions (étirement, perturbation) |
| `apply_position_offset(offset)` / `apply_momentum_offset(offset)` | | translation globale (ex. se placer dans le référentiel du barycentre) |
| `add_sink(mass, pos, velocity, accretion_radius)` | | ajoute une particule sink (après `init_scheduler`) |
| `get_sinks()` | → liste de dicts `pos, velocity, sph_acceleration, ext_acceleration, mass, angular_momentum, accretion_radius` | |
| `get_sum(field_name, field_type)` | | somme globale d'un champ (diagnostic) |
| `get_closest_part_to(pos)` | | position de la particule la plus proche |

Exemple (Kelvin-Helmholtz, `run_kelvin_helmholtz.py:147`) :

```python
def vel_func(r):
    x, y, z = r
    vx = 0.5 if abs(y) < 0.25 else -0.5
    return (vx, 0.01 * np.sin(4 * np.pi * x), 0.0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)
```

Exemple « dépend de la densité » (`run_dustysettle_tva.py:226`) :

```python
def s_init(patchdata):
    rho = model.get_particle_mass() * (model.get_hfact() / np.array(patchdata["hpart"])) ** 3
    return np.sqrt(rho * eps0)


model.overwrite_field_value_f64("s_j", s_init, 0)
```

> ⚠️ `hpart` (donc ρ) n'est cohérent qu'après un premier `model.timestep()`. Si votre
> initialisation dépend de ρ, faites un `timestep()` d'abord (ou `set_dt(0.0)` puis
> `timestep()` après la modification, cf. `run_dustysettle_tva.py:930`).

---

## 8. Faire avancer la simulation

| Méthode (`model.`) | Signature | Rôle |
|---|---|---|
| `timestep()` | → `TimestepLog` | un pas complet (utilisé au début pour converger h et initialiser les dérivées) |
| `evolve_once()` | | un pas de temps |
| `evolve_until(target_time, niter_max=-1, max_walltime=-1)` | kw sauf 1er | avance jusqu'à `target_time` (dernier pas raccourci pour tomber pile) ; renvoie `EvolveUntilResults` (`reach_target_time`, `reach_niter_max`, `reach_max_walltime`, `iter_count`) |
| `evolve_once_override_time(t_curr, dt_input)` | | un pas imposé (tests) |
| `get_time()` / `set_time(t)` | | temps courant |
| `get_dt()` / `set_next_dt(dt)` / `set_dt(dt)` | | pas de temps |
| `set_cfl_cour`, `set_cfl_force`, `set_eta_sink` | | idem config, modifiables en cours de route |
| `set_cfl_multipler(lambda)`, `set_cfl_mult_stiffness(cstiff)` | | multiplicateur CFL (redémarrage doux) |
| `change_htolerances(coarse=, fine=)` | kw | tolérance d'évolution de h par itération (défaut 1.1/1.1). Mettre `coarse=1.3` au 1er pas accélère la convergence (`run_sph_basic_disc.py`). `change_htolerance(x)` est **déprécié** |
| `add_timestep_callback(step_begin=None, step_end=None)` | kw ; fonctions sans argument | code Python exécuté à chaque début/fin de pas |
| `get_solver_dot_graph()` / `get_solver_tex()` | | graphe/équations du solveur (pédagogique !) |

Boucle typique :

```python
for i, t in enumerate(np.linspace(0, t_end, ndump)):
    if model.get_time() > t:  # utile en cas de reprise
        continue
    model.evolve_until(t)
    model.dump(f"dump_{i:04}.sham")
```

---

## 9. Entrées/sorties et reprise

| Méthode | Rôle |
|---|---|
| `model.dump(fname)` | dump de reprise Shamrock (`.sham`) : config + particules + sinks + temps |
| `model.load_from_dump(fname)` | recharge un `.sham` (remplace config + `init_scheduler` + setup) |
| `model.do_vtk_dump(fname, add_patch_world_id)` | sortie VTK (ParaView) |
| `model.make_phantom_dump().save_dump(fname)` | sortie Phantom |
| `model.set_debug_dump(bool, fname)` | dumps de debug |
| `model.solver_logs_*` (`last_rate`, `last_obj_count`, `step_count`, `cumulated_step_time`, `reset_*`, `last_system_metrics`) | performances |

Reprise automatique et rotation des checkpoints : voir `SimulationRunner.do_checkpoint` ([§11](#11-orchestrer-une-simulation--simulationrunner--co)).

---

## 10. Récupérer, analyser et visualiser les données

### 10.1 Données brutes

| Fonction | Rôle |
|---|---|
| `ctx.collect_data()` | dict `{nom_champ: numpy array}` de **toutes** les particules, rassemblé sur le rang 0 (petites simus uniquement) |
| `model.compute_field(name, field_type, custom_getter=None)` | champ distribué (objet `Field`) : `name` = un champ du layout, ou `"rho"`, `"inv_hpart"`, `"unity"`, ou `"custom"` avec `custom_getter(size, patchdata) -> np.array` |
| `shamrock.map_fields_f64(fonction, **champs)` | combine des `Field` (ex. ε = s²/ρ, `run_dustywave_tva.py:137`) |

### 10.2 Rendus (interpolation SPH sur grille/rayons)

Chaque méthode existe en trois variantes : par nom (`name, field_type, ..., custom_getter=None`),
ou en passant directement un `Field` f64 / f64_3 issu de `compute_field`.

| Méthode | Arguments | Sortie |
|---|---|---|
| `render_cartesian_slice(name, field_type, center, delta_x, delta_y, nx, ny)` | plan passant par `center`, engendré par `delta_x`, `delta_y` | image ny×nx (coupe) |
| `render_cartesian_column_integ(...)` | idem | densité de colonne ∫ f dz |
| `render_slice(name, field_type, positions)` | liste de points | valeurs interpolées |
| `render_column_integ(name, field_type, rays)` | liste de `shammath.Ray` | intégrales le long de rayons |
| `render_azymuthal_integ(name, field_type, ring_rays)` | `RingRay` | moyennes azimutales |

### 10.3 Analyses toutes faites

| Fonction | Retour |
|---|---|
| `model.make_analysis_sodtube(sod, direction, time_val, x_ref, x_min, x_max).compute_L2_dist()` | erreurs L2 (ρ, v, P) vs solution exacte `shamrock.phys.SodTube` |
| `model.make_analysis_disc().collect_data(Rmin, Rmax, Nbin, ctx)` | dict `radius, counter, Sigma, lx, ly, lz, tilt, twist, psi, Hsq` |
| `shamrock.model_sph.analysisBarycenter(model=m).get_barycenter()` | (barycentre, masse) |
| `shamrock.model_sph.analysisEnergyKinetic(model=m).get_kinetic_energy()` | |
| `shamrock.model_sph.analysisEnergyPotential(model=m).get_potential_energy()` | |
| `shamrock.model_sph.analysisTotalMomentum(model=m).get_total_momentum()` | |
| `shamrock.model_sph.analysisAngularMomentum(model=m).get_angular_momentum()` | |
| `shamrock.model_sph.analysisDustMass(model=m).get_dust_mass()` | |

---

## 11. Orchestrer une simulation : `SimulationRunner` & co.

### 11.1 `SimulationRunner` (`src/pylib/shamrock/utils/SimulationRunner.py`)

Une classe déclarative qui gère : setup **ou** reprise automatique depuis le dernier
checkpoint, boucle d'évolution, déclenchement de callbacks (analyse, dumps).

```python
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup


class Simulation(SimulationRunner):
    t_end = 0.245  # obligatoire
    dump_prefix = "out/dump_"  # optionnel : active checkpoint/reprise

    @callback(at_tsim=list(np.linspace(0, 0.245, 10)))
    def analysis(self, ianalysis): ...  # self.model, self.model.get_time()

    @callback(walltime_interval=30.0)
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):  # config + init_scheduler + génération + champs
        ...


Simulation(model).run()
```

| Élément | Rôle |
|---|---|
| `@callback(*, tsim_interval=None, iter_count_interval=None, walltime_interval=None, at_tsim=None)` | déclenche la méthode `f(self, i)` selon le temps simulé, le nombre d'itérations, le temps mural, ou à des dates précises (au moins un critère) |
| `@simulation_setup` | méthode de setup (appelée seulement s'il n'y a pas de dump à reprendre) |
| `run()` | reprise ou setup, puis `evolve_until` jusqu'au prochain callback… jusqu'à `t_end` |
| `do_checkpoint(i, **kwargs)` | dump + métadonnées JSON (état des callbacks) |
| `rank_0_print(*a)` | print sur le rang 0 seulement |

### 11.2 Gestion des dumps (`shamrock.utils.dump`)

`ShamrockDumpHandleHelper(model, dump_prefix, ext=".sham", metadata=False)` :
`get_dump_name(i)`, `get_last_dump()`, `load_dump(i)`,
`write_dump(i, metadata=None, purge_old_dumps=False, keep_first=1, keep_last=3)`,
`purge_old_dumps(...)`, **`load_last_dump_or(fonction_setup)`** (reprend si possible, sinon
exécute le setup). Utilisé dans `run_circular_disc_central_pot.py:205`.

### 11.3 Analyse et figures (`shamrock.utils.analysis`, `shamrock.utils.plot`)

| Outil | Rôle |
|---|---|
| `AnalysisHelper(analysis_folder, analysis_prefix)` | `analysis_save(i, data)`, `load_analysis(i)`, `get_list_analysis_id()` : sauvegarde de séries temporelles (utilisé dans `run_sod.py`) |
| `ColumnDensityPlot(model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix)`, `SliceDensityPlot`, `SliceVzPlot`, `ColumnAverageVzPlot`, `SliceDiffVthetaProfile`, `VerticalShearGradient`, `SliceByPlot`, `SliceBthetaPlot`, `SliceDtPart`, `ColumnParticleCount`… | rendus standards prêts à l'emploi : `analysis_save(i)`, `make_plot(i, ...)`, `render_all()`, `render_gif(...)` |
| `PerfHistory(model, folder, prefix)` | historique des performances |
| `shamrock.utils.plot.show_image_sequence(glob, render_gif=True, ...)` | animation à partir d'images |
| `shamrock.utils.plot.DotGraph(dot)` | affiche un graphe DOT (`gen.get_dot()`, `model.get_solver_dot_graph()`) |
| `shamrock.matplotlib.set_shamrock_mpl_style()` | style des figures de la doc |

### 11.4 Aides physiques de haut niveau

| Outil | Rôle |
|---|---|
| `shamrock.utils.disc_setup.StandardDisc(units, center_mass, disc_mass, rin, rout, H_r_0=0.05, q=0.5, p=1.5, r0=1.0, H_factor=1.0, rotation="subkeplerian", inner_tapering=False, outer_tapering=False)` | disque LP07 clé en main : `get_profiles()`, `cs0()`, `part_mass(npart)`, `make_generator(setup, npart, random_seed=666)` (`run_circular_disc_sink.py:94`) |
| `shamrock.utils.DustMRNDistribution.DustMRNDistribution(codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges)` | distribution de grains MRN multi-espèces (`run_dustysettle_tva.py:159`) |
| `shamrock.utils.numba_helper.maybe_njit(fn)` | accélère les fonctions de profil si numba est installé |

---

## 12. Le modèle Godunov / Ramses (grille AMR)

Bindings : `src/shammodels/ramses/src/pyRamsesModel.cpp` ; config :
`src/shammodels/ramses/include/shammodels/ramses/SolverConfig.hpp`.

### 12.1 Différences conceptuelles avec SPH

- Les objets sont des **blocs** de 2×2×2 = 8 cellules, repérés par des coordonnées **entières**
  (`cell_min`, `cell_max`, type `i64_3`) ; la position physique = coordonnée entière ×
  `scale_factor`.
- Les variables sont **conservatives** : `rho`, `rhovel` (f64_3), `rhoetot` (énergie totale
  volumique) ; + `rho_dust`, `rhovel_dust` (nvar = ndust) avec poussière ; `phi`, `phi_old` avec
  auto-gravité ; `rho_gas_pscal` avec scalaires passifs.
- Les lambdas d'initialisation reçoivent **les deux coins de la cellule** `(rmin, rmax)`, pas un
  point.

### 12.2 Squelette (d'après `examples/ramses/run_rayleigh_taylor.py:83-152`)

```python
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

sz, base = 2, 16  # taille de bloc (coord. entières, ≥ 2) et nb de blocs
cfg = model.gen_default_config()
cfg.set_scale_factor(L / (sz * base))
cfg.set_eos_gamma(gamma)
cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.set_Csafe(0.8)
cfg.add_ext_force_constant_gravity(g=(0.0, -g, 0.0))
cfg.set_boundary_condition("x", "periodic")
cfg.set_boundary_condition("y", "reflective")
model.set_solver_config(cfg)  # lève une exception si appelé après init_scheduler

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))


def rho_map(rmin, rmax):
    x, y, z = rmin
    return 2.0 if y > 0.5 else 1.0


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", lambda rmin, rmax: P0 / (gamma - 1))
model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (0.0, 0.0, 0.0))

for t in np.linspace(0, t_end, n_frames):
    model.evolve_until(t)
    rho = model.render_slice("rho", "f64", positions)
```

### 12.3 Configuration (`cfg.*`)

| Thème | Méthodes | Défaut / remarques |
|---|---|---|
| Grille | `set_scale_factor(f)` | 1 |
| CFL | `set_Csafe(c)` | 0.9 |
| Prédiction | `set_face_time_interpolation(bool)` | True (type MUSCL-Hancock) |
| Riemann | `set_riemann_solver_rusanov()`, `_hll()`, `_hllc()` | HLL |
| Limiteur | `set_slope_lim_none()`, `_vanleer_f()`, `_vanleer_std()`, `_vanleer_sym()`, `_minmod()` | `VanLeer_sym` |
| EOS | `set_eos_gamma(g)` (= `set_eos_adiabatic(g)`), `set_eos_reset_barotropic(rho_crit=, cs0=, gamma_baro=, gamma=)` | γ = 5/3. `set_eos_barotropic` existe mais est **refusée** par `check_config` |
| Bords | `set_boundary_condition(axis, bc_type)` avec `axis ∈ {"x","y","z"}`, `bc_type ∈ {"periodic","reflective","outflow"}` | périodique partout |
| Forces | `add_ext_force_constant_gravity(g=(gx,gy,gz))` (cumulable) ; `set_ext_force_coupling(shamrock.model_ramses.ExtForceCoupling.Midpoint / .Trapezoidal)` | Midpoint |
| Poussière (exp.) | `set_dust_mode_dhll(ndust)`, `set_dust_mode_hb(ndust)`, `set_dust_mode_none()` | |
| Traînée | `set_drag_mode_no_drag()`, `set_drag_mode_irk1(frictional_heating)`, `set_drag_mode_expo(bool)`, `set_alpha_values(alpha)` (**ajoute** un α = 1/t_s par espèce : l'appeler une fois par espèce) | `irk2` exposé mais non implémenté |
| Auto-gravité (exp.) | `set_gravity_mode_cg()`, `_bicgstab()`, `_no_gravity()`, `set_self_gravity_Niter_max(n)`, `set_self_gravity_tol(tol)`, `set_self_gravity_happy_breakdown_tol(tol)` ; requiert `ExtForceCoupling.Trapezoidal`, pas de poussière ni de traînée, et `set_units(...)` pour G | `pcg` exposé mais refusé |
| AMR | `set_amr_mode_none()`, `set_amr_mode_density_based(crit_mass=)`, `set_amr_mode_pseudo_gradient_based(error_min=, error_max=)`, `set_amr_mode_jeans_length_based(N_jeans=, T_init=)`, `set_amr_mode_shear_based(Threshold=)` (T majuscule !), `set_amr_mode_old(bool)` (False = nouvelle AMR, exp.), `set_first/second_order_interpolation_mode()` | |
| Scalaires passifs (exp.) | `set_npscal_gas(n)` | |
| Divers | `set_units(u)`, `set_scheduler_config(split_load_value=, merge_load_value=)`, `to_json()`, `from_json(d)` | |

### 12.4 Méthodes du modèle

| Méthode | Rôle |
|---|---|
| `init_scheduler(split, merge)` / `init()` | layout + scheduler + **construction du graphe de solveur** |
| `make_base_grid(bmin, cell_size, cell_count)` | grille de base : origine entière, taille d'un bloc (≥ 2 par axe), nombre de blocs par axe |
| `set_field_value_lambda_f64(name, f(rmin, rmax), offset=0)` / `_f64_3` | conditions initiales (offset = espèce de poussière) |
| `evolve_once()`, `timestep()`, `evolve_until(t, niter_max=-1)`, `evolve_once_override_time(t, dt)` | avance (le 1er pas a dt = 0 : il ne calcule que la CFL) |
| `get_time()`, `get_dt()`, `set_time(t)`, `set_next_dt(dt)` | |
| `add_timestep_callback(step_begin=, step_end=)` | |
| `dump(f)`, `load_from_dump(f)`, `dump_vtk(f)`, `ctx.collect_data()` | E/S (⚠ `load_from_dump` ne reconstruit pas le graphe : vérifier la reprise avant de s'y fier) |
| `render_slice(name, field_type, positions)` | valeurs aux points |
| `get_cell_coords((bmin, bmax), lid)` | coins physiques de la cellule `lid ∈ [0,8)` d'un bloc |
| `make_analysis_sodtube(...).compute_L2_dist()` | validation Sod |
| `get_solver_dot_graph()`, `get_solver_tex()` | graphe du solveur (ici **tout** le pas est en graphe : très instructif) |

Variantes à lire : `run_jeans_instability_timescales.py` (auto-gravité, unités G = 1),
`run_sod_amr_reflective.py` (nouvelle AMR), `run_kh.py`, `run_toro_shocks.py`,
`examples/TO_MIGRATE/ramses/godunov_dusty_wave_test.py` (poussière + traînée).

---

## 13. Les modèles GSPH et Zeus

### 13.1 GSPH (`src/shammodels/gsph/src/pyGSPHModel.cpp`)

SPH « Godunov » : la force entre deux particules vient d'un solveur de Riemann à l'interface
(Inutsuka 2002, Cha & Whitworth 2003). Moins de viscosité artificielle, bon pour les chocs.

```python
model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
cfg = model.gen_default_config()
cfg.set_riemann_exact()  # ou set_riemann_iterative(), set_riemann_hllc()
cfg.set_force_inutsuka_v2()  # ou set_force_cha_whitworth() (défaut)
cfg.set_reconstruct_piecewise_constant()
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)
model.resize_simulation_box(bmin, bmax)
model.add_cube_hcp_3d(dr, bmin_gauche, bmax_gauche)
model.set_field_in_box("uint", "f64", u_L, bmin_gauche, bmax_gauche)
model.set_particle_mass(model.total_mass_to_part_mass(totmass))
model.evolve_until(0.245)
```

| Config | Rôle |
|---|---|
| `set_riemann_iterative(tolerance=1e-6, max_iter=20)`, `set_riemann_exact(tolerance=1e-8, max_iter=100)`, `set_riemann_hllc()` | solveur de Riemann |
| `set_reconstruct_piecewise_constant()` | reconstruction ordre 1 |
| `set_force_cha_whitworth()`, `set_force_inutsuka_v2()` | formulation de la force |
| `set_eos_adiabatic(gamma)`, `set_eos_isothermal(cs)` | EOS |
| `set_boundary_free()`, `set_boundary_periodic()` | bords |
| `add_ext_force_point_mass(central_mass, Racc, central_pos=)` | potentiel central |
| `set_units`, `set_cfl_cour`, `set_cfl_force`, `set_particle_mass`, `set_scheduler_config`, `set_tree_reduction_level`, `set_neigh_cache_strategy` | comme SPH |

Modèle : mêmes méthodes de base que SPH (`init_scheduler`, `evolve_once/until`, `timestep`,
`get_box_dim_fcc_3d`, `add_cube_fcc_3d/hcp_3d`, `total_mass_to_part_mass`, `dump`,
`load_from_dump`, `do_vtk_dump`…), mais l'initialisation des champs s'appelle
`set_field_in_box(name, type, value, bmin, bmax, ivar=0)`, `set_field_in_sphere(...)`,
`apply_field_from_position_f64(name, f)` / `_f64_3`.
Exemples : `examples/gsph/run_sod.py`, `run_blast_wave.py`, `run_123_problem.py`.

### 13.2 Zeus (`src/shammodels/zeus/src/pyAMRZeusModel.cpp`)

Schéma à grille décalée type ZEUS. Même logique de grille que Ramses.

```python
model = shamrock.get_Model_Zeus(context=ctx, vector_type="f64_3", grid_repr="i64_3")
cfg = model.gen_default_config()
cfg.set_scale_factor(f)
cfg.set_eos_gamma(1.4)
cfg.set_consistent_transport(True)
cfg.set_van_leer(True)  # + set_Csafe(c)
model.set_solver_config(cfg)
model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (nx, ny, nz))
model.set_field_value_lambda_f64("rho", lambda rmin, rmax: ...)
model.set_field_value_lambda_f64("eint", lambda rmin, rmax: ...)
model.set_field_value_lambda_f64_3("vel", lambda rmin, rmax: (0, 0, 0))
t = 0
for i in range(N):
    t = model.evolve_once(t, dt)  # dt imposé par l'utilisateur
```

Champs : `rho`, `eint`, `vel`. Autres méthodes : `dump_vtk`, `get_cell_coords`,
`make_analysis_sodtube`. Exemple : `examples/tests_ci/sod_tube_zeus.py`.

---

# Partie II — Ajouter un terme physique (C++)

## 14. Architecture interne d'un modèle

Pour le modèle SPH (`src/shammodels/sph/`) :

| Fichier | Contenu | Vous le touchez quand… |
|---|---|---|
| `include/.../SolverConfig.hpp` + `src/SolverConfig.cpp` | structure de config, `set_layout` (champs des particules), `set_ghost_layout` (champs communiqués aux voisins), sérialisation JSON | vous ajoutez une option ou un champ |
| `src/pySPHModel.cpp` | bindings Python de la config et du modèle | vous exposez une option en Python |
| `include/.../Model.hpp`, `src/Model.cpp` | façade : setup, accès champs, dumps | rarement |
| `include/.../Solver.hpp`, `src/Solver.cpp` | `init_solver_graph()` (construction du graphe), `evolve_once()` (le pas de temps), `update_derivs()` | vous insérez une étape dans le pas |
| `include/.../modules/SolverStorage.hpp` | champs temporaires partagés entre étapes (`pressure`, `soundspeed`, `omega`, `neigh_cache`, `old_axyz`…) | vous avez besoin d'un champ intermédiaire |
| `include/.../modules/*.hpp`, `src/modules/*.cpp` | un module / nœud par opération | vous écrivez la physique |
| `include/.../math/*.hpp` | formules SPH élémentaires (`forces.hpp`, `density.hpp`, `q_ab.hpp`, `mhd.hpp`) | vous écrivez une interaction de paire |
| `src/shammodels/common/` | ce qui est partagé entre modèles : `EOSConfig.hpp`, `ExtForceConfig.hpp`, nœuds `AddForce*`, `ForwardEuler` | vous écrivez une force/EOS réutilisable |
| `src/shamphys/` | physique « pure » sans dépendance aux modèles (EOS, solutions analytiques, orbites) | vous ajoutez une formule réutilisable |
| `CMakeLists.txt` du modèle | liste des `.cpp` | vous ajoutez un `.cpp` |

### Deux styles de modules coexistent

1. **Module « historique »** : une classe `ModuleX<Tvec, SPHKernel>` construite avec
   `(ShamrockCtx &context, Config &solver_config, Storage &storage)`, avec une méthode
   (`compute_eos()`, `update_derivs()`, `add_ext_forces()`…) qui va chercher ses champs dans le
   scheduler par leur nom. Exemple : `modules/ComputeEos.hpp`, `modules/UpdateDerivs.hpp`.
   Simple, mais le module peut lire/écrire n'importe quoi.
2. **Nœud de solver graph** (direction actuelle du code) : une classe qui hérite de
   `shamrock::solvergraph::INode`, déclare explicitement ses entrées (RO) et sorties (RW), et
   ne touche **que** cela. Exemples : `common/modules/AddForceVelocityDissipation.hpp`,
   `sph/modules/NodeComputePressureGrad.hpp`, `common/modules/ForwardEuler.hpp`.
   Doc : `doc/sphinx/source/dev_doc/solvergraph*.md`.

**Pour du code neuf, écrivez un nœud** ; branchez-le si besoin depuis un module historique
(c'est exactement ce que fait `ExternalForces.cpp`).

---

## 15. Anatomie d'un pas de temps SPH

`Solver::evolve_once()` (`src/shammodels/sph/src/Solver.cpp:2731`) — intégrateur
**leapfrog KDK prédicteur-correcteur** (à la Phantom) :

```text
 0. callbacks step_begin (Python)
 1. équilibrage de charge : ComputeLoadBalanceValue + scheduler_step
 2. storage.solver_sequence.evaluate()   ← graphe "time_step" (init_solver_graph, l.1185) :
      sink accretion → point mass accretion → sink predictor → dt_to_half_dt
      → set_gpart_mass → attach fields to scheduler
      → leapfrog predictor  [ v += dt/2·a ; u += dt/2·du ; (B/ρ, ψ, ε, s_j…) ; x += dt·v ; v,u += dt/2·… ]
      → part killing step (si kill spheres) → sink ext force
 3. ExternalForces::compute_ext_forces_indep_v()   → axyz_ext (forces indépendantes de v)
 4. conditions aux limites sur positions, réordonnancement éventuel
 5. sph_prestep : arbres, ghosts (zones fantômes inter-patches/périodiques),
    itération h–ρ (omega), cache de voisins
 6. self gravity sequence (si activée)
 7. BOUCLE CORRECTEUR (do … while need_rerun_corrector, max 50) :
      a. communicate_merge_ghosts_fields   (copie des champs du ghost layout chez les voisins)
      b. diff. operators : divv, curlv, dtdivv ; update_artificial_viscosity ; J (NIMHD)
      c. compute_eos_fields                → storage.pressure, storage.soundspeed
      d. prepare_corrector                 (sauvegarde old_axyz, old_duint, …)
      e. update_derivs(dt)  :
           - nœuds dust TVA (gradient de P, Δv)…
           - UpdateDerivs::update_derivs()  → axyz, duint (pression + AV) ou MHD/dust
           - ExternalForces::add_ext_forces() → axyz += axyz_ext + forces dépendant de v
      f. GW, luminosité, ConservativeCheck
      g. correcteur leapfrog : v += dt/2 (a_new − a_old), idem u, B/ρ, ψ, ε, Δv, s_j
      h. critère de convergence ; calcul du prochain dt (CFL courant, force, sinks, dust, MHD…)
 8. nettoyage (ghosts, arbres, caches), set_next_dt, set_time(t+dt)
 9. callbacks step_end (Python)
```

Conséquences pratiques :

- Un terme qui **ne dépend que des positions** (potentiel) → étape 3 (`axyz_ext`, calculé une
  fois par pas).
- Un terme qui **dépend de v, u, ou de champs recalculés** → étape 7e (dans la boucle correcteur).
- Une **nouvelle variable évoluée** → il faut l'insérer dans le prédicteur (2), le
  `prepare_corrector` (7d) et le correcteur (7g), plus les ghosts si les voisins en ont besoin.
- Un nouveau **critère de pas de temps** → étape 7h (modules `ComputeCFL*`).

Astuce : `print(model.get_solver_dot_graph())` affiche la partie du pas déjà migrée en graphe.

---

## 16. Briques C++ de base

### 16.1 Types (`dev_doc/types.md`)

| Type | Sens |
|---|---|
| `u8, u16, u32, u64`, `i8…i64`, `f16, f32, f64` | entiers/flottants de taille explicite |
| `f64_3`, `f32_3`, `i32_3`… | vecteurs SYCL (`.x()`, `[i]`, `sycl::dot`, `sycl::length`, `sycl::cross`) |
| `Tvec` | paramètre template du modèle (= `f64_3`) |
| `Tscal = shambase::VecComponent<Tvec>` | scalaire associé (= `f64`) |
| `SPHKernel<Tscal>` / `Kernel` | noyau SPH : `Kernel::W_3d(r,h)`, `Kernel::dW_3d(r,h)`, `Kernel::Rkern`, `Kernel::hfactd` |
| littéraux `15_u64`, `1.0_f64` | |

### 16.2 Données distribuées

| Classe | Rôle |
|---|---|
| `PatchScheduler` (`ctx.sched`, `scheduler()`) | possède les patches ; `pdl_old()` = layout ; `for_each_patchdata_nonempty([&](Patch p, PatchDataLayer &pdat){…})` ; `get_total_obj_count()` |
| `PatchDataLayerLayout` | liste (nom, type, nvar) des champs : `add_field<T>(name, nvar)`, `get_field_idx<T>(name)` |
| `PatchDataLayer` | les données d'un patch : `get_field<T>(idx)`, `get_field_buf_ref<T>(idx)`, `get_obj_cnt()` |
| `PatchDataField<T>` | un champ d'un patch : `get_buf()`, `get_nvar()`, `field_raz()` (mise à zéro) |
| `sham::DeviceBuffer<T>` | buffer GPU : `get_read_access(deps)` / `get_write_access(deps)` / `complete_event_state(e)`, `copy_to_stdvec()`, `copy_from_stdvec()` |
| `shambase::DistributedData<T>` | `map<patch_id, T>` : `get(id)`, `add_obj(id, v)`, `for_each`, **`for_each_ordered`** (déterministe), `map<U>(f)` |
| `shamrock::ComputeField<T>` / `SchedulerUtility::make_compute_field<T>(name, nvar)` | champ temporaire de même taille que les patches |
| `SchedulerUtility::fields_leapfrog_corrector<T>(ifield, idfield, old_d, eps_sq, dt/2)` | correcteur leapfrog générique |

### 16.3 Lancer un noyau GPU

```cpp
#include "shambackends/kernel_call.hpp"        // un patch
sham::kernel_call(q,
    sham::MultiRef{buf_in1, buf_in2},   // entrées  → const T*
    sham::MultiRef{buf_out},            // sorties  → T*
    n,                                  // nombre de threads
    [=](u32 i, const Tvec *in1, const Tscal *in2, Tvec *out) { out[i] = …; });

#include "shambackends/kernel_call_distrib.hpp" // tous les patches d'un coup
sham::distributed_data_kernel_call(
    shamsys::instance::get_compute_scheduler_ptr(),
    sham::DDMultiRef{edges.a.get_spans(), edges.b.get_spans()},
    sham::DDMultiRef{edges.out.get_spans()},
    edges.sizes.indexes,                // DistributedData<u32> : nb de threads par patch
    functor);
```

Dépendances et synchronisations sont gérées automatiquement (pas de `wait()` à la main).
La file : `shamsys::instance::get_compute_scheduler().get_queue()`.
Le style « historique » (`q.submit(deps, [&](sycl::handler &cgh){ shambase::parallel_for(cgh, n, "nom", [=](u64 gid){…}); })`
+ `get_read_access`/`complete_event_state`) est visible dans `UpdateDerivs.cpp:96`.

### 16.4 Solver graph : nœuds et arêtes

```cpp
#define NODE_EDGES(X_RO, X_RW)                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, eta)           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_velocity)\
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)             \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

class MonNoeud : public shamrock::solvergraph::INode {
  public:
    EXPAND_NODE_EDGES(NODE_EDGES)        // génère set_edges(...) et get_edges()
    void _impl_evaluate_internal() override { auto edges = get_edges(); … }
    std::string _impl_get_label() const override { return "MonNoeud"; }
    std::string _impl_get_tex() const override { return "…"; }
};
#undef NODE_EDGES
```

Arêtes (edges) les plus utiles :

| Arête | Contenu | Méthodes clés |
|---|---|---|
| `IDataEdge<T>` | une valeur (`.data`) | `make_shared(label, tex)` |
| `Indexes<u32>` | nb d'objets par patch (`.indexes`) | |
| `IFieldSpan<T>` | interface « vue sur un champ distribué » | `get_spans()`, `check_sizes(sizes)`, `ensure_sizes(sizes)` |
| `FieldRefs<T>` | références vers des champs **existants** du scheduler | `set_refs(DDPatchDataFieldRef)` |
| `Field<T>` | champ **possédé** (temporaire) | `Field<T>(nvar, label, tex)`, `get_field(id)` |
| `shammodels::sph::solvergraph::NeighCache` | listes de voisins par patch | passé tel quel à un noyau → `tree::ObjectCacheIterator` |
| `ScalarEdge`, `ScalarsEdge`, `DDSharedScalar`… | divers | |

Nœuds utilitaires : `NodeSetEdge<E>(lambda)` (remplit une arête), `OperationSequence(nom, {nœuds})`,
`OperationIf(nom, then, else)` + arête `IDataEdge<bool>`, `NodeFreeAlloc`, `GetFieldRefFromLayer<T>(pdl, "nom")`,
`ForwardEuler<T>` (u += dt·du), `NodeAXPY`, `NodeSumReduction`, `ExchangeGhostField<T>`…

Graphe nommé : `solver_graph.register_edge("nom", Edge(...))`, `register_node("nom", Node{})`,
`get_edge_ptr<E>("nom")`, `get_node_ptr_base("nom")`, `get_node_ref_base("nom").evaluate()`.

### 16.5 Boucle sur les voisins SPH

Dans un noyau (`NodeComputePressureGrad.cpp`) :

```cpp
shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs); // ploop_ptrs = neigh cache
particle_looper.for_each_object(id_a, [&](u32 id_b) {
    Tvec dr    = xyz_a - xyz[id_b];
    Tscal rab2 = sycl::dot(dr, dr);
    Tscal h_b  = hpart[id_b];
    if (rab2 > h_a*h_a*Rker2 && rab2 > h_b*h_b*Rker2) return;   // hors support
    Tscal rab  = sycl::sqrt(rab2);
    Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);
    Tscal Fab_a = Kernel::dW_3d(rab, h_a);
    …
});
```

Les champs lus pour `id_b` doivent contenir **les particules fantômes** (taille
`part_counts_with_ghost`) ; les sorties ont la taille `part_counts`.

Formules SPH réutilisables (`src/shammodels/sph/include/shammodels/sph/math/`) :

| Fonction | Fichier | Rôle |
|---|---|---|
| `rho_h(m, h, hfact)`, `h_rho(m, rho, hfact)` | `density.hpp` | relation h–ρ |
| `sph_pressure_symetric(m, ρa², ρb², Pa, Pb, Ωa, Ωb, ∇Wa, ∇Wb)` | `forces.hpp:60` | force de pression symétrique |
| `sph_pressure_symetric_av(...)` | `forces.hpp:100` | idem + viscosité artificielle |
| `duint_dt_pressure(...)` | `forces.hpp:138` | du/dt de compression |
| `vsig_hydro`, `vsig_u`, `lambda_shock_conductivity` | `forces.hpp` | vitesses de signal, conductivité |
| `add_to_derivs_sph_artif_visco_cond(...)` | `forces.hpp:173` | contribution complète pression+AV+conductivité d'une paire |
| `q_av(rho, vsig, v·r̂)`, `q_av_disc(...)` | `q_ab.hpp` | terme de viscosité |
| `mhd.hpp` | | forces magnétiques, induction |

### 16.6 Configuration, logs, erreurs

| Outil | Usage |
|---|---|
| `std::variant` + `std::get_if<Mode>(&cfg.xxx)` | style de toutes les options (EOS, AV, dust, forces…) |
| `to_json` / `from_json` (nlohmann) | chaque option doit être sérialisable (dumps `.sham`, `print_status`) |
| `solver_config.get_constant_G()`, `get_constant_c()`, `get_constant_mu_0()` | constantes dans les unités code |
| `shamlog_debug_ln("Tag", ...)`, `logger::info_ln`, `ON_RANK_0(...)` | logs |
| `shambase::throw_with_loc<std::invalid_argument>("msg")`, `shambase::make_except_with_loc<...>` | exceptions avec localisation |
| `StackEntry stack_loc{};` / `__shamrock_stack_entry();` | à mettre en tête de fonction (pile d'appels/profiling) |
| `shambase::get_check_ref(ptr)` | déréférence un `shared_ptr` en vérifiant qu'il n'est pas nul |

---

## 17. Recettes : ajouter de la physique pas à pas

### 17.A Un terme de force local (sans voisins)

Fil rouge : la dissipation de vitesse **a = −η v**, déjà implémentée — reproduisez le même
chemin pour votre force `MaForce`. Six fichiers :

**1. La donnée de config** — `src/shammodels/common/include/shammodels/common/ExtForceConfig.hpp`

```cpp
struct MaForce { Tscal param1; Tvec param2; };               // dans ExtForceVariant
using VariantForce = std::variant<..., VelocityDissipation, MaForce>;
// dans ExtForceConfig :
using MaForce = typename ExtForceVariant<Tvec>::MaForce;
inline void add_ma_force(Tscal p1, Tvec p2) {
    ext_forces.push_back(ExtForceVariant<Tvec>{MaForce{p1, p2}});
}
```

**2. La sérialisation JSON** — même fichier, `to_json` (≈ l.210) et `from_json` (≈ l.270) :
ajouter une branche `"force_type": "ma_force"`. Sans cela, les dumps et `print_status` échouent
(`throw_unimplemented`).

**3. Le nœud qui calcule la force** — nouveau fichier
`src/shammodels/common/include/shammodels/common/modules/AddForceMaForce.hpp`, calqué sur
`AddForceVelocityDissipation.hpp` :

```cpp
#define NODE_EDGES(X_RO, X_RW)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, param1)              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)     \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

template<class Tvec>
class AddForceMaForce : public shamrock::solvergraph::INode {
    using Tscal = shambase::VecComponent<Tvec>;
  public:
    EXPAND_NODE_EDGES(NODE_EDGES)
    inline void _impl_evaluate_internal() {
        __shamrock_stack_entry();
        auto edges = get_edges();
        edges.spans_positions.check_sizes(edges.sizes.indexes);
        edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);
        Tscal p1 = edges.param1.data;
        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_positions.get_spans()},
            sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
            edges.sizes.indexes,
            [p1](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                axyz_ext[gid] += /* votre formule */;   // += : les forces se cumulent
            });
    }
    inline std::string _impl_get_label() const { return "AddForceMaForce"; }
    inline std::string _impl_get_tex() const { return "TODO"; }
};
#undef NODE_EDGES
```

**4. Le branchement dans le pas de temps** — `src/shammodels/sph/src/modules/ExternalForces.cpp`
(+ alias `EF_MaForce` dans `ExternalForces.hpp:60`) :

- si la force ne dépend **que des positions** : ajouter une branche
  `else if (EF_MaForce *f = std::get_if<EF_MaForce>(&var_force.val)) {…}` dans
  `compute_ext_forces_indep_v()` (modèle : branche `EF_PointMass`, l.135) ; elle écrit dans
  `axyz_ext`, calculé une fois par pas ;
- si elle dépend de la **vitesse** (ou change pendant le correcteur) : branche dans
  `add_ext_forces()` (modèle : `EF_VelocityDissipation`, l.567), qui écrit directement dans
  `axyz` à chaque itération du correcteur, et laisser une branche **vide** dans
  `compute_ext_forces_indep_v()` (sinon `throw_unimplemented`) ;
- dans les deux fonctions, toute branche doit exister (même vide) : il y a un `else
  throw_unimplemented` à la fin des chaînes de `if`.

**5. L'exposition Python** — `src/shammodels/sph/src/pySPHModel.cpp` (près de l.492) :

```cpp
.def("add_ext_force_ma_force",
     [](TConfig &self, Tscal param1, Tvec param2) {
         self.ext_force_config.add_ma_force(param1, param2);
     },
     py::kw_only(), py::arg("param1"), py::arg("param2"),
     R"==(Docstring : formule, unités, référence.)==")
```

**6. Le reste** : si la force est aussi pertinente pour les sinks, voir `SinkParticlesUpdate`
et le nœud `"sink ext force"` ; si elle impose un pas de temps (raideur), voir
[17.F](#17f-un-nouveau-critère-de-pas-de-temps) ; écrire un exemple dans `examples/sph/`
(il sert de test CI et de doc), et un test unitaire ([§18](#18-règles-du-jeu)).

> La même mécanique `ExtForceConfig` est partagée avec le modèle Ramses (dossier `common/`) :
> une force bien écrite dans `common/modules/` est réutilisable des deux côtés.

### 17.B Un terme qui somme sur les voisins

Exemple : un terme de diffusion, de conduction, de traînée bi-fluide, un opérateur différentiel…
Modèle à copier : `modules/NodeComputePressureGrad.{hpp,cpp}` (le plus lisible).

1. **Entrées** : `gpart_mass` (`IDataEdge<Tscal>`), `part_counts` et `part_counts_with_ghost`
   (`Indexes<u32>`), les champs lus **avec ghosts** (`IFieldSpan<…>`), `neigh_cache`
   (`NeighCache`). **Sortie** : un `IFieldSpan` de taille `part_counts`.
2. Écrire le noyau comme un **foncteur** (`struct KernelX { Tscal pmass; void operator()(u32 id_a,
   const Tvec* xyz, …, shamrock::tree::ObjectCache::ptrs_read ploop_ptrs, Tvec* out) const; }`) :
   plus lisible qu'une lambda et réutilisable.
3. Dans `_impl_evaluate_internal` : `check_sizes(part_counts_with_ghost)` sur les entrées,
   `ensure_sizes(part_counts)` sur les sorties, puis `distributed_data_kernel_call` en passant
   `edges.neigh_cache` dans le `DDMultiRef` des entrées.
4. Instancier pour les 6 noyaux en fin de `.cpp` :
   `template class …<f64_3, M4>; … <f64_3, C6>;`
5. Déclarer le `.cpp` dans `src/shammodels/sph/CMakeLists.txt`.
6. Le brancher dans `Solver::update_derivs` (`Solver.cpp`, juste avant
   `UpdateDerivs::update_derivs`) en construisant les arêtes à partir du storage, comme au
   bloc TVA l.2680 :
   `map_field_refs(scheduler(), idx, *refs)` pour un champ du layout ;
   `storage.part_counts`, `storage.part_counts_with_ghost`, `storage.neigh_cache`,
   `storage.omega`, `storage.pressure`, `storage.positions_with_ghosts`,
   `storage.hpart_with_ghosts`.
7. Si votre terme a besoin d'un champ **des voisins** qui n'est pas déjà dans le ghost layout,
   l'ajouter à `SolverConfig::set_ghost_layout` **et** à `communicate_merge_ghosts_fields`
   (`Solver.cpp:2126`).

**Symétrie et conservation** : écrire les termes de paire sous forme antisymétrique
(a_ab = −a_ba), comme `sph_pressure_symetric`, pour conserver impulsion/énergie ;
`ConservativeCheck` vous alertera sinon.

### 17.C Une nouvelle variable évoluée (nouvelle équation d'évolution)

Exemple déjà présent : `s_j` / `ds_j_dt` pour la poussière TVA — suivez « `s_j` » dans
`Solver.cpp` avec `grep -n s_j`. Étapes :

1. **Option de config** qui active la physique (+ JSON + binding Python) — §17.A pts 1, 2, 5.
2. **Layout** : dans `SolverConfig::set_layout` (`SolverConfig.cpp`), `pdl.add_field<Tscal>("q", nvar)`
   et `pdl.add_field<Tscal>("dq_dt", nvar)` sous condition (`if (has_field_q())`). Ajouter le
   prédicat `has_field_q()` dans `SolverConfig.hpp`.
3. **Ghosts** si les voisins lisent `q` : `set_ghost_layout` + `communicate_merge_ghosts_fields`.
4. **Graphe** (`init_solver_graph`) : `register_edge("q", FieldRefs<Tscal>(...))`, un nœud
   `GetFieldRefFromLayer<Tscal>(pdl, "q")` dans la séquence `"attach fields"`, et un
   `ForwardEuler<Tscal>(nvar)` dans `make_half_step_sequence` (prédicteur, demi-pas).
5. **Correcteur** : `old_dq_dt` dans `SolverStorage`, sauvegarde dans `prepare_corrector`,
   puis `utility.fields_leapfrog_corrector<Tscal>(iq, idq_dt, storage.old_dq_dt.get(), eps, dt/2)`
   dans `evolve_once` (à côté des blocs `has_s_j_field`), sans oublier le reset du
   `Component` en fin de pas.
6. **Le calcul de `dq_dt`** : un nœud (17.B si voisins) appelé dans `update_derivs`.
7. **Positivité** : s'il faut `q ≥ 0`, voir `ForwardEulerPositive` et
   `fields_leapfrog_corrector_positive_only`.
8. **Sorties** : le champ apparaît automatiquement dans `collect_data`, les dumps et
   `compute_field("q", "f64")` ; ajouter la sortie VTK/Phantom si nécessaire
   (`modules/io/VTKDump.cpp`, `io/PhantomDump.cpp`).

### 17.D Une nouvelle équation d'état

1. Formule « pure » dans `src/shamphys/include/shamphys/eos.hpp` : un
   `struct EOS_MonEOS { static constexpr T pressure(...); static constexpr T soundspeed(...); }`.
2. Variante dans `src/shammodels/common/include/shammodels/common/EOSConfig.hpp`
   (`struct MonEOS {…}`, ajout au `Variant`, `set_mon_eos(...)`, JSON).
3. Prédicat et setter dans `SolverConfig.hpp` (cf. `is_eos_adiabatic`, `set_eos_adiabatic`) ;
   si elle a besoin de `uint` ou `soundspeed`, adapter `has_field_uint()` /
   `has_field_soundspeed()`.
4. Branche de calcul dans `src/shammodels/sph/src/modules/ComputeEos.cpp` (chaîne de
   `std::get_if<SolverEOS_…>` l.657) qui remplit `storage.pressure` et `storage.soundspeed`.
5. Binding `set_eos_mon_eos` dans `pySPHModel.cpp` ; test dans `src/tests/shammodels/EOSConfigTests.cpp`.
6. Vérifier les autres consommateurs de l'EOS : export Phantom (`io/PhantomDumpEOSUtils.cpp`),
   Ramses/GSPH s'ils doivent la supporter.

### 17.E Un nouveau générateur / modificateur de setup

1. Classe dérivée de `ISPHSetupNode` dans `modules/setup/` (modèle : `ModifierOffset.hpp`) :
   `is_done()`, `next_n(u32 nmax) -> PatchDataLayer` (produit au plus `nmax` particules),
   `get_name()`, `get_dot_subgraph()`.
2. Méthode fabrique `make_…` dans `modules/SPHSetup.hpp/.cpp`.
3. Binding dans le bloc `py::class_<TSPHSetup>` de `pySPHModel.cpp`.
4. Pour les générateurs aléatoires : graine explicite (`std::mt19937_64(seed)`) → déterminisme.

> Souvent inutile : un `make_modifier_filter` + `remap_positions` + `push_particle`
> depuis Python suffisent pour la plupart des conditions initiales exotiques.

### 17.F Un nouveau critère de pas de temps

Modules `modules/ComputeCFL*.hpp` (Courant, Force, Dust1Fluid, DustDrift, NIMHD, DivBCleaning,
SinkSink) : chacun calcule un dt candidat par particule, puis une réduction `min` globale.
Ajouter un module analogue et l'intégrer au calcul de `next_cfl` à la fin de la boucle
correcteur dans `evolve_once`. `set_show_cfl_detail(True)` affichera votre critère.

### 17.G Un terme source dans Ramses

Dans Ramses, **tout le pas de temps est déjà un graphe** (`Solver::init_solver_graph`,
`src/shammodels/ramses/src/Solver.cpp:291`), évalué par `storage.solver_sequence->evaluate()`
(l.2176), puis suivi de l'intégration temporelle hors graphe.

Ordre du graphe : ghosts (recherche, échange MPI, fusion) → arbres → voisins bloc/cellule →
AABB des cellules → (coordonnées, niveaux AMR, masse moyenne) → **ConsToPrim**
(`NodeConsToPrimGas*`, `NodeConsToPrimDust` : ρv, ρE → v, P) → **gradients limités**
(`SlopeLimitedScalarGradient`, `SlopeLimitedVectorGradient`) → dérivées temporelles des
primitives (demi-pas) → **interpolation aux faces** (`InterpolateToFace*`) → **flux**
(`NodeComputeFluxGasMode<…, Rusanov|HLL|HLLC>`) → **`NodeSumFluxHydro`** qui produit
`storage.dtrho`, `dtrhov`, `dtrhoe` → **forces externes** (l.1646-2052 : `ResetExtForceAcc`,
auto-gravité CG/BiCGSTAB + `NodeSelfGravityAcceleration`, chaque force configurée, puis
`ApplyExtForceGas` / `ApplyExtForceGasTrapezoidal`).
Après le graphe (`evolve_once`, l.2065) : `TimeIntegrator::forward_euler(dt)` ou
`DragIntegrator`, raffinement AMR, `BarotropicEnergyReset`, `ComputeCFL`.

Recette :

1. **Config** : paramètres + `is_xxx_on()` + validation dans `check_config()`
   (`SolverConfig.hpp:383`) + JSON dans `src/SolverConfig.cpp`. Pour une force, ajouter une
   alternative à `config/ExtForceConfig.hpp` (variant + JSON).
2. **Python** : `config_cls.def("set_xxx", …)` dans `pyRamsesModel.cpp`.
3. **Champs persistants** éventuels : `SolverConfig::set_layout` (`SolverConfig.cpp:31`) et, si les
   voisins doivent les lire, le ghost layout (`Solver.cpp:304`).
4. **Arêtes temporaires** : `std::shared_ptr<solvergraph::Field<T>>` dans
   `modules/SolverStorage.hpp` (cf. `a_ext`, `phi_g`).
5. **Nœud** : `modules/MonNoeud.{hpp,cpp}` calqué sur `AddExtForceConstantGravity` (force) ou
   `ApplyExtForceGas` (terme source), dans le namespace `shammodels::basegodunov::modules` ;
   instanciation explicite en fin de `.cpp` (`template class …<f64_3>;` pour ces deux-là — copier
   les paramètres template du module imité) ; ajout au `CMakeLists.txt`.
6. **Branchement** dans `init_solver_graph` :
   - **une force** : branche dans le `std::visit` sur `ext_force_config.ext_forces`
     (l.1957-1972) qui *ajoute* une accélération à chaque `a_ext_level` ; le couplage
     Midpoint/Trapézoïdal et l'application au gaz (et à la poussière) sont alors gratuits. Si
     elle dépend de la position, activer le champ de coordonnées
     (`is_coordinate_field_required`) ;
   - **un terme source général** : un nœud placé après `NodeSumFluxHydro` qui ajoute sa
     contribution à `storage.dtrhov` / `storage.dtrhoe` → intégré par `TimeIntegrator` ;
   - **un terme raide/implicite** (comme la traînée) : dans la branche d'intégration de
     `evolve_once` (l.2180-2193), sous forme de module `(context, solver_config, storage)`.
7. **Pas de temps** : combiner votre contrainte avec `ComputeCFL` (l.2233).
8. **Exemple** dans `examples/ramses/` et vérification avec `model.get_solver_dot_graph()`.

---

## 18. Règles du jeu

### 18.1 Déterminisme (invariant du projet — `dev_doc/determinism.md`)

Deux exécutions identiques doivent donner des sorties **bit à bit identiques**. À proscrire :

- les `atomic fetch_add` sur des flottants (ordre de sommation non fixé) → utiliser les
  primitives `shamalgs` (réductions, scans, tris) ;
- itérer sur un conteneur dont l'ordre n'est pas fixé pour faire une somme → `for_each_ordered` ;
- les graines aléatoires non fixées, les décisions basées sur le temps mesuré.

Vérification : lancer deux fois le même setup et comparer les dumps.

### 18.2 Conventions (`AGENTS.md`)

- Classes/structs/enums en `CamelCase`, fonctions/variables/membres en `lower_case`,
  macros en `UPPER_CASE`, paramètres template de type en `CamelCase` (`Tvec`), non-type en
  `lower_case` (`dim`).
- Fichier = nom de la classe qu'il contient (`AddForceMaForce.hpp`), ou `lower_case` pour un
  ensemble de fonctions libres.
- `.clang-format`, `.clang-tidy`, `pre-commit run --all-files` avant de committer.
- Titre de commit : `[sph][common] add MaForce external force`.

### 18.3 Compiler et tester

```bash
cd build
./shamenv_do shammake shammodels_sph   # compile seulement la cible modifiée
./shamenv_do shammake                  # tout (pour lancer les tests ou un script)
./shamenv_do ./shamrock --smi          # liste des devices
./shamenv_do ./shamrock_test --sycl-cfg 0:0 --loglevel 1 --unittest
./shamenv_do ./shamrock --sycl-cfg 0:0 --rscript ../examples/sph/run_sod.py
```

Test unitaire (`dev_doc/testing.md`) : un `.cpp` dans `src/tests/<module>/` (découvert
automatiquement) :

```cpp
#include "shamtest/shamtest.hpp"
NEW_TEST(Unittest, "shammodels/common/AddForceMaForce", 1) {  // 1 = nb de rangs MPI
    // construire des arêtes, set_edges, evaluate, comparer à la solution analytique
    REQUIRE_FLOAT_EQUAL(valeur, attendu, tolerance);
}
```

Macros disponibles (`src/shamtest/`) : `REQUIRE`, `REQUIRE_EQUAL`, `REQUIRE_FLOAT_EQUAL`,
`REQUIRE_EXCEPTION_THROW`, et leurs variantes `_NAMED`. Modèle : `src/tests/shammodels/EOSConfigTests.cpp`.

Bonnes pratiques de validation physique : un exemple dans `examples/` qui compare à une
solution analytique (onde, relaxation exponentielle, orbite…) avec une erreur L2 sauvée en JSON
(modèle : `run_sod.py`, `run_dustywave_tva.py`).

---

## Annexe A — Champs SPH disponibles

Défini par `SolverConfig::set_layout` (`src/shammodels/sph/src/SolverConfig.cpp`). Ce sont les
noms à utiliser dans `set_value_in_a_box`, `set_field_value_lambda_*`, `overwrite_field_value_*`,
`compute_field`, `collect_data`.

| Champ | Type | nvar | Présent si |
|---|---|---|---|
| `xyz` | f64_3 | 1 | toujours (position) |
| `vxyz` | f64_3 | 1 | toujours (vitesse) |
| `axyz` | f64_3 | 1 | toujours (accélération totale) |
| `axyz_ext` | f64_3 | 1 | toujours (accélération externe, indépendante de v) |
| `hpart` | f64 | 1 | toujours (longueur de lissage) |
| `part_id` | u64 | 1 | `set_particle_tracking(True)` |
| `uint`, `duint` | f64 | 1 | EOS avec énergie interne (adiabatique…) |
| `alpha_AV` | f64 | 1 | AV variable (MM97, CD10) |
| `divv`, `dtdivv`, `curlv` | f64 / f64 / f64_3 | 1 | AV variable (`dtdivv`, `curlv` : CD10) |
| `soundspeed` | f64 | 1 | EOS localement isotherme ou AV qui en a besoin |
| `B/rho`, `dB/rho`, `drho/dt` | f64_3 / f64_3 / f64 | 1 | MHD |
| `psi/ch`, `dpsi/ch` | f64 | 1 | MHD (nettoyage de divergence) |
| `divB`, `curlB` | f64 / f64_3 | 1 | MHD |
| `J` | f64_3 | 1 | MHD non idéale |
| `epsilon`, `dtepsilon` | f64 | ndust | poussière monofluide complète |
| `deltav`, `dtdeltav` | f64_3 | ndust | poussière monofluide complète |
| `s_j`, `ds_j_dt`, `delta_v` | f64 / f64 / f64_3 | ndust | poussière TVA |
| `luminosity` | f64 | 1 | `use_luminosity(True)` |
| `dt_part` | f64 | 1 | `set_save_dt_to_fields(True)` |

Champs fantômes (communiqués entre patches, `set_ghost_layout`) : `hpart`, `uint`, `vxyz`,
`omega`, (+ `axyz`, `soundspeed`, `B/rho`, `psi/ch`, `curlB`, `epsilon`, `deltav`, `s_j` selon
la physique).

---

## Annexe B — Où chercher quoi

| Question | Fichier |
|---|---|
| Quelles méthodes Python a le modèle SPH ? | `src/shammodels/sph/src/pySPHModel.cpp` |
| Quelles options de config existent ? | `src/shammodels/sph/include/shammodels/sph/SolverConfig.hpp` |
| Quels champs portent les particules ? | `src/shammodels/sph/src/SolverConfig.cpp` (`set_layout`) |
| Dans quel ordre se passe un pas de temps ? | `src/shammodels/sph/src/Solver.cpp` (`init_solver_graph`, `evolve_once`) |
| Comment s'écrit la force de pression / l'AV ? | `src/shammodels/sph/src/modules/UpdateDerivs.cpp`, `math/forces.hpp` |
| Comment ajouter une force externe ? | `common/ExtForceConfig.hpp`, `common/modules/AddForce*.hpp`, `sph/src/modules/ExternalForces.cpp` |
| Formules d'EOS | `src/shamphys/include/shamphys/eos.hpp`, `sph/src/modules/ComputeEos.cpp` |
| Noyaux SPH | `src/shammath/include/shammath/sphkernels.hpp` |
| Lancer un noyau GPU | `src/shambackends/include/shambackends/kernel_call*.hpp`, `dev_doc/kernel_call.md` |
| Écrire un nœud | `src/shamsolvergraph/include/shamsolvergraph/node/INode.hpp`, `dev_doc/solvergraph*.md` |
| Arêtes disponibles | `src/shamrock/include/shamrock/solvergraph/` |
| Exemples de setups | `examples/sph/`, `examples/ramses/`, `examples/gsph/` |
| Utilitaires Python | `src/pylib/shamrock/` |
