# Stochastic simulation of Patagonian tortoise movement

This repository contains two simulations I developed for my master's-equivalent thesis at Instituto Balseiro (Argentina). They model the movement of the Patagonian tortoise (*Chelonoidis chilensis*) over a full year, from GPS-scale steps of a few seconds up to seasonal changes in behaviour. The simulated trajectories are compared with GPS tracking data from wild tortoises, using the mean-squared displacement (MSD) and displacement distributions.

Both scripts are written in Python and compiled with [Numba](https://numba.pydata.org/): one runs on the CPU, the other on an NVIDIA GPU with CUDA.

## Two models of the night

The two scripts are not simply CPU and GPU versions of the same model. They make **different assumptions about what happens overnight**, which is one of the questions I explore in my thesis.

| | `LogNormal_CPU_FINAL.py` | `GPU_FINAL_Noches_v2.py` |
| --- | --- | --- |
| **Model** | Nightly reset | Night-spanning rest |
| **Overnight** | Every bout ends before 20:00 | A rest bout can last through the night |
| **At dawn** | Always starts the day walking | Continues whatever it was doing the evening before |
| **Hardware** | CPU, one tortoise at a time | GPU, one tortoise per CUDA thread (up to millions) |
| **Step length** | Lognormal | Lognormal, power law or constant |
| **MSD error** | Pooled over all (tortoise, reference time) pairs | Averaged per tortoise first |
| **Extras** | — | Intra-day MSD mode, walking density, batch processing |

Comparing the two shows how much the assumption about the night changes the simulated statistics.

## The model

Each tortoise performs a **stop-and-go correlated random walk** in 2D.

- **Steps:** one step per tick (1.7 s). Step lengths are drawn from a lognormal distribution (cm), and the heading is the previous heading plus Gaussian noise.
- **Bouts:** walking and resting alternate. Bout durations follow truncated power laws, *P*(*t*) ∝ *t*<sup>α</sup>, so long bouts are much more common than they would be in a memoryless process.
- **Day and night:** tortoises are active from 08:00 to 20:00 and do not move at night.
- **Refuges:** every night during exploration, the sleeping position is stored as a refuge (burrow).

The year is divided into three stages:

1. **Exploration** (January, and again after the winter): free movement; new refuges are registered every night.
2. **Intermediate** (February–March): each day the tortoise chooses a known refuge within its daily reach and walks towards it with a heading bias, then rests once it arrives.
3. **Pre-brumation** (April–October): the tortoise restricts itself to up to five nearby refuges and keeps homing to them.

The radius used to choose reachable refuges is **calibrated automatically**: before the main run, the code simulates one active day many times and uses the mean distance covered, so the radius always matches the current movement parameters.

## What the code shows

- **Memory-efficient statistics.** Storing a full year of positions for a million tortoises would need about 150 TB. Instead, positions are saved only at pre-selected times on a logarithmic grid, and the MSD is accumulated on the fly.
- **GPU parallelism.** Each tortoise is a CUDA thread with its own random-number stream. Displacement histograms are built directly on the GPU with atomic operations, and tortoises are processed in batches sized automatically to fit the available GPU memory.
- **Correct uncertainty estimates.** In the GPU version, the MSD is first averaged over reference times within each tortoise, so that correlated samples are not counted as independent.
- **Sampling conditioned on the day–night cycle.** In the GPU version, rest durations are drawn from a power law conditioned on ending in daylight, by splitting the distribution into one interval per day and inverting the CDF within the chosen interval.

## Repository contents

```
LogNormal_CPU_FINAL.py    CPU simulation, nightly reset model
GPU_FINAL_Noches_v2.py    GPU simulation, night-spanning rest model
README.md                 This file
LICENSE                   MIT licence
```

Each script begins with a detailed docstring describing the model, the algorithms and every output file.

## Requirements

- Python 3.9 or later
- `numpy`
- `numba`
- For the GPU version: an NVIDIA GPU with CUDA and a CUDA-enabled Numba installation

```bash
pip install numpy numba
```

## Usage

**CPU version** (1,000 tortoises, one year):

```bash
python LogNormal_CPU_FINAL.py --T 18550395 --simulation_runs 1000
```

**GPU version** (one million tortoises, one year, lognormal steps):

```bash
python GPU_FINAL_Noches_v2.py --bool_Lognormal --T 18550395 --simulation_runs 1000000
```

> **Note:** in the GPU version, `--bool_Lognormal` must be given explicitly. Without it (and without `--bool_Powerlaw`), the simulation uses a constant step length instead.

Run either script with `--help` to see all options, including the step-length and bout-duration parameters, the stage boundaries and the MSD settings.

## Output

Results are written to `<folder_name>/<sub_folder_name>/`:

| File | Content |
| --- | --- |
| `msd_<stage>.csv` | MSD and its standard error for each lag |
| `taus_<stage>.csv` | Lag times (ticks) |
| `Delta_distances_histogram_<stage>.csv` | Distribution of displacements between samples |
| `example_trajectory.npy` | Trajectory of one tortoise, for plotting |
| `example_attractors.npy` | Refuge positions of that tortoise |
| `walking_density.csv` | Fraction of daytime spent moving (GPU only) |

`<stage>` is `exploration`, `intermediate` or `prebrumation`. The docstring at the top of each script lists every output file.

## Data

The GPS tracking data used to fit and validate these models were collected by field collaborators and are not yet published, so they are **not included** in this repository. The simulations run without any input data.

## Known limitations

- The CPU version pools all (tortoise, reference time) pairs when computing the MSD standard error, which underestimates it. The GPU version corrects this.
- Movement is not switched off during brumation itself; the pre-brumation stage represents homing to a small set of winter refuges.
- The refuge buffers are sized for a one-year simulation. Longer simulations accumulate more refuges than the buffers hold.

## Author

Luciano Carlos López Bertazza — Instituto Balseiro, Argentina
luciano.lopez@ib.edu.ar

## Licence

Released under the [MIT License](LICENSE).
