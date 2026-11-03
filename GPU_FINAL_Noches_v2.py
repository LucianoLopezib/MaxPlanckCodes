"""
================================================================================
Tortoise Yearly Cycle Simulation — GPU Version (CUDA via Numba)
================================================================================

OVERVIEW
--------
This code simulates the 2D random-walk movement of desert tortoises over a full
yearly cycle. Each tortoise is assigned to one CUDA thread and all N_SIM
trajectories are computed in parallel on the GPU. The simulation runs for T
timesteps (default ~1 year at 1.7 s/step) using a lognormal step-length
distribution and a Gaussian angular noise model.

The full (N_SIM × T × 2) position array is never allocated. Positions are
recorded only at a small set of pre-selected snapshot timesteps and
displacement samples are accumulated directly into shared histograms on the
GPU using atomic operations.

BATCH PROCESSING
----------------
  For large N_SIM (e.g. 10^6), allocating all attractor and snapshot arrays
  simultaneously would exceed available VRAM (error 700). The simulation is
  therefore split into batches of batch_size tortoises. For each batch:
    1. GPU arrays (snapshots, attractors, rng_states) are allocated for
       batch_size threads only.
    2. The kernel runs and fills the snapshot buffer and shared histograms.
    3. Snapshot positions are copied back and stored in a CPU array.
    4. Histogram counts are accumulated across batches by simple addition.
  After all batches the MSD is computed from the full CPU snapshot array and
  the accumulated histograms are normalised into PDFs.

  Memory per batch (batch_size=100,000, n_snaps≈5376, MAX_ATTR=50):
    snapshots  : 100k × 5376 × 2 × 4 B = 4300 MB   ← dominant allocation
    attr arrays: 3 × 100k × 65 × 4 B  =   78 MB
    rng_states : 100k × 48 B           =    5 MB
    histograms : 3 × 100 × 4 B         = negligible
    ─────────────────────────────────────────────────────────────────────
    Total                               ≈ 4400 MB  (~4.3 GB)

  n_snaps grows with the MSD configuration (n_refs=48, base=1.5, stage
  boundaries) and is computed at runtime. The VRAM guard below
  automatically caps batch_size so d_snapshots fits in available VRAM.

EXAMPLE TRAJECTORY
------------------
  For visualisation purposes, the full position history of the very first
  tortoise (tid == 0, always in the first batch) is saved at a decimated
  time resolution controlled by --traj_every (default: one point per hour).
  Its attractor positions are also saved. These files allow plotting a
  representative individual trajectory alongside the statistical outputs.

  Files saved:
    example_trajectory.npy  — (n_traj_points, 2) x/y positions
    example_attractors.npy  — (n_attractors, 2)  burrow positions

BIOLOGICAL MODEL
----------------
The yearly cycle is divided into three behavioural stages:

  Stage 0 — Exploration (days 0 to day_exploration_end):
    Correlated random walk. Every night the sleeping position is registered as
    a new refuge (attractor). Walk and stop durations follow a power-law
    distribution, capped to the remaining daytime window.

  Stage 1 — Intermediate / Homing (days day_exploration_end to day_revisit_end):
    At dawn a random attractor is selected; the tortoise walks toward it and
    stops for the rest of the day once it arrives (arrived_at_burrow).

  Stage 2 — Pre-brumation (days day_revisit_end to day_brumation_end):
    At dawn the 5 nearest attractors within 1 km are found; one is chosen at
    random and the tortoise homes toward it with stronger heading bias (×1.5).

DAILY RHYTHM
------------
  Night  (20:00–08:00): no movement; attractor registered at 20:00 in stage 0.
  Dawn reset (08:00)  : walk/stop counters reset; fresh walk_time drawn;
                        daily target attractor selected for stages 1 and 2.
  Day    (08:00–20:00): movement governed by stage rules above.

MSD — ENSEMBLE APPROACH
-----------------------
  Reference time per stage: day 2 of the stage at 11 AM (transient decayed).
  Lag grid: tau_k = round(2^k), k = 0, 1, ... while t_ref + tau_k < stage end.
  MSD(tau) = <|r(t_ref+tau) - r(t_ref)|^2>  over N_SIM tortoises
  SEM(tau) = std(r^2) / sqrt(N_SIM)

DELTA DISTANCES
---------------
  Every Delta_T timesteps each thread computes its displacement since the
  previous sample. The kernel accumulates a shared histogram directly on the
  GPU using cuda.atomic.add — no per-thread arrays, negligible VRAM usage.
  Three separate histograms are maintained (one per stage). Night ticks are
  excluded. Bin range: [0, max_step × Delta_T] so no values are ever clamped.

CUDA-SPECIFIC CONSTRAINTS
-------------------------
  - Python built-ins (min, max) are NOT available inside @cuda.jit kernels.
  - cuda.local.array() MUST be declared at the kernel's top scope.
  - cuda.synchronize() is called after each kernel launch to surface errors
    immediately at the launch site rather than later at copy_to_host().
  - Each thread gets an independent xoroshiro128p RNG stream.

OUTPUT FILES  (written to folder_name/sub_folder_name/)
-------------------------------------------------------
  snapshots.npy                         — (N_SIM, n_snaps, 2) positions
  snapshot_times.npy                    — absolute timesteps of all snapshots
  attr_x.npy, attr_y.npy, attr_count.npy
  t_refs.csv                            — reference timestep per stage
  taus_{stage}.csv                      — lag values for each stage
  msd_{stage}.csv                       — columns: msd, sem
  Delta_distances_histogram_{stage}.csv — columns: distance, pdf, poisson_error
  example_trajectory.npy                — full trajectory of tortoise 0
  example_attractors.npy                — burrow positions of tortoise 0

USAGE
-----
  python LogNormal_GPU_FINAL.py --T 18550395 --simulation_runs 1000000 [options]
  See the CLI section at the bottom for all arguments and their defaults.

DEPENDENCIES
------------
  numpy, numba (cuda + xoroshiro128p RNG), argparse, os, time, math
================================================================================
"""

import numpy as np
import math
import time

from numba import cuda, njit
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
    xoroshiro128p_normal_float32
)

import os
import argparse


# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

# Walk/stop duration limits in timesteps.
# Max_time_mov    ≈ 2.79 hours  (upper bound for a single walk bout)
# Max_time_freeze ≈ 27.7 hours  (upper bound for a single stop bout)


data_max         = 5.6667    # maximum step length in cm (lognormal rejection cap)
Mean_Angular_Var = np.pi / 36  # maximum heading correction per step (5 degrees)


# ==============================================================================
# CUDA DEVICE FUNCTIONS
# ==============================================================================

@cuda.jit(device=True)
def powerlaw_interval(rng_states, tid, alpha, min_time, max_time):
    """
    Draw from a truncated power-law P(x)~x^alpha on [min_time, max_time]
    using the inverse-CDF method. alpha==-1 gives log-uniform.
    """
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    if alpha == -1.0:
        return min_time * (max_time / min_time) ** u
    else:
        a1 = alpha + 1.0
        return ((max_time**a1 - min_time**a1) * u + min_time**a1) ** (1.0 / a1)


@cuda.jit(device=True)
def clip_value(v, lo, hi):
    """Clamp v to [lo, hi] (Python min/max are not available in kernels)."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@cuda.jit(device=True)
def powerlaw_ending_in_day(rng_states, tid, alpha, min_time, max_time,
                           t_start, tic_per_day, night_ends, night_begins):
    D  = float(tic_per_day)
    h  = float(t_start % tic_per_day)
    a1 = alpha + 1.0
    K  = int(max_time / D) + 2          # number of candidate days

    # pass 1: total weight
    wsum = 0.0
    for k in range(K):
        lo = clip_value(night_ends   - h + k*D, min_time, max_time)
        hi = clip_value(night_begins - h + k*D, min_time, max_time)
        wsum += math.log(hi/lo) if alpha == -1.0 else math.fabs(hi**a1 - lo**a1)
    if wsum <= 0.0:
        return max_time

    # pass 2: pick the interval and invert the CDF inside it
    r = xoroshiro128p_uniform_float32(rng_states, tid) * wsum
    for k in range(K):
        lo = clip_value(night_ends   - h + k*D, min_time, max_time)
        hi = clip_value(night_begins - h + k*D, min_time, max_time)
        w  = math.log(hi/lo) if alpha == -1.0 else math.fabs(hi**a1 - lo**a1)
        if r < w or k == K - 1:
            break
        r -= w
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    if alpha == -1.0:
        return lo * (hi/lo) ** u
    return ((hi**a1 - lo**a1) * u + lo**a1) ** (1.0 / a1)

@cuda.jit(device=True)
def wrap_angle(angle):
    """Wrap angle to (-pi, pi]."""
    if angle > math.pi:
        angle -= 2.0 * math.pi
    elif angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@cuda.jit(device=True)
def lognormal_step(rng_states, tid, mu, sigma):
    """Draw one step length from LogNormal(mu, sigma)."""
    n = xoroshiro128p_normal_float32(rng_states, tid)
    return math.exp(mu + sigma * n)


@cuda.jit(device=True)
def gaussian_angle_lognormal_step(x, y, theta, rng_states, tid,
                                   mu_vel, sigma_vel, angular_sigma, max_step):
    """
    One correlated random-walk step. Heading perturbed by Normal(0, angular_sigma),
    step length from LogNormal(mu_vel, sigma_vel) with rejection above max_step.
    Returns new (x, y, heading).
    """
    noise     = xoroshiro128p_normal_float32(rng_states, tid)
    new_angle = wrap_angle(theta + angular_sigma * noise)
    step      = lognormal_step(rng_states, tid, mu_vel, sigma_vel)
    attempts  = 0
    while step > max_step and attempts < 10:
        step     = lognormal_step(rng_states, tid, mu_vel, sigma_vel)
        attempts += 1
    return x + step * math.cos(new_angle), y + step * math.sin(new_angle), new_angle


@cuda.jit(device=True)
def powerlaw_step(rng_states, tid):
    """
    Draw one step length from a truncated power law P(l) ~ l^-2
    on [step_min, step_max] using the inverse-CDF method.
    For alpha=-2, alpha+1=-1:
      l = 1 / ((1/step_max - 1/step_min) * u + 1/step_min)
    Constants hardcoded — CUDA device functions do not support default arguments.
    """
    step_min = 0.3137
    step_max = 5.6667
    u = xoroshiro128p_uniform_float32(rng_states, tid)
    return 1.0 / ((1.0/step_max - 1.0/step_min) * u + 1.0/step_min)


@cuda.jit(device=True)
def gaussian_angle_powerlaw_step(x, y, theta, rng_states, tid,
                                  angular_sigma):
    noise     = xoroshiro128p_normal_float32(rng_states, tid)
    new_angle = wrap_angle(theta + angular_sigma * noise)
    step      = powerlaw_step(rng_states, tid)
    return x + step * math.cos(new_angle), y + step * math.sin(new_angle), new_angle


@cuda.jit(device=True)
def gaussian_angle_fixed_step(x, y, theta, rng_states, tid, angular_sigma, constant_vel):
    noise     = xoroshiro128p_normal_float32(rng_states, tid)
    new_angle = wrap_angle(theta + angular_sigma * noise)
    step      = constant_vel  # fixed step length
    return x + step * math.cos(new_angle), y + step * math.sin(new_angle), new_angle


@cuda.jit(device=True)
def guided_walker(x, y, theta, target_x, target_y, rng_states, tid,
                  mu_vel, sigma_vel, angular_sigma, e_theta, max_step, bool_Lognormal, bool_Powerlaw, constant_vel):
    """
    One biased random-walk step toward (target_x, target_y). Heading correction
    clamped to ±e_theta per step, then Gaussian noise added. Same lognormal
    step length as the free walk. Returns new (x, y, heading).
    """
    target_angle = math.atan2(target_y - y, target_x - x)
    delta = target_angle - theta
    if delta > math.pi:
        delta -= 2.0 * math.pi
    elif delta < -math.pi:
        delta += 2.0 * math.pi
    if delta > e_theta:
        turn = e_theta
    elif delta < -e_theta:
        turn = -e_theta
    else:
        turn = delta
    noise     = xoroshiro128p_normal_float32(rng_states, tid)
    new_angle = wrap_angle(theta + turn + angular_sigma * noise)
    if bool_Lognormal:
        step      = lognormal_step(rng_states, tid, mu_vel, sigma_vel)
    elif bool_Powerlaw:
        step      = powerlaw_step(rng_states, tid)
    else:
        step      = constant_vel  # fixed step length
    attempts  = 0
    while step > max_step and attempts < 10:
        if bool_Lognormal:
            step     = lognormal_step(rng_states, tid, mu_vel, sigma_vel)
        elif bool_Powerlaw:
            step     = powerlaw_step(rng_states, tid)
        else:
            step     = 1.0
        attempts += 1
    return x + step * math.cos(new_angle), y + step * math.sin(new_angle), new_angle


@cuda.jit(device=True)
def create_attraction_point(x, y, attr_x, attr_y, attr_fit, attr_count, rng_states, tid):
    """
    Register (x, y) as a new attractor with a random fitness from Uniform(0,1).
    Silently skips if the attractor buffer is full.
    """
    idx = attr_count[tid]
    if idx >= attr_x.shape[1]:
        return
    attr_x[tid, idx]   = x
    attr_y[tid, idx]   = y
    attr_fit[tid, idx] = xoroshiro128p_uniform_float32(rng_states, tid)
    attr_count[tid]   += 1


@cuda.jit(device=True)
def in_burrow(x, y, ref_x, ref_y):
    """True if (x,y) is within 30 m of (ref_x, ref_y). Uses squared distance."""
    dx = x - ref_x
    dy = y - ref_y
    return (dx * dx + dy * dy) < 900.0


@cuda.jit(device=True)
def find_closest_attractors(x, y, attr_x, attr_y, attr_count, tid,
                             max_k, radius2, best_idx, best_dist):
    """
    Find the max_k nearest attractors within squared-distance radius2.
    best_idx and best_dist must be cuda.local.array() declared at kernel top scope.
    Slots with no qualifying attractor are left as -1 / 1e20.
    """
    for i in range(max_k):
        best_dist[i] = 1e20
        best_idx[i]  = -1
    count = attr_count[tid]
    for i in range(count):
        dx = attr_x[tid, i] - x
        dy = attr_y[tid, i] - y
        d2 = dx*dx + dy*dy
        if d2 <= radius2:
            for j in range(max_k):
                if d2 < best_dist[j]:
                    for k in range(max_k - 1, j, -1):
                        best_dist[k] = best_dist[k-1]
                        best_idx[k]  = best_idx[k-1]
                    best_dist[j] = d2
                    best_idx[j]  = i
                    break


# ==============================================================================
# CPU HELPER FUNCTIONS
# ==============================================================================

@njit
def powerlaw_stop_interval(alpha=-2, min_time=1, max_time=1000):
    """
    CPU version of the power-law duration sampler — used by the calibration
    function only. The kernel uses the CUDA device version above.
    """
    u = np.random.uniform(0, 1)
    if alpha == -1:
        return min_time * (max_time / min_time) ** u
    else:
        return ((max_time**(alpha+1) - min_time**(alpha+1)) * u
                + min_time**(alpha+1)) ** (1.0 / (alpha+1))

@njit
def calibration_step_length(mu_vel, sigma_vel, bool_Lognormal, bool_Powerlaw, constant_vel):
    """
    Host-side step length mirroring the kernel's guided_walker step selection, so
    the radius calibration matches whichever movement mode the simulation uses:
      - lognormal: LogNormal(mu_vel, sigma_vel), rejected above data_max (<=10 tries)
      - powerlaw : truncated power law on [0.3137, 5.6667] (same as powerlaw_step)
      - otherwise: a fixed step of length constant_vel
    """
    if bool_Lognormal:
        step = np.random.lognormal(mu_vel, sigma_vel)
        attempts = 0
        while step > data_max and attempts < 10:
            step = np.random.lognormal(mu_vel, sigma_vel)
            attempts += 1
        return step
    elif bool_Powerlaw:
        step_min = 0.3137
        step_max = 5.6667
        u = np.random.uniform(0.0, 1.0)
        return 1.0 / ((1.0 / step_max - 1.0 / step_min) * u + 1.0 / step_min)
    else:
        return constant_vel


@njit
def calibrate_stage1_radius(mu_vel, sigma_vel, angular_sigma, alpha_walk, alpha_stop,
                              bool_Lognormal, bool_Powerlaw, constant_vel,
                              night_ends_c, night_begins_c, tic_per_day_c,
                              e_theta=np.pi/36, Max_time_mov=1000, Max_time_freeze=10000,
                              n_reps=500, n_burnin_days=2):
    """
    Estimate the mean net displacement a tortoise can achieve in one full active
    day, with actual walk/stop power-law alternation and a guided walk toward a
    target placed infinitely far away (never reached, so the tortoise always
    walks in the same direction when the walk state is active).

    Including stops is essential: the distribution of walk/stop durations
    directly determines what fraction of the day the tortoise is moving, so the
    calibrated radius automatically reflects the current alpha_walk / alpha_stop
    parameters. The per-step length follows the same movement mode as the kernel
    (lognormal / powerlaw / constant), so the radius is consistent with the
    configured step distribution as well.

    The walk/stop rules are the SAME as in the kernel: walk bouts are capped
    at min(Max_time_mov, time left until night); stop bouts are drawn from the
    full distribution [1, Max_time_freeze] conditioned on ending in daytime, so
    a stop may span the night and the day does not start with a forced walk.
    To reach that regime, each replica runs n_burnin_days full days first and
    the displacement is measured over the following active day (dawn to dusk).

    Returns the mean end-of-day displacement in cm — used directly as
    stage1_radius so the filter is self-consistent with the simulation dynamics.
    """
    total_disp = 0.0
    target_x   = 1.0e9          # effectively infinite distance — never reached
    target_y   = 0.0

    n_days = n_burnin_days + 1
    t_meas_start = n_burnin_days * tic_per_day_c + night_ends_c    # dawn of measured day
    t_meas_end   = n_burnin_days * tic_per_day_c + night_begins_c  # dusk of measured day

    for _ in range(n_reps):
        x     = 0.0
        y     = 0.0
        angle = np.random.uniform(-np.pi, np.pi)
        x0    = 0.0
        y0    = 0.0

        # Same initial condition as the kernel: resting until the first dawn
        WALK       = False
        count_walk = 0
        walk_time  = 0.0
        stop_end   = float(night_ends_c)

        for t in range(1, n_days * tic_per_day_c):
            time_in_day = t % tic_per_day_c

            if t == t_meas_start:
                x0 = x
                y0 = y
            if t == t_meas_end:
                break

            # Night: no movement; a walk reaching nightfall turns into a stop
            if time_in_day < night_ends_c or time_in_day >= night_begins_c:
                if time_in_day == night_begins_c and WALK:
                    WALK       = False
                    count_walk = 0
                    stop_end   = t + powerlaw_ending_in_day_cpu(
                        alpha_stop, 1.0, float(Max_time_freeze), t,
                        tic_per_day_c, night_ends_c, night_begins_c)
                continue

            remaining_day = night_begins_c - time_in_day

            if WALK:
                if count_walk < walk_time:
                    # Inline guided step toward far-away target
                    target_angle = np.arctan2(target_y - y, target_x - x)
                    delta = (target_angle - angle + np.pi) % (2*np.pi) - np.pi
                    if delta > e_theta:
                        turn = e_theta
                    elif delta < -e_theta:
                        turn = -e_theta
                    else:
                        turn = delta
                    new_angle   = np.random.normal(angle + turn, angular_sigma)
                    new_angle   = (new_angle + np.pi) % (2*np.pi) - np.pi
                    step_length = calibration_step_length(mu_vel, sigma_vel, bool_Lognormal, bool_Powerlaw, constant_vel)
                    x    += step_length * np.cos(new_angle)
                    y    += step_length * np.sin(new_angle)
                    angle = new_angle
                    count_walk += 1
                else:
                    # Walk bout finished: stop drawn from the full distribution,
                    # conditioned on ending in daytime (may span the night)
                    WALK       = False
                    count_walk = 0
                    stop_end   = t + powerlaw_ending_in_day_cpu(
                        alpha_stop, 1.0, float(Max_time_freeze), t,
                        tic_per_day_c, night_ends_c, night_begins_c)
            else:
                if t >= stop_end:
                    # Stop bout finished: walk capped by the time left until night
                    WALK     = True
                    max_walk = remaining_day
                    if max_walk > Max_time_mov:
                        max_walk = Max_time_mov
                    walk_time = powerlaw_stop_interval(alpha_walk, min_time=1, max_time=max_walk)

        total_disp += np.sqrt((x - x0)**2 + (y - y0)**2)

    return total_disp / n_reps


@njit
def powerlaw_ending_in_day_cpu(alpha, min_time, max_time,
                               t_start, tic_per_day, night_ends, night_begins):
    """
    CPU (numba) twin of the device function powerlaw_ending_in_day: draws a
    stop duration from the truncated power law conditioned on the stop ending
    in daytime. Used only by calibrate_stage1_radius.
    """
    D  = float(tic_per_day)
    h  = float(t_start % tic_per_day)
    a1 = alpha + 1.0
    lo = np.empty(3)
    hi = np.empty(3)
    w  = np.empty(3)
    wsum = 0.0
    for k in range(3):
        l = night_ends   - h + k * D
        u = night_begins - h + k * D
        l = min(max(l, min_time), max_time)
        u = min(max(u, min_time), max_time)
        lo[k] = l
        hi[k] = u
        if alpha == -1.0:
            w[k] = np.log(u / l)
        else:
            w[k] = abs(u**a1 - l**a1)
        wsum += w[k]
    if wsum <= 0.0:
        return max_time
    r = np.random.random() * wsum
    k = 0
    while k < 2 and r > w[k]:
        r -= w[k]
        k += 1
    v = np.random.random()
    if alpha == -1.0:
        return lo[k] * (hi[k] / lo[k]) ** v
    return ((hi[k]**a1 - lo[k]**a1) * v + lo[k]**a1) ** (1.0 / a1)


def compute_stages(T, tic_per_day,
                   day_exploration_end, day_revisit_end, day_brumation_end):
    """
    Build a stage-index array of length T (int32, for GPU transfer).
    0=exploration, 1=intermediate, 2=pre-brumation. Calendar wraps mod 365.
    """
    stages = np.zeros(T, dtype=np.int32)
    for t in range(T):
        day = (t // tic_per_day) % 365
        if   day < day_exploration_end: stages[t] = 0
        elif day < day_revisit_end: stages[t] = 1
        elif day < day_brumation_end:   stages[t] = 2
        else:                           stages[t] = 0
    return stages


def compute_snapshot_times(T, tic_per_day, night_ends,
                            day_exploration_end, day_revisit_end, day_brumation_end,
                            seconds_per_time_step,
                            n_refs=48, ref_spacing_minutes=30, base=1.5):
    """
    Precompute snapshot timesteps for an MSD averaged over multiple reference
    times. For each stage, n_refs reference times are placed on day 2 of the
    stage, spaced ref_spacing_minutes apart and covering 24 hours. The lag
    grid is the same for all reference times within a stage, capped by the
    last (most restrictive) reference time.

    Returns
    -------
    snapshot_times   : sorted int32 array of all unique absolute timesteps
    t_refs_per_stage : list of 3 int64 arrays of shape (n_refs,)
    taus_per_stage   : list of 3 int64 lag arrays (same for all refs in a stage)
    ref_snap_idxs    : list of 3 int32 arrays of shape (n_refs,)
    lag_snap_idxs    : list of 3 int32 arrays of shape (n_refs, n_lags)
    """
    ref_spacing_ticks = int(ref_spacing_minutes * 60 / seconds_per_time_step)

    stage_day_starts = [0,                  day_exploration_end, day_revisit_end]
    stage_day_ends   = [day_exploration_end, day_revisit_end, day_brumation_end  ]

    t_refs_per_stage = []
    taus_per_stage   = []
    all_snaps        = set()

    for d_start, d_end in zip(stage_day_starts, stage_day_ends):
        stage_t_start = d_start * tic_per_day
        stage_t_end   = min(d_end * tic_per_day, T)

        # n_refs reference times starting at day 2 of the stage
        t_ref_0 = stage_t_start + tic_per_day
        t_refs  = np.array([t_ref_0 + k * ref_spacing_ticks for k in range(n_refs)],
                            dtype=np.int64)
        t_refs_per_stage.append(t_refs)

        for t_ref in t_refs:
            all_snaps.add(int(t_ref))

        # Lag grid capped by the LAST reference time (smallest available window)
        max_tau = stage_t_end - int(t_refs[-1]) - 1
        taus = []
        k = 0
        while True:
            tau = int(round(base ** k))
            if tau > max_tau:
                break
            taus.append(tau)
            k += 1
        taus = np.array(taus, dtype=np.int64)
        taus_per_stage.append(taus)

        for t_ref in t_refs:
            for tau in taus:
                all_snaps.add(int(t_ref) + int(tau))

    snapshot_times = np.array(sorted(all_snaps), dtype=np.int32)
    snap_lookup    = {int(s): i for i, s in enumerate(snapshot_times)}

    ref_snap_idxs = []
    lag_snap_idxs = []
    for t_refs, taus in zip(t_refs_per_stage, taus_per_stage):
        r_idxs = np.array([snap_lookup[int(t)] for t in t_refs], dtype=np.int32)
        ref_snap_idxs.append(r_idxs)

        n_lags = len(taus)
        l_idxs = np.zeros((len(t_refs), n_lags), dtype=np.int32)
        for ki, t_ref in enumerate(t_refs):
            for j, tau in enumerate(taus):
                l_idxs[ki, j] = snap_lookup[int(t_ref) + int(tau)]
        lag_snap_idxs.append(l_idxs)

    return snapshot_times, t_refs_per_stage, taus_per_stage, ref_snap_idxs, lag_snap_idxs


def compute_snapshot_times_daily(T, tic_per_day,
                                  day_exploration_end, day_revisit_end, day_brumation_end,
                                  seconds_per_time_step,
                                  start_hour=9.0, end_hour=19.0,
                                  burnin_days=5, every_days=1, base=1.5):
    """
    Snapshot times for the DAILY (intra-day) MSD.

    One reference time per day, at start_hour (clock time of the simulation),
    and lags from 1 tick up to (end_hour - start_hour), so every window lies
    within the same day [start_hour, end_hour]. The MSD is averaged over all
    days of the stage (and then over tortoises, as in the standard MSD).

    The first burnin_days of each stage are skipped, so that the reference
    days are not affected by the synchronized initial condition or by the
    stage transition. every_days > 1 uses only one day out of every_days
    (reduces memory: n_snaps ~ n_days * n_lags).

    Same return signature as compute_snapshot_times, so the accumulation and
    finalization functions are reused unchanged.
    """
    start_tick = int(round(start_hour * 3600.0 / seconds_per_time_step))
    end_tick   = int(round(end_hour   * 3600.0 / seconds_per_time_step))
    max_tau    = end_tick - start_tick
    if max_tau <= 0:
        raise ValueError("msd_end_hour must be later than msd_start_hour")

    # Common lag grid: log-spaced (base**k), unique, plus the full window
    taus = sorted({int(round(base ** k)) for k in range(200)
                   if int(round(base ** k)) <= max_tau} | {max_tau})
    taus = np.array(taus, dtype=np.int64)

    stage_day_starts = [0,                   day_exploration_end, day_revisit_end]
    stage_day_ends   = [day_exploration_end, day_revisit_end,     day_brumation_end]

    t_refs_per_stage = []
    taus_per_stage   = []
    all_snaps        = set()

    for d_start, d_end in zip(stage_day_starts, stage_day_ends):
        refs = []
        for d in range(d_start + burnin_days, d_end, every_days):
            t_ref = d * tic_per_day + start_tick
            if t_ref + max_tau < min(d_end * tic_per_day, T):
                refs.append(t_ref)
        t_refs = np.array(refs, dtype=np.int64)
        t_refs_per_stage.append(t_refs)
        stage_taus = taus if len(t_refs) > 0 else np.zeros(0, dtype=np.int64)
        taus_per_stage.append(stage_taus)
        for t_ref in t_refs:
            all_snaps.add(int(t_ref))
            for tau in stage_taus:
                all_snaps.add(int(t_ref) + int(tau))

    snapshot_times = np.array(sorted(all_snaps), dtype=np.int32)
    snap_lookup    = {int(v): i for i, v in enumerate(snapshot_times)}

    ref_snap_idxs = []
    lag_snap_idxs = []
    for t_refs, stage_taus in zip(t_refs_per_stage, taus_per_stage):
        ref_snap_idxs.append(np.array([snap_lookup[int(t)] for t in t_refs], dtype=np.int32))
        l_idxs = np.zeros((len(t_refs), len(stage_taus)), dtype=np.int32)
        for ki, t_ref in enumerate(t_refs):
            for j, tau in enumerate(stage_taus):
                l_idxs[ki, j] = snap_lookup[int(t_ref) + int(tau)]
        lag_snap_idxs.append(l_idxs)

    return snapshot_times, t_refs_per_stage, taus_per_stage, ref_snap_idxs, lag_snap_idxs


def accumulate_msd_batch(batch_snaps, ref_snap_idxs, lag_snap_idxs,
                          taus_per_stage, msd_sum_r2, msd_sum_r4, msd_count):
    """
    Accumulate per-tortoise MSD statistics from one batch of snapshots.

    For each tortoise i and each lag j, first average r² over all n_refs
    reference times to get r²_i(j) — the per-tortoise estimate. Then
    accumulate sum(r²_i) and sum(r²_i²) across tortoises.

    This two-level averaging is statistically correct: the n_refs references
    from the same tortoise are correlated, so variance must be computed only
    across the N_SIM independent tortoises — not across N_SIM × n_refs pairs.

    Parameters
    ----------
    batch_snaps : (batch_size, n_snaps, 2) float32
    msd_sum_r2, msd_sum_r4, msd_count : lists of 3 float64 arrays — modified in-place
    """
    for s in range(3):
        r_idxs = ref_snap_idxs[s]    # (n_refs,)
        l_idxs = lag_snap_idxs[s]    # (n_refs, n_lags)
        n_lags = len(taus_per_stage[s])

        # Reference positions: (batch_size, n_refs)
        ref_x = batch_snaps[:, r_idxs, 0]
        ref_y = batch_snaps[:, r_idxs, 1]

        for j in range(n_lags):
            # Lag positions: (batch_size, n_refs)
            lag_x = batch_snaps[:, l_idxs[:, j], 0]
            lag_y = batch_snaps[:, l_idxs[:, j], 1]

            # r² for every (tortoise, ref) pair: (batch_size, n_refs)
            r2_ik = (lag_x - ref_x)**2 + (lag_y - ref_y)**2

            # Per-tortoise average over the n_refs references: (batch_size,)
            r2_i = np.mean(r2_ik, axis=1)

            # Accumulate sums across tortoises for final mean and SEM
            msd_sum_r2[s][j] += np.sum(r2_i,    dtype=np.float64)
            msd_sum_r4[s][j] += np.sum(r2_i**2, dtype=np.float64)
            msd_count[s][j]  += len(r2_i)


def finalize_msd(msd_sum_r2, msd_sum_r4, msd_count, taus_per_stage):
    """
    Compute final MSD and SEM from per-tortoise accumulated sums.

    MSD(τ)  = mean_i( r²_i(τ) )
    SEM(τ)  = std_i( r²_i(τ) ) / sqrt(N_SIM)
            = sqrt( (E[r⁴_i] - E[r²_i]²) / N_SIM )

    The variance is computed only across independent tortoises, not across
    the n_refs × N_SIM pairs, which would underestimate SEM by sqrt(n_refs).

    Returns list of (taus, msd, sem) per stage.
    """
    results = []
    for s in range(3):
        taus   = taus_per_stage[s].astype(np.float64)
        n_lags = len(taus)
        msd    = np.zeros(n_lags)
        sem    = np.zeros(n_lags)
        for j in range(n_lags):
            N = msd_count[s][j]
            if N == 0:
                continue
            mean_r2 = msd_sum_r2[s][j] / N
            mean_r4 = msd_sum_r4[s][j] / N
            msd[j]  = mean_r2
            var_r2  = max(mean_r4 - mean_r2**2, 0.0)
            sem[j]  = np.sqrt(var_r2 / N)
        results.append((taus, msd, sem))
    return results


def str2bool(v):
    """Parse 'True'/'False'-like strings for argparse ('--MSD_Diario True')."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def hist_to_frequency(hist, total):
    """
    Convert a raw count histogram into a plain frequency histogram.

    frequencies : count_i / total  — fraction of samples in each bin
    errors      : sqrt(count_i) / total  — Poisson uncertainty

    Returns (None, None) if total == 0 (no samples for this stage).
    """
    if total == 0:
        return None, None
    frequencies = hist / total
    errors      = np.sqrt(hist) / total
    return frequencies, errors


# ==============================================================================
# CUDA KERNEL
#
# Each thread simulates one tortoise for the full T timesteps.
# Writes only to:
#   - snapshots[tid, snap_ptr, :]  at designated snapshot times
#   - shared histograms via cuda.atomic.add at every Delta_T ticks
#   - traj_x/traj_y[t // traj_every] for tid==0 (example trajectory)
# ==============================================================================

@cuda.jit
def simulate_kernel(
    # Snapshot buffer
    snapshots,        # (batch_size, n_snaps, 2) float32
    snapshot_times,   # (n_snaps,) int32 — sorted absolute timesteps
    n_snaps,
    # Shared histograms (one per stage) — updated with atomic adds
    hist_0,           # (n_bins,) int32 — exploration
    hist_1,           # (n_bins,) int32 — intermediate
    hist_2,           # (n_bins,) int32 — pre-brumation
    total_0,          # (1,) int32 — total samples in exploration
    total_1,          # (1,) int32 — total samples in intermediate
    total_2,          # (1,) int32 — total samples in pre-brumation
    bin_width,        # float32 — metres per histogram bin
    n_bins,           # int — number of histogram bins
    Delta_T,          # int — ticks between displacement samples
    # Attractor arrays
    attr_x,           # (batch_size, MAX_ATTR) float32
    attr_y,
    attr_fit,
    attr_count,       # (batch_size,) int32
    # Stage array
    stages,           # (T,) int32
    # RNG states
    rng_states,
    # Simulation length
    T,
    # Movement parameters
    bool_Lognormal, bool_Powerlaw, constant_walk, bool_constant_walk_time, constant_stop, bool_constant_stop_time, # bools that select behaviours
    mu_vel, sigma_vel, angular_sigma, max_step, constant_vel,
    alpha_walk, alpha_stop,
    e_theta,
    tic_per_day, night_begins, night_ends,
    Max_time_mov_k, Max_time_freeze_k,
    stage1_radius2,   # float32 — calibrated reachable distance² for stage 1 target selection
    stage2_radius2,   # float32 — fraction of stage1_radius² for stage 2 shortlist search
    # Example trajectory — written only for tid==0
    traj_x,           # (n_traj_points,) float32
    traj_y,           # (n_traj_points,) float32
    traj_every,       # int — ticks between trajectory samples
    is_first_batch,   # int — 1 if this is batch 0, 0 otherwise
    # Missed-burrow counter — atomic add when a day ends without reaching target
    missed_days,      # (1,) int32
    # Walking-density accumulators (summed over all threads, flushed once per thread)
    walk_steps,       # (1,) int64 — active steps on which the tortoise actually moved
    active_steps      # (1,) int64 — all active (non-night) steps
):
    tid = cuda.grid(1)
    if tid >= snapshots.shape[0]:
        return

    # Local arrays — declared at top scope (CUDA compiler requirement)
    best_idx      = cuda.local.array(5,  dtype=np.int32)   # scratch: nearest attractors
    best_dist     = cuda.local.array(5,  dtype=np.float32) # scratch: their squared distances
    shortlist_idx = cuda.local.array(5,  dtype=np.int32)   # stage-2 fixed shortlist (built once at stage entry)
    stage1_cand   = cuda.local.array(50, dtype=np.int32)   # stage-1 all-within-radius candidates

    # Movement state
    x                 = 0.0
    y                 = 0.0
    theta             = 0.0
    # Start resting; the first rest ends at the first dawn (t = night_ends),
    # where the first walk bout is drawn by the normal Stop -> Walk transition.
    WALK              = False
    count_walk        = 0.0
    count_stop        = 0.0
    walk_time         = Max_time_mov_k
    stop_time         = 0.0
    stop_end          = float(night_ends)   # absolute tick at which the current stop ends
    target_idx        = 0
    arrived_at_burrow = False

    # Stage 2 shortlist state — shortlist is built once on the first dawn of
    # pre-brumation and reused every subsequent dawn (CPU-equivalent behaviour)
    first_brumation = True   # becomes False after shortlist is built
    n_shortlist     = 0      # how many valid entries are in shortlist_idx

    # Walking-density counters — per-thread registers, flushed once at kernel end
    # to avoid global-atomic contention on every step.
    my_walk_steps   = np.int64(0)   # active steps with an actual move
    my_active_steps = np.int64(0)   # all active (non-night) steps

    # Reference position for displacement sampling
    old_x = 0.0
    old_y = 0.0

    # Forward pointer into sorted snapshot_times — O(1) check per tick
    snap_ptr = 0

    for t in range(1, T):

        stage       = stages[t]
        time_in_day = t % tic_per_day

        # ======================================================================
        # NIGHT  (20:00 – 08:00)
        # At 20:00 in stage 0: register sleeping position as new attractor.
        # Snapshots still recorded if this tick is a snapshot time.
        # 'continue' skips movement and delta accumulator — night zeros excluded.
        # ======================================================================
        if time_in_day >= night_begins or time_in_day < night_ends:

            if time_in_day == night_begins and stage == 0:
                create_attraction_point(
                    x, y, attr_x, attr_y, attr_fit, attr_count, rng_states, tid)

            # Count days in stages 1/2 where the tortoise never reached its burrow
            if time_in_day == night_begins and stage > 0 and not arrived_at_burrow:
                cuda.atomic.add(missed_days, 0, 1)

            # Walk bout cut by nightfall -> start a stop that must end in daytime.
            # (A stop already in progress simply continues through the night.)
            if time_in_day == night_begins and WALK:
                WALK       = False
                count_walk = 0.0
                if bool_constant_stop_time:
                    stop_time = constant_stop
                else:
                    stop_time = powerlaw_ending_in_day(
                        rng_states, tid, alpha_stop, 1.0, Max_time_freeze_k,
                        t, tic_per_day, night_ends, night_begins)
                stop_end = t + stop_time

            while snap_ptr < n_snaps and snapshot_times[snap_ptr] == t:
                snapshots[tid, snap_ptr, 0] = x
                snapshots[tid, snap_ptr, 1] = y
                snap_ptr += 1

            # Record trajectory during night — must be here before continue,
            # otherwise night ticks are never written and stay at the
            # initialised zero value, creating false returns to (0,0).
            if tid == 0 and is_first_batch == 1 and t % traj_every == 0:
                traj_idx = t // traj_every
                if traj_idx < traj_x.shape[0]:
                    traj_x[traj_idx] = x
                    traj_y[traj_idx] = y

            continue

        # ======================================================================
        # DAWN RESET  (08:00)
        # Reset walk/stop state; draw fresh walk_time capped to full daytime.
        # Stages 1/2: select today's target attractor.
        # ======================================================================
        if time_in_day == night_ends:
            # Behavioural state is NOT reset: a stop that started before the
            # night continues until stop_end (which always falls in daytime).
            arrived_at_burrow = False

            # Midday: halfway through the active window. The arrived_at_burrow
            # flag is only enforced after this point so the tortoise is free
            # to move during the morning regardless of burrow proximity.
            midday = night_ends + (night_begins - night_ends) // 2

            # Stage 1: pick uniformly from ALL attractors within the calibrated
            # radius (mean daily displacement). Fallback to full list if none qualify.
            if stage == 1 and attr_count[tid] > 0:
                n_cand = 0
                for i in range(attr_count[tid]):
                    dx = attr_x[tid, i] - x
                    dy = attr_y[tid, i] - y
                    if dx*dx + dy*dy <= stage1_radius2 and n_cand < 50:
                        stage1_cand[n_cand] = i
                        n_cand += 1
                if n_cand > 0:
                    u      = xoroshiro128p_uniform_float32(rng_states, tid)
                    choice = int(u * n_cand)
                    if choice >= n_cand:
                        choice = n_cand - 1
                    target_idx = stage1_cand[choice]
                else:
                    # Fallback: no burrow within radius — pick any
                    u      = xoroshiro128p_uniform_float32(rng_states, tid)
                    choice = int(u * attr_count[tid])
                    if choice >= attr_count[tid]:
                        choice = attr_count[tid] - 1
                    target_idx = choice
                arrived_at_burrow = in_burrow(
                    x, y, attr_x[tid, target_idx], attr_y[tid, target_idx])

            # Stage 2: build the shortlist ONCE on the first dawn of the stage.
            # Uses the same calibrated radius as stage 1 (reachable in one day)
            # so only burrows the tortoise can actually reach are considered.
            # The 5 nearest within that radius are stored and reused every dawn.
            elif stage == 2 and attr_count[tid] > 0:
                if first_brumation:
                    find_closest_attractors(
                        x, y, attr_x, attr_y, attr_count, tid,
                        5, stage2_radius2, best_idx, best_dist)  # fraction of the calibrated daily range
                    n_shortlist = 0
                    for i in range(5):
                        if best_idx[i] != -1:
                            shortlist_idx[n_shortlist] = best_idx[i]
                            n_shortlist += 1
                    # Fallback: nothing within 1 km — use the origin burrow
                    if n_shortlist == 0:
                        shortlist_idx[0] = 0
                        n_shortlist      = 1
                    first_brumation = False
                if n_shortlist > 0:
                    u          = xoroshiro128p_uniform_float32(rng_states, tid)
                    choice     = int(u * n_shortlist)
                    if choice >= n_shortlist:
                        choice = n_shortlist - 1
                    target_idx = shortlist_idx[choice]
                    arrived_at_burrow = in_burrow(
                        x, y, attr_x[tid, target_idx], attr_y[tid, target_idx])

        # ======================================================================
        # DAY  (08:00 – 20:00)
        # remaining_day caps new bout durations drawn mid-day.
        # ======================================================================
        remaining_day = night_begins - time_in_day

        # Walking-density: this tick is active (we are past the night 'continue').
        # Remember the position so we can tell afterwards whether a step was taken.
        my_active_steps += 1
        px = x
        py = y

        if WALK:
            effective_walk_time = walk_time
            if effective_walk_time > remaining_day:
                effective_walk_time = remaining_day

            if count_walk < effective_walk_time:

                # Stage 0: uncorrelated random walk
                if stage == 0:
                    if bool_Lognormal:
                        x, y, theta = gaussian_angle_lognormal_step(
                            x, y, theta, rng_states, tid,
                            mu_vel, sigma_vel, angular_sigma, max_step)
                    elif bool_Powerlaw:
                        x, y, theta = gaussian_angle_powerlaw_step(
                            x, y, theta, rng_states, tid,
                            angular_sigma)
                    else:
                        x, y, theta = gaussian_angle_fixed_step(
                            x, y, theta, rng_states, tid,
                            angular_sigma, constant_vel)

                # Stage 1: guided walk; rest once arrived at daily burrow
                # (only enforced after midday so morning movement is free)
                elif stage == 1:
                    if not arrived_at_burrow or time_in_day < midday:
                        tx = attr_x[tid, target_idx]
                        ty = attr_y[tid, target_idx]
                        x, y, theta = guided_walker(
                            x, y, theta, tx, ty, rng_states, tid,
                            mu_vel, sigma_vel, angular_sigma, e_theta, max_step, bool_Lognormal, bool_Powerlaw, constant_vel)
                        if time_in_day >= midday:
                            arrived_at_burrow = in_burrow(x, y, tx, ty)

                # Stage 2: stronger guided walk; rest once arrived
                # (only enforced after midday so morning movement is free)
                else:
                    if not arrived_at_burrow or time_in_day < midday:
                        tx = attr_x[tid, target_idx]
                        ty = attr_y[tid, target_idx]
                        x, y, theta = guided_walker(
                            x, y, theta, tx, ty, rng_states, tid,
                            mu_vel, sigma_vel, angular_sigma,
                            e_theta, max_step, bool_Lognormal, bool_Powerlaw, constant_vel)
                        if time_in_day >= midday:
                            arrived_at_burrow = in_burrow(x, y, tx, ty)

                count_walk += 1.0

            else:
                # Walk bout exhausted → transition to stop
                WALK       = False
                count_walk = 0.0
                # Full distribution [1, Max_time_freeze_k], conditioned on the
                # stop ending in daytime (it may span the night).
                if bool_constant_stop_time:
                    stop_time = constant_stop
                else:
                    stop_time = powerlaw_ending_in_day(
                        rng_states, tid, alpha_stop, 1.0, Max_time_freeze_k,
                        t, tic_per_day, night_ends, night_begins)
                stop_end = t + stop_time

        else:
            if t < stop_end:
                pass   # still resting (night ticks count towards the stop)
            else:
                # Stop bout exhausted → transition back to walk
                WALK       = True
                count_stop = 0.0
                max_walk   = remaining_day
                if max_walk > Max_time_mov_k:
                    max_walk = Max_time_mov_k
                if bool_constant_walk_time:
                    walk_time = constant_walk
                else:
                    walk_time = powerlaw_interval(rng_states, tid, alpha_walk, 1.0, max_walk)

        # Walking-density: a move occurred iff the position changed this tick.
        # (A walk step always changes x or y; stops and burrow-resting do not.)
        if x != px or y != py:
            my_walk_steps += 1

        # ======================================================================
        # SNAPSHOT CHECK
        # ======================================================================
        while snap_ptr < n_snaps and snapshot_times[snap_ptr] == t:
            snapshots[tid, snap_ptr, 0] = x
            snapshots[tid, snap_ptr, 1] = y
            snap_ptr += 1

        # ======================================================================
        # DELTA DISTANCE — atomic histogram update
        # Bin range [0, max_dist] covers the full theoretical displacement range.
        # Clamp is a safety guard only; legitimate values never exceed max_dist.
        # ======================================================================
        if t % Delta_T == 0:
            dx   = x - old_x
            dy   = y - old_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.0:
                bin_idx = int(dist / bin_width)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                elif bin_idx < 0:
                    bin_idx = 0

                if stage == 0:
                    cuda.atomic.add(hist_0, bin_idx, 1)
                    cuda.atomic.add(total_0, 0, 1)
                elif stage == 1:
                    cuda.atomic.add(hist_1, bin_idx, 1)
                    cuda.atomic.add(total_1, 0, 1)
                else:
                    cuda.atomic.add(hist_2, bin_idx, 1)
                    cuda.atomic.add(total_2, 0, 1)

            old_x = x
            old_y = y

        # ======================================================================
        # EXAMPLE TRAJECTORY — recorded only for tid==0 in the first batch.
        # Saves position at every traj_every ticks for visualisation.
        # Note: night-time recording is handled inside the night block above
        # so that night positions are not left as zero.
        # ======================================================================
        if tid == 0 and is_first_batch == 1 and t % traj_every == 0:
            traj_idx = t // traj_every
            if traj_idx < traj_x.shape[0]:
                traj_x[traj_idx] = x
                traj_y[traj_idx] = y

    # ──────────────────────────────────────────────────────────────────────────
    # End of time loop: flush this thread's walking-density counts to the global
    # accumulators with a single atomic add each (cheap — once per thread).
    # ──────────────────────────────────────────────────────────────────────────
    cuda.atomic.add(walk_steps,   0, my_walk_steps)
    cuda.atomic.add(active_steps, 0, my_active_steps)


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):

    start_time = time.perf_counter()
    N_SIM                 = args.simulation_runs
    T                     = args.T
    seconds_per_time_step = args.seconds_per_time_step
    bool_Lognormal        = args.bool_Lognormal
    bool_Powerlaw         = args.bool_Powerlaw
    bool_constant_walk_time = args.bool_constant_walk_time
    bool_constant_stop_time = args.bool_constant_stop_time
    Max_time_mov          = args.Max_time_mov
    Max_time_freeze       = args.Max_time_freeze

    

    # Cast movement parameters to float32 for GPU efficiency
    constant_vel  = np.float32(args.Constant_step_length)
    mu_vel        = np.float32(args.mu_vel)
    sigma_vel     = np.float32(args.sigma_vel)
    angular_sigma = np.float32(args.angular_sigma)
    alpha_walk    = np.float32(args.alpha_walk)
    alpha_stop    = np.float32(args.alpha_stop)
    constant_stop = np.float32(args.value_constant_stop_time)
    constant_walk = np.float32(args.value_constant_walk_time)
    e_theta       = np.float32(args.e_theta)   # 5-degree max heading correction per step
    if bool_Lognormal or bool_Powerlaw:
        max_step      = np.float32(5.6667)        # maximum step length in cm
    else:
        max_step      = np.float32(constant_vel)           # default maximum step length in cm

    tic_per_day  = int(24*60*60 / seconds_per_time_step)
    night_begins = int(20*60*60 / seconds_per_time_step)
    night_ends   = int( 8*60*60 / seconds_per_time_step)

    Max_time_mov_f    = np.float32(Max_time_mov*1.7/seconds_per_time_step)      # 1.7× due to not having in consideration the posibility
    Max_time_freeze_f = np.float32(Max_time_freeze*1.7/seconds_per_time_step)   # of a diferent second per time step  

    # ── Stage 1 radius calibration ────────────────────────────────────────────
    # Run on CPU before the GPU simulation. Uses float64 parameters (not float32)
    # so the calibration is identical to the CPU version.
    print("Calibrating stage 1 radius (500 reps)...")
    stage1_radius_cm = calibrate_stage1_radius(
        float(args.mu_vel), float(args.sigma_vel), float(args.angular_sigma),
        float(args.alpha_walk), float(args.alpha_stop),
        bool_Lognormal, bool_Powerlaw, float(constant_vel),
        night_ends, night_begins, tic_per_day, e_theta,
        Max_time_mov_f, Max_time_freeze_f)
    stage1_radius2_f = np.float32(stage1_radius_cm ** 2)
    print(f"  Mean daily displacement = {stage1_radius_cm:.1f} cm = {stage1_radius_cm/100:.2f} m  "
          f"→ stage1_radius = {stage1_radius_cm:.1f} cm")
    stage2_radius2_f = np.float32(stage1_radius_cm ** 2 * args.stage2_radius_fraction ** 2)
    print(f"  stage2_radius = stage1_radius × {args.stage2_radius_fraction} = "
          f"{stage1_radius_cm * args.stage2_radius_fraction:.1f} cm = "
          f"{stage1_radius_cm * args.stage2_radius_fraction / 100:.2f} m")

    # Needed by the VRAM budget formula below (declared early).
    MAX_ATTR = 65   # 30 or 60 days of exploration → max 30 attractors; 65 is safe
    threads  = 128

    # Requested batch size — may be reduced by the VRAM guard below.
    batch_size = min(args.batch_size, N_SIM)

    # Create output directories
    os.makedirs(args.folder_name, exist_ok=True)
    The_folder = os.path.join(args.folder_name, args.sub_folder_name)
    os.makedirs(The_folder, exist_ok=True)

    # Precompute stage array — shared by all threads and all batches
    stages = compute_stages(
        T, tic_per_day,
        args.day_exploration_end,
        args.day_revisit_end,
        args.day_brumation_end)

    # Precompute snapshot times — shared by all threads and all batches
    if args.MSD_Diario:
        # Intra-day MSD: one reference per day at msd_start_hour, lags up to msd_end_hour
        snapshot_times, t_refs_per_stage, taus_per_stage, ref_snap_idxs, lag_snap_idxs = \
            compute_snapshot_times_daily(
                T, tic_per_day,
                args.day_exploration_end,
                args.day_revisit_end,
                args.day_brumation_end,
                seconds_per_time_step,
                start_hour=args.msd_start_hour,
                end_hour=args.msd_end_hour,
                burnin_days=args.msd_burnin_days,
                every_days=args.msd_every_days)
        for s_i, refs_i in enumerate(t_refs_per_stage):
            print(f"[MSD diario] stage {s_i}: {len(refs_i)} reference days, "
                  f"{len(taus_per_stage[s_i])} lags")
    else:
        snapshot_times, t_refs_per_stage, taus_per_stage, ref_snap_idxs, lag_snap_idxs = \
            compute_snapshot_times(
                T, tic_per_day, night_ends,
                args.day_exploration_end,
                args.day_revisit_end,
                args.day_brumation_end,
                seconds_per_time_step)

    n_snaps     = len(snapshot_times)
    stage_names = ['exploration', 'intermediate', 'prebrumation']

    # ── VRAM budget check and automatic batch_size guard ──────────────────────
    # d_snapshots dominates VRAM: batch_size × n_snaps × 2 × 4 bytes (float32).
    # n_snaps grows with MSD complexity (n_refs × n_lags per stage) and can
    # easily reach 5000–8000, making the docstring's "69 snapshots / 55 MB"
    # estimate obsolete.  Auto-cap batch_size so snapshots alone stay within
    # the VRAM budget and leave ~1.5 GB headroom for attrs, RNG, stages, etc.
    try:
        free_vram, total_vram = cuda.current_context().get_memory_info()
    except Exception:
        free_vram, total_vram = 5 * 1024**3, 6 * 1024**3   # conservative 5 GB fallback

    snap_bytes_per_tortoise = int(n_snaps) * 2 * 4          # float32
    other_bytes_per_tortoise = (3 * MAX_ATTR * 4            # attr_x/y/fit
                                 + 4                         # attr_count int32
                                 + 48)                       # xoroshiro128p state
    bytes_per_tortoise = snap_bytes_per_tortoise + other_bytes_per_tortoise
    shared_bytes = (T * 4                                    # d_stages
                    + n_snaps * 4                            # d_snap_times
                    + (T // args.traj_every + 1) * 2 * 4   # d_traj_x/y
                    + 8 * 4)                                 # 6 hist + 2 misc scalars
    headroom     = 1_500 * 1024**2                           # 1.5 GB safety margin
    vram_budget  = max(0, free_vram - shared_bytes - headroom)
    safe_batch   = max(1, int(vram_budget / bytes_per_tortoise))

    if batch_size > safe_batch:
        print(f"  [VRAM guard] Requested batch_size={batch_size:,} would need "
              f"~{batch_size * bytes_per_tortoise / 1e6:.0f} MB for snapshots+attrs alone.\n"
              f"  Free VRAM ≈ {free_vram/1e9:.2f} GB  →  auto-capping to {safe_batch:,}.")
        batch_size = safe_batch

    n_batches = (N_SIM + batch_size - 1) // batch_size

    snap_mb   = batch_size * snap_bytes_per_tortoise / 1e6
    print(f"N_SIM={N_SIM}  batch_size={batch_size}  n_batches={n_batches}")
    print(f"Snapshot times: {n_snaps} total  (48 refs × 30 min on day 2 per stage)")
    print(f"  d_snapshots per batch: {snap_mb:.0f} MB  "
          f"({batch_size:,} × {n_snaps} snaps × 2 × 4 B)")
    for s, (t_refs, taus) in enumerate(zip(t_refs_per_stage, taus_per_stage)):
        print(f"  Stage {s} ({stage_names[s]}): {len(t_refs)} refs, {len(taus)} lags each")

    # Incremental MSD accumulators — avoids storing (N_SIM, n_snaps, 2) on CPU.
    # msd_sum_r2[s][j] = sum of per-tortoise r²_i(j) over all tortoises
    # msd_sum_r4[s][j] = sum of r²_i(j)² — needed for variance across tortoises
    # msd_count[s][j]  = number of tortoises accumulated = N_SIM when done
    msd_sum_r2 = [np.zeros(len(taus), dtype=np.float64) for taus in taus_per_stage]
    msd_sum_r4 = [np.zeros(len(taus), dtype=np.float64) for taus in taus_per_stage]
    msd_count  = [np.zeros(len(taus), dtype=np.int64)   for taus in taus_per_stage]

    # Histogram parameters.
    # max_dist = max_step * Delta_T covers the full theoretical displacement range
    # so no value is ever clamped to the last bin.
    
    max_dist  = float(max_step * args.Delta_T)
    n_bins    = args.n_hist_bins
    bin_width = max_dist / n_bins

    # Example trajectory parameters — one sample per traj_every ticks
    traj_every   = args.traj_every
    n_traj_pts   = T // traj_every + 1

    # ── Shared GPU arrays — transferred once, reused by all batches ───────────
    d_stages     = cuda.to_device(stages)
    d_snap_times = cuda.to_device(snapshot_times)
    
    # Shared histograms: allocated once and accumulated across all batches
    d_hist_0  = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
    d_hist_1  = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
    d_hist_2  = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
    d_total_0 = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_total_1 = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_total_2 = cuda.to_device(np.zeros(1, dtype=np.int64))

    # Example trajectory buffer (written only by tid==0 in batch 0)
    d_traj_x     = cuda.to_device(np.zeros(n_traj_pts, dtype=np.float32))
    d_traj_y     = cuda.to_device(np.zeros(n_traj_pts, dtype=np.float32))

    # Missed-burrow counter — shared across all batches, never reset
    d_missed_days = cuda.to_device(np.zeros(1, dtype=np.int32))

    # Walking-density accumulators — shared across all batches, never reset.
    # walk_steps  : active steps on which a tortoise actually moved
    # active_steps: all active (non-night) steps
    d_walk_steps   = cuda.to_device(np.zeros(1, dtype=np.int64))
    d_active_steps = cuda.to_device(np.zeros(1, dtype=np.int64))

    # Attractor arrays from the first batch — saved as part of the example output
    example_attr_x     = None
    example_attr_y     = None
    example_attr_count = None

    # ── BATCH LOOP ────────────────────────────────────────────────────────────
    for batch_idx in range(n_batches):

        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, N_SIM)
        cur_size    = batch_end - batch_start

        is_first = 1 if batch_idx == 0 else 0

        # Per-batch GPU arrays — allocated fresh each batch to limit VRAM usage
        attr_x     = np.zeros((cur_size, MAX_ATTR), dtype=np.float32)
        attr_y     = np.zeros((cur_size, MAX_ATTR), dtype=np.float32)
        attr_fit   = np.zeros((cur_size, MAX_ATTR), dtype=np.float32)
        attr_count = np.ones(cur_size, dtype=np.int32)   # origin burrow pre-registered

        d_attr_x     = cuda.to_device(attr_x)
        d_attr_y     = cuda.to_device(attr_y)
        d_attr_fit   = cuda.to_device(attr_fit)
        d_attr_count = cuda.to_device(attr_count)
        d_snapshots  = cuda.to_device(np.zeros((cur_size, n_snaps, 2), dtype=np.float32))

        # Independent RNG stream for each thread in this batch
        rng_states = create_xoroshiro128p_states(cur_size, seed=42 + batch_idx)

        blocks = (cur_size + threads - 1) // threads

        simulate_kernel[blocks, threads](
            d_snapshots, d_snap_times, n_snaps,
            d_hist_0, d_hist_1, d_hist_2,
            d_total_0, d_total_1, d_total_2,
            np.float32(bin_width), n_bins,
            args.Delta_T,
            d_attr_x, d_attr_y, d_attr_fit, d_attr_count,
            d_stages,
            rng_states,
            T,
            bool_Lognormal, bool_Powerlaw, constant_walk, bool_constant_walk_time, constant_stop, bool_constant_stop_time,
            mu_vel, sigma_vel, angular_sigma, max_step, constant_vel,  
            alpha_walk, alpha_stop,
            e_theta,
            tic_per_day, night_begins, night_ends,
            Max_time_mov_f, Max_time_freeze_f,
            stage1_radius2_f,
            stage2_radius2_f,
            d_traj_x, d_traj_y, traj_every, is_first,
            d_missed_days,
            d_walk_steps, d_active_steps
        )
        # Surface kernel errors immediately
        cuda.synchronize()

        # Accumulate MSD from this batch immediately — avoids large CPU array
        batch_snaps = d_snapshots.copy_to_host()   # (cur_size, n_snaps, 2)
        accumulate_msd_batch(batch_snaps, ref_snap_idxs, lag_snap_idxs,
                              taus_per_stage, msd_sum_r2, msd_sum_r4, msd_count)

        # Save attractor data from the first batch for the example output
        if batch_idx == 0:
            example_attr_x     = d_attr_x.copy_to_host()
            example_attr_y     = d_attr_y.copy_to_host()
            example_attr_count = d_attr_count.copy_to_host()

        print(f"  Batch {batch_idx+1}/{n_batches} done "
              f"(tortoises {batch_start}–{batch_end-1})")

        # Explicitly release per-batch GPU arrays so CUDA frees them before
        # the next iteration allocates fresh ones. Without this, Numba defers
        # deallocation and the peak usage momentarily doubles (old + new batch).
        del d_snapshots, d_attr_x, d_attr_y, d_attr_fit, d_attr_count, rng_states
        cuda.synchronize()

    print("All batches finished.")

    # ── POST-PROCESSING ON CPU ─────────────────────────────────────────────────

    # Copy final histograms and counts from GPU
    hist_0  = d_hist_0.copy_to_host()
    hist_1  = d_hist_1.copy_to_host()
    hist_2  = d_hist_2.copy_to_host()
    total_0 = int(d_total_0.copy_to_host()[0])
    total_1 = int(d_total_1.copy_to_host()[0])
    total_2 = int(d_total_2.copy_to_host()[0])

    # Empirical walking density: fraction of active (non-night) time the tortoises
    # spent actually moving, pooled over every tortoise and every active step.
    # This is the value to use as p_walk in a per-step Bernoulli (memoryless) model:
    # for such a model the long-run walking fraction equals p_walk exactly.
    walk_steps_total   = int(d_walk_steps.copy_to_host()[0])
    active_steps_total = int(d_active_steps.copy_to_host()[0])
    walking_density = (walk_steps_total / active_steps_total
                       if active_steps_total > 0 else 0.0)

    # Copy example trajectory
    traj_x_host = d_traj_x.copy_to_host()
    traj_y_host = d_traj_y.copy_to_host()

    # Finalize MSD from per-tortoise accumulated sums
    msd_results = finalize_msd(msd_sum_r2, msd_sum_r4, msd_count, taus_per_stage)

    # Histogram bin centres
    edges       = np.linspace(0.0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────

    np.save(os.path.join(The_folder, "snapshot_times.npy"), snapshot_times)

    # Per-stage displacement PDFs
    for hist, total, name in zip(
            [hist_0, hist_1, hist_2],
            [total_0, total_1, total_2],
            stage_names):
        frequencies, err = hist_to_frequency(hist, total)
        if frequencies is not None:
            out = np.column_stack([bin_centers, frequencies, err])
            np.savetxt(
                os.path.join(The_folder, f'Delta_distances_histogram_{name}.csv'),
                out, delimiter=',',
                header='distance,frequency,poisson_error', comments='')

    # Empirical walking density (use as p_walk in the per-step Bernoulli model)
    np.savetxt(
        os.path.join(The_folder, 'walking_density.csv'),
        np.array([[walking_density, walk_steps_total, active_steps_total]]),
        delimiter=',',
        header='walking_density,walk_steps,active_steps', comments='')

    # First reference tick of each stage (-1 if the stage has no references,
    # e.g. a stage the simulation never reaches in the daily-MSD mode)
    np.savetxt(os.path.join(The_folder, 't_refs.csv'),
               np.array([t_refs[0] if len(t_refs) > 0 else -1
                         for t_refs in t_refs_per_stage], dtype=np.int64),
               delimiter=',')
    msd_tag = '_diario' if args.MSD_Diario else ''
    if args.MSD_Diario:
        # All daily reference ticks (one per day) of each stage
        for s, t_refs in enumerate(t_refs_per_stage):
            np.savetxt(os.path.join(The_folder, f't_refs_diario_{stage_names[s]}.csv'),
                       np.asarray(t_refs, dtype=np.int64), fmt='%d', delimiter=',')
    for s, taus in enumerate(taus_per_stage):
        np.savetxt(os.path.join(The_folder, f'taus{msd_tag}_{stage_names[s]}.csv'),
                   taus.astype(np.float64), delimiter=',')

    # MSD results: two columns — msd and standard error of the mean
    for s, (taus, msd, sem) in enumerate(msd_results):
        if taus is None:
            continue
        msd_out = np.column_stack([msd, sem])
        np.savetxt(os.path.join(The_folder, f'msd{msd_tag}_{stage_names[s]}.csv'),
                   msd_out, delimiter=',',
                   header='msd,sem', comments='')

    # Example trajectory: (n_traj_pts, 2) array of x/y positions sampled
    # every traj_every ticks for tortoise 0 (always in batch 0)
    traj_out = np.column_stack([traj_x_host, traj_y_host])
    np.save(os.path.join(The_folder, 'example_trajectory.npy'), traj_out)

    # Example attractor positions for tortoise 0: (n_attractors, 2)
    # attr_x and attr_y were copied from batch 0 before it was overwritten
    n_attr_0     = int(example_attr_count[0])
    attractors_0 = np.column_stack([
        example_attr_x[0, :n_attr_0],
        example_attr_y[0, :n_attr_0]
    ])
    np.save(os.path.join(The_folder, 'example_attractors.npy'), attractors_0)

    elapsed = time.perf_counter() - start_time
    full_gb = N_SIM * T * 2 * 4 / 1e9
    print(f"Done. Runtime: {elapsed:.1f} s")
    print(f"(Full trajectory would be {full_gb:.2f} GB — used incremental MSD instead)")

    # Days in stages 1/2 where the tortoise never reached its target burrow.
    # A high count may indicate the radius is too small or Max_time_mov too short.
    total_missed = int(d_missed_days.copy_to_host()[0])
    print(f"Days burrow not reached (stages 1+2): {total_missed} "
          f"across {N_SIM} tortoises")

    # Empirical walking density — the p_walk to feed a per-step Bernoulli model
    print(f"Walking density (active-time fraction moved): {walking_density:.6f} "
          f"({walk_steps_total} / {active_steps_total} steps)")


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tortoise yearly cycle simulation — GPU version (CUDA via Numba)")
    parser.add_argument('--T',                     type=int,   default=18550395,
                        help="Total timesteps (default: one full year at 1.7 s/step)")
    parser.add_argument('--seconds_per_time_step', type=float, default=1.7,
                        help="Duration of each timestep in seconds")
    parser.add_argument('--simulation_runs',       type=int,   default=1000,
                        help="Number of independent tortoise trajectories")
    parser.add_argument('--batch_size',            type=int,   default=100000,
                        help="Tortoises per kernel launch (tune to fit GPU VRAM)")
    parser.add_argument('--Delta_T',               type=int,   default=530,
                        help="Interval between displacement samples (timesteps)")
    parser.add_argument('--MSD_Diario', type=str2bool, nargs='?', const=True, default=False,
                        help="Intra-day MSD: one reference per day at --msd_start_hour, "
                             "lags up to --msd_end_hour (e.g. '--MSD_Diario True')")
    parser.add_argument('--msd_start_hour',  type=float, default=9.0,
                        help="Clock hour of the daily MSD reference (default 9)")
    parser.add_argument('--msd_end_hour',    type=float, default=19.0,
                        help="Clock hour at which the daily MSD window ends (default 19)")
    parser.add_argument('--msd_burnin_days', type=int,   default=5,
                        help="Days skipped at the start of each stage before taking references")
    parser.add_argument('--msd_every_days',  type=int,   default=1,
                        help="Use one reference day out of every N (reduces memory)")
    parser.add_argument('--bool_Lognormal',        action='store_true', default=False,
                        help='Enable lognormal (default: disabled)')
    parser.add_argument('--bool_Powerlaw',         action='store_true', default=False,
                        help='Enable powerlaw distribution')
    parser.add_argument('--Max_time_mov', type=int, default=5930,
                        help='Maximum walk bout duration in timesteps (default 5930 ≈ 2.8 hours at 1.7 s/step)')
    parser.add_argument('--Max_time_freeze', type=int, default=58871,
                        help='Maximum freeze bout duration in timesteps (default 58871 ≈ 28 hours at 1.7 s/step)')
    parser.add_argument('--bool_constant_walk_time', action='store_true', default=False,
                        help='Use constant walk time')
    parser.add_argument('--bool_constant_stop_time', action='store_true', default=False,
                        help='Use constant stop time')
    parser.add_argument('--value_constant_walk_time', type=int, default=50,
                        help='Value for constant walk time (if enabled)')
    parser.add_argument('--value_constant_stop_time', type=int, default=50,
                        help='Value for constant stop time (if enabled)')
    parser.add_argument('--mu_vel',                type=float, default=0.82)
    parser.add_argument('--sigma_vel',             type=float, default=0.83)
    parser.add_argument('--angular_sigma',         type=float, default=0.5)
    parser.add_argument('--Constant_step_length',  type=float, default=1.0)
    parser.add_argument('--alpha_walk',            type=float, default=-2.0)
    parser.add_argument('--alpha_stop',            type=float, default=-2.0)
    parser.add_argument('--day_exploration_end',   type=int,   default=30)
    parser.add_argument('--day_revisit_end',   type=int,   default=90)
    parser.add_argument('--day_brumation_end',     type=int,   default=304)
    parser.add_argument('--e_theta',               type=float, default=np.pi / 36, 
                        help="Maximum heading correction per step (default 5 degrees)")
    parser.add_argument('--n_hist_bins',           type=int,   default=100,
                        help="Number of bins for delta-distance histograms")
    parser.add_argument('--traj_every',            type=int,   default=1059,
                        help="Ticks between trajectory samples for the example "
                             "tortoise (default 2118 ≈ 1 hour at 1.7 s/step)")
    parser.add_argument('--stage2_radius_fraction', type=float, default=0.5,
                        help="Stage 2 search radius as a fraction of the calibrated "
                             "stage 1 radius (default 0.5 = half the daily range)")
    parser.add_argument('--folder_name',           type=str,   default='GPU_Simulations')
    parser.add_argument('--sub_folder_name',       type=str,   default='LogNormal_GPU')
    args = parser.parse_args()

    if args.seconds_per_time_step >= 24*60*60:
        raise ValueError("seconds_per_time_step must be less than one day.")

    main(args)