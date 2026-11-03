"""
================================================================================
Patagonian Tortoise Yearly Cycle Simulation -- CPU version ("nightly reset" model)
================================================================================

OVERVIEW
--------
Simulates the 2D movement of Patagonian tortoises (Chelonoidis chilensis) over
a full yearly cycle as a stop-and-go correlated random walk. Each tortoise is
simulated independently for T timesteps (default T = 18,550,395 ticks = 365
days at 1.7 s per tick). The time loop is compiled to machine code with Numba
(@njit), and the N_SIM tortoises are simulated one after another on the CPU.

UNITS
-----
  Positions and step lengths : centimetres (cm)
  Time                       : ticks (1 tick = seconds_per_time_step, default 1.7 s)

MOVEMENT RULES
--------------
  Step length : LogNormal(mu_vel, sigma_vel), redrawn until <= data_max (5.67 cm).
  Heading     : previous heading + Gaussian noise of s.d. angular_sigma (rad),
                i.e. a correlated random walk (CRW).
  Guided walk : in stages 1 and 2 the heading first turns towards the target
                by at most e_theta = 5 degrees per tick, then noise is added.
  Bouts       : walking and stopping alternate. Bout durations are drawn from
                truncated power laws P(t) ~ t^alpha (alpha_walk, alpha_stop),
                capped by Max_time_mov / Max_time_freeze AND by the time left
                until 20:00.

DAILY RHYTHM -- RESET MODEL
---------------------------
  Night (20:00 - 08:00) : no movement.
  Dawn  (08:00)         : behaviour is reset. The tortoise always starts the
                          day walking, with a newly drawn walk bout.
  Day   (08:00 - 20:00) : movement follows the stage rules below.

  Because every bout is capped to finish before 20:00, no bout ever spans a
  night: each day starts from scratch. This is the key difference from the GPU
  version (GPU_FINAL_Noches_v2.py), in which rest bouts can continue through
  the night.

YEARLY STAGES  (calendar day = (t // ticks_per_day) % 365)
--------------------------------------------------------
  Stage 0 -- Exploration, days [0, day_exploration_end) and [day_brumation_end, 365):
    Free CRW with walk/stop bouts. At 20:00 every night the sleeping position
    is stored as a refuge ("attractor"). The origin (0, 0) is pre-registered
    as the home refuge.

  Stage 1 -- Intermediate / homing, days [day_exploration_end, day_brumation_begin):
    Each dawn a target is chosen uniformly among the refuges within
    stage1_radius (the calibrated mean daily displacement, see CALIBRATION);
    if none qualify, any refuge is chosen. The tortoise performs a guided walk
    towards it and, once within 30 cm of the target, rests for the rest of
    the day.

  Stage 2 -- Pre-brumation, days [day_brumation_begin, day_brumation_end):
    On the first tick of the stage a shortlist of up to 5 nearest refuges
    within stage2_radius = stage2_radius_fraction x stage1_radius is built
    (fallback: all refuges). Each dawn one of them is chosen at random and the
    same guided walk and rest-on-arrival rule as in stage 1 is applied.
    Note: movement is not switched off during brumation itself; stage 2
    represents homing to a small set of winter refuges.

CALIBRATION OF THE STAGE-1 RADIUS
---------------------------------
  Before the main simulation, 500 replicas of one active day are run with the
  same walk/stop rules and a guided walk towards a very distant target. The
  mean net displacement is used as stage1_radius, so the target-selection
  radius is consistent with the movement parameters in use.

MSD -- MEMORY-EFFICIENT ENSEMBLE ESTIMATE
-----------------------------------------
  Storing full trajectories (N_SIM x T x 2) would require ~300 GB for 1,000
  tortoises over one year, so
  positions are stored only at selected "snapshot" ticks:
    - n_refs = 48 reference times per stage, 30 min apart, covering day 2 of
      the stage (from 00:00), so the transient from (0, 0) has decayed;
    - lags tau_k = round(1.5^k) until the end of the stage.
  MSD(tau) = mean over tortoises and reference times of |r(t_ref + tau) - r(t_ref)|^2
  SEM(tau) = sqrt(var(r^2) / N), with N = N_SIM x n_refs pooled pairs.
  Note: pairs from the same tortoise are correlated, so this SEM is a lower
  bound. The GPU version averages over reference times per tortoise first.

DISPLACEMENT HISTOGRAMS
-----------------------
  Every Delta_T ticks during daytime, the distance moved since the previous
  sample is stored for the current stage. Night ticks are skipped and zero
  displacements are discarded, so the histogram describes movement only.
  Fixed bin edges on [0, data_max x Delta_T] make CPU and GPU histograms
  directly comparable.

OUTPUT FILES  (written to folder_name/sub_folder_name/)
-------------------------------------------------------
  snapshot_times.npy                    -- absolute ticks of all snapshots
  t_refs.csv                            -- first reference tick of each stage
  taus_{stage}.csv                      -- lag values (ticks) for each stage
  msd_{stage}.csv                       -- columns: msd, sem
  Delta_distances_histogram_{stage}.csv -- columns: distance, frequency, poisson_error
  Attractors.txt                        -- refuge positions of the first tortoise
  example_trajectory.npy                -- (n_points, 2) x/y of the first tortoise,
                                           sampled every traj_every ticks (~1 h)
  example_attractors.npy                -- (n_attractors, 2) refuges of the first tortoise
  where {stage} is exploration, intermediate or prebrumation.

USAGE
-----
  python LogNormal_CPU_FINAL.py --T 18550395 --simulation_runs 1000 [options]
  See the command-line interface at the bottom for all arguments.

DEPENDENCIES
------------
  numpy, numba
================================================================================
"""

import numpy as np
import os
import time
import argparse
from numba import njit
from numba.typed import List


# ==============================================================================
# GLOBAL PARAMETERS
# Default values — all can be overridden via CLI arguments.
# ==============================================================================

data_max        = 5.6667   # maximum allowed step length in cm (lognormal rejection cap)
angular_sigma   = 0.5      # std deviation of angular noise in radians
sigma_vel       = 0.83     # lognormal sigma for step-length distribution
mu_vel          = 0.82     # lognormal mu for step-length distribution
simulation_runs = 1000     # number of independent tortoise trajectories
Delta_T         = 500      # timestep interval between displacement samples

# Walk/stop duration limits in ticks (at 1.7 s per tick).
# Max_time_mov    ≈ 2.8 hours   (upper bound for a single walk bout)
# Max_time_freeze ≈ 27.8 hours  (upper bound for a single stop bout; in practice
#                                 stops are also capped by the time left until 20:00)
Max_time_mov    = 5930
Max_time_freeze = 58871

# Temporal discretisation
seconds_per_time_step = 1.7   # seconds per simulation timestep

# Yearly stage boundaries in calendar days (day 0 = 1 January); each stage
# includes its first day and excludes its last.
day_exploration_end  = 30    # first day of stage 1 (end of exploration)
day_brumation_begin  = 90    # first day of stage 2 (pre-brumation homing)
day_brumation_end    = 304   # first day after stage 2; exploration resumes

# Derived temporal constants, used inside the @njit functions.
# Note: these use the default 1.7 s per tick; main() recomputes them locally
# for the calibration and snapshot grid.
night_begins     = int(20*60*60 / seconds_per_time_step)   # 8 PM in ticks
night_ends       = int( 8*60*60 / seconds_per_time_step)   # 8 AM in ticks
tic_per_day      = int(24*60*60 / seconds_per_time_step)   # ticks per 24 h

# Maximum angular correction toward a target per step (5 degrees)
Mean_Angular_Var = np.pi / 36


# ==============================================================================
# NUMBA JIT FUNCTIONS
# Functions decorated with @njit are compiled to machine code on their first
# call, so later calls run at compiled speed.
# ==============================================================================

@njit
def powerlaw_stop_interval(alpha=-2, min_time=1, max_time=1000):
    """
    Draw a random duration from a truncated power-law distribution P(x) ~ x^alpha
    on the interval [min_time, max_time] using the inverse-CDF method.
    Used for both walk and stop bouts (the name is historical).

    Special case alpha == -1: P(x) ~ 1/x, i.e. a log-uniform distribution.
    """
    u = np.random.uniform(0, 1)
    if alpha == -1:
        return min_time * (max_time / min_time) ** u
    else:
        return ((max_time**(alpha+1) - min_time**(alpha+1)) * u
                + min_time**(alpha+1)) ** (1.0 / (alpha+1))


@njit
def gaussian_angle_lognormal_step(r0x, r0y, theta0, mu_vel, sigma_r, angular_sigma):
    """
    Advance position by one correlated random-walk step.

    The new heading is sampled from Normal(theta0, angular_sigma), i.e. the
    previous heading plus noise, and wrapped to [-pi, pi]. The step length (cm)
    is drawn from LogNormal(mu_vel, sigma_r), redrawn until it is <= data_max.

    Returns new (x, y, heading).
    """
    new_angle   = np.random.normal(theta0, angular_sigma)
    new_angle   = (new_angle + np.pi) % (2*np.pi) - np.pi   # wrap to [-pi, pi]
    step_length = np.random.lognormal(mu_vel, sigma_r)
    while step_length > data_max:
        step_length = np.random.lognormal(mu_vel, sigma_r)
    x = step_length * np.cos(new_angle) + r0x
    y = step_length * np.sin(new_angle) + r0y
    return x, y, new_angle


@njit
def guided_walker(x_0, y_0, theta_0, e_theta, target_point, mu_vel, sigma_r, angular_sigma):
    """
    Advance position by one biased random-walk step toward a target.

    The heading correction toward target_point is clamped to ±e_theta per step,
    then Gaussian noise (same sigma as the free walk) is added. Step length
    follows the same lognormal distribution as the free walk.

    Returns new (x, y, heading).
    """
    # Compute the bearing from current position to target
    target_angle = np.arctan2(target_point[1] - y_0, target_point[0] - x_0)

    # Shortest angular difference wrapped to [-pi, pi]
    delta = (target_angle - theta_0 + np.pi) % (2*np.pi) - np.pi

    # Clamp the steering correction to the maximum allowed turn per step
    if delta > e_theta:
        turn = e_theta
    elif delta < -e_theta:
        turn = -e_theta
    else:
        turn = delta

    mean_angle  = theta_0 + turn
    new_angle   = np.random.normal(mean_angle, angular_sigma)
    new_angle   = (new_angle + np.pi) % (2*np.pi) - np.pi
    step_length = np.random.lognormal(mu_vel, sigma_r)
    while step_length > data_max:
        step_length = np.random.lognormal(mu_vel, sigma_r)
    x_new = x_0 + step_length * np.cos(new_angle)
    y_new = y_0 + step_length * np.sin(new_angle)
    return x_new, y_new, new_angle


@njit
def create_attraction_points(pos):
    """
    Register a position as a refuge (attractor) with a random fitness value
    drawn from Uniform(0, 1). Returns a (position_array, fitness) tuple.
    Note: the fitness value is stored but not currently used in target choice.
    """
    point     = pos
    fitness   = np.random.random()
    attractor = (point, fitness)
    return attractor


@njit
def in_burrow(x, y, ref):
    """
    Return True if (x, y) is within 30 cm of the attractor referenced by ref.
    Uses squared distance (threshold = 900 cm^2) to avoid the square root.
    """
    ref_x = ref[0][0]
    ref_y = ref[0][1]
    return (x - ref_x)**2 + (y - ref_y)**2 < 900


@njit
def simulate_single_run(T, mu_vel, sigma_vel, angular_sigma, alpha_stop, alpha_walk,
                         Delta_T, stages, snapshot_times, n_snaps,
                         traj_x, traj_y, traj_every, save_traj,
                         stage1_radius2, stage2_radius2):
    """
    Simulate one tortoise for T timesteps.

    Parameters
    ----------
    T              : total number of timesteps
    mu_vel         : lognormal mu for step length
    sigma_vel      : lognormal sigma for step length
    angular_sigma  : std of angular noise (radians)
    alpha_stop     : power-law exponent for stop durations
    alpha_walk     : power-law exponent for walk durations
    Delta_T        : interval between displacement samples (ticks)
    stages         : int64 array of length T — yearly stage at each tick
    snapshot_times : int64 array — absolute timesteps at which to record position
    n_snaps        : length of snapshot_times
    traj_x         : float64 array of length (T // traj_every + 1) — written when
                     save_traj is True, otherwise ignored (pass a size-1 dummy)
    traj_y         : same as traj_x for the y coordinate
    traj_every     : int — ticks between trajectory samples
    save_traj      : bool — True only for the first run (run 0)
    stage1_radius2 : squared radius (cm^2) for stage-1 target selection
    stage2_radius2 : squared radius (cm^2) for the stage-2 shortlist

    Returns
    -------
    snap_x, snap_y           : float64 arrays, shape (n_snaps,)
    delta_distances_{0,1,2}  : float64 arrays, displacement samples per stage
    attr_arr                 : (N_attractors, 2) float64 array of attractor xy
    missed_burrow_days       : number of stage-1/2 days the target was not reached
    """

    # The home burrow at the origin is pre-registered as the first attractor
    Attraction_points = [(np.array([0.0, 0.0]), 0.0)]

    # Initial position and a random heading to break symmetry
    x     = 0.0
    y     = 0.0
    angle = np.random.uniform(-np.pi, np.pi)

    # Walk / stop state machine variables
    WALK            = True             # True = walking, False = stopped
    count_walk_time = 0                # ticks elapsed in the current walk bout
    count_stop_time = 0                # ticks elapsed in the current stop bout
    walk_time       = float(Max_time_mov)   # duration of the current walk bout
    stop_time       = 0.0                   # duration of the current stop bout

    # Stage 2 bookkeeping
    first_brumation_step = True    # triggers nearest-attractor search on stage entry
    arrived_at_burrow    = False   # True once daily target is reached
    atractor_selector    = 0       # index of the current daily target attractor

    # Count of days in stages 1/2 where the tortoise never reached its target burrow
    missed_burrow_days = 0

    # Output arrays for snapshot positions
    snap_x   = np.zeros(n_snaps)
    snap_y   = np.zeros(n_snaps)
    snap_ptr = 0   # forward pointer into the sorted snapshot_times array

    # Candidate buffer for stage-1 uniform-within-radius selection (size 50).
    # In the first simulated year at most 31 refuges exist when stage 1 runs
    # (origin + one per exploration night). Simulations longer than one year
    # accumulate more refuges and would need a larger buffer.
    stage1_candidates = np.empty(50, dtype=np.int64)

    # Per-stage displacement sample arrays (over-allocated; sliced on return)
    delta_distances_0 = np.zeros(T // Delta_T + 2)   # exploration
    delta_distances_1 = np.zeros(T // Delta_T + 2)   # intermediate
    delta_distances_2 = np.zeros(T // Delta_T + 2)   # pre-brumation
    delta_index_0     = 0
    delta_index_1     = 0
    delta_index_2     = 0
    old_x, old_y      = 0.0, 0.0   # reference position for displacement sampling

    # Pre-initialise the stage-2 shortlist so Numba can infer the List type
    # before the first brumation step actually builds it
    Closest_Attraction_points = List()
    Closest_Attraction_points.append(Attraction_points[0])

    for t in range(T):

        stage       = stages[t]
        time_in_day = t % tic_per_day

        # ======================================================================
        # NIGHT  (20:00 – 08:00)
        # The tortoise does not move at night. At exactly 20:00 in stage 0 the
        # current sleeping position is registered as a new attractor. Snapshots
        # are still recorded if this tick is a designated snapshot time.
        # Displacements are not sampled at night because the position is frozen.
        # ======================================================================
        if time_in_day >= night_begins or time_in_day < night_ends:

            if time_in_day == night_begins and stage == 0:
                Attraction_points.append(create_attraction_points(np.array([x, y])))

            # Count days in stages 1/2 where the tortoise never reached its burrow
            if time_in_day == night_begins and stage > 0 and not arrived_at_burrow:
                missed_burrow_days += 1

            while snap_ptr < n_snaps and snapshot_times[snap_ptr] == t:
                snap_x[snap_ptr] = x
                snap_y[snap_ptr] = y
                snap_ptr += 1

            continue   # skip all movement and accumulation for this tick

        # ======================================================================
        # DAWN RESET  (exactly 08:00 each day) -- the "nightly reset"
        # Whatever the tortoise did the previous evening, it starts the day
        # walking: counters are reset and a fresh walk bout is drawn, capped at
        # min(Max_time_mov, length of the daytime window). For stages 1 and 2
        # the daily target is selected further below, at the same tick.
        # ======================================================================
        if time_in_day == night_ends:
            WALK              = True
            count_stop_time   = 0
            count_walk_time   = 0
            stop_time         = 0.0
            arrived_at_burrow = False   # reset: the day starts without a completed target

            remaining_day_dawn = night_begins - night_ends
            max_walk_dawn = remaining_day_dawn
            if max_walk_dawn > Max_time_mov:
                max_walk_dawn = Max_time_mov
            walk_time = powerlaw_stop_interval(alpha_walk, min_time=1, max_time=max_walk_dawn)

        # ======================================================================
        # DAY  (08:00 – 20:00)
        # remaining_day shrinks toward zero as dusk approaches and is used to
        # cap the duration of any new walk or stop bout drawn mid-day, preventing
        # bouts from running past 20:00.
        # ======================================================================
        remaining_day = night_begins - time_in_day

        # ----------------------------------------------------------------------
        # STAGE 0 — Exploration: correlated random walk with walk/stop bouts
        # ----------------------------------------------------------------------
        if stage == 0:
            if WALK:
                if (walk_time - count_walk_time) > 0:
                    # Still within the walk bout: take one step
                    x, y, angle = gaussian_angle_lognormal_step(
                        x, y, angle, mu_vel, sigma_vel, angular_sigma)
                    count_walk_time += 1
                else:
                    # Walk bout finished: transition to stop and draw a new duration
                    max_stop = remaining_day
                    if max_stop > Max_time_freeze:
                        max_stop = Max_time_freeze
                    stop_time       = powerlaw_stop_interval(
                        alpha_stop, min_time=1, max_time=max_stop)
                    WALK            = False
                    count_walk_time = 0
            else:
                if (stop_time - count_stop_time) > 0:
                    # Still in the stop bout: remain stationary
                    count_stop_time += 1
                else:
                    # Stop bout finished: draw a new walk duration and resume
                    max_walk = remaining_day
                    if max_walk > Max_time_mov:
                        max_walk = Max_time_mov
                    walk_time       = powerlaw_stop_interval(
                        alpha_walk, min_time=1, max_time=max_walk)
                    WALK            = True
                    count_stop_time = 0

        # ----------------------------------------------------------------------
        # STAGE 1 — Intermediate: biased walk toward a randomly chosen attractor
        # ----------------------------------------------------------------------
        elif stage == 1:

            # Select today's target at dawn: pick uniformly from ALL attractors
            # within the calibrated radius (mean daily displacement).
            # Fallback to the full list if none qualify.
            if time_in_day == night_ends:
                n_cand = 0
                for i in range(len(Attraction_points)):
                    pt = Attraction_points[i][0]
                    dx = pt[0] - x
                    dy = pt[1] - y
                    if dx*dx + dy*dy <= stage1_radius2:
                        stage1_candidates[n_cand] = i
                        n_cand += 1
                if n_cand > 0:
                    atractor_selector = stage1_candidates[np.random.randint(0, n_cand)]
                else:
                    # Fallback: no burrow within radius — pick any
                    atractor_selector = np.random.randint(0, len(Attraction_points))
                arrived_at_burrow = in_burrow(x, y, Attraction_points[atractor_selector])

            # Once the burrow is reached the tortoise rests for the rest of the day
            if not arrived_at_burrow:
                if WALK:
                    if (walk_time - count_walk_time) > 0:
                        x, y, angle = guided_walker(
                            x, y, angle, Mean_Angular_Var,
                            Attraction_points[atractor_selector][0],
                            mu_vel, sigma_vel, angular_sigma)
                        count_walk_time  += 1
                        arrived_at_burrow = in_burrow(x, y, Attraction_points[atractor_selector])
                    else:
                        max_stop = remaining_day
                        if max_stop > Max_time_freeze:
                            max_stop = Max_time_freeze
                        stop_time       = powerlaw_stop_interval(
                            alpha_stop, min_time=1, max_time=max_stop)
                        WALK            = False
                        count_walk_time = 0
                else:
                    if (stop_time - count_stop_time) > 0:
                        count_stop_time += 1
                    else:
                        max_walk = remaining_day
                        if max_walk > Max_time_mov:
                            max_walk = Max_time_mov
                        walk_time       = powerlaw_stop_interval(
                            alpha_walk, min_time=1, max_time=max_walk)
                        WALK            = True
                        count_stop_time = 0

        # ----------------------------------------------------------------------
        # STAGE 2 — Pre-brumation: homing toward up to 5 nearest attractors
        # ----------------------------------------------------------------------
        else:

            if first_brumation_step:
                # On the first tick of stage 2, build the shortlist of up to 5
                # nearest attractors within stage2_radius. If none qualify, fall
                # back to the full list so the tortoise always has a valid target.
                Closest_Attraction_points = List()
                best_distances = np.full(5, 1e18)
                best_points    = List()
                for _ in range(5):
                    best_points.append(Attraction_points[0])
                count = 0
                for i in range(len(Attraction_points)):
                    pt    = Attraction_points[i][0]
                    dx    = pt[0] - x
                    dy    = pt[1] - y
                    dist2 = dx*dx + dy*dy
                    if dist2 <= stage2_radius2:   # fraction of the calibrated daily range
                        for j in range(5):
                            if dist2 < best_distances[j]:
                                # Insert into sorted top-5 list (shift elements right)
                                for k in range(4, j, -1):
                                    best_distances[k] = best_distances[k-1]
                                    best_points[k]    = best_points[k-1]
                                best_distances[j] = dist2
                                best_points[j]    = Attraction_points[i]
                                if count < 5:
                                    count += 1
                                break
                for i in range(count):
                    Closest_Attraction_points.append(best_points[i])
                if len(Closest_Attraction_points) == 0:
                    for i in range(len(Attraction_points)):
                        Closest_Attraction_points.append(Attraction_points[i])
                first_brumation_step = False
                atractor_selector    = np.random.randint(0, len(Closest_Attraction_points))

            # Re-select from the shortlist every dawn
            if time_in_day == night_ends:
                atractor_selector = np.random.randint(0, len(Closest_Attraction_points))
                arrived_at_burrow = in_burrow(x, y, Closest_Attraction_points[atractor_selector])

            # Same burrow-stop logic as stage 1; heading bias identical to stage 1.
            if not arrived_at_burrow:
                if WALK:
                    if (walk_time - count_walk_time) > 0:
                        x, y, angle = guided_walker(
                            x, y, angle, Mean_Angular_Var,
                            Closest_Attraction_points[atractor_selector][0],
                            mu_vel, sigma_vel, angular_sigma)
                        count_walk_time  += 1
                        arrived_at_burrow = in_burrow(x, y, Closest_Attraction_points[atractor_selector])
                    else:
                        max_stop = remaining_day
                        if max_stop > Max_time_freeze:
                            max_stop = Max_time_freeze
                        stop_time       = powerlaw_stop_interval(
                            alpha_stop, min_time=1, max_time=max_stop)
                        WALK            = False
                        count_walk_time = 0
                else:
                    if (stop_time - count_stop_time) > 0:
                        count_stop_time += 1
                    else:
                        max_walk = remaining_day
                        if max_walk > Max_time_mov:
                            max_walk = Max_time_mov
                        walk_time       = powerlaw_stop_interval(
                            alpha_walk, min_time=1, max_time=max_walk)
                        WALK            = True
                        count_stop_time = 0

        # ======================================================================
        # EXAMPLE TRAJECTORY SAMPLE
        # Recorded only for the first run (save_traj=True). Saves position at
        # every traj_every ticks so the output is compact but still captures
        # the full spatial path for visualisation.
        # ======================================================================
        if save_traj and t % traj_every == 0:
            traj_idx = t // traj_every
            traj_x[traj_idx] = x
            traj_y[traj_idx] = y

        # ======================================================================
        # SNAPSHOT CHECK
        # snapshot_times is sorted, so we advance a single pointer and use a
        # while-loop to handle the rare case of two snapshots at the same tick.
        # ======================================================================
        while snap_ptr < n_snaps and snapshot_times[snap_ptr] == t:
            snap_x[snap_ptr] = x
            snap_y[snap_ptr] = y
            snap_ptr += 1

        # ======================================================================
        # DELTA DISTANCE SAMPLE (daytime ticks only)
        # Every Delta_T ticks compute the displacement from the previous sample
        # and store it in the array for the current stage. The reference position
        # old_x/old_y is shared across stages: a sample straddling a stage
        # boundary is attributed to the stage at time t (the new stage).
        # Zero displacements are removed later, when the histogram is built.
        # ======================================================================
        if t % Delta_T == 0:
            dist = np.sqrt((x - old_x)**2 + (y - old_y)**2)
            if stage == 0:
                delta_distances_0[delta_index_0] = dist
                delta_index_0 += 1
            elif stage == 1:
                delta_distances_1[delta_index_1] = dist
                delta_index_1 += 1
            else:
                delta_distances_2[delta_index_2] = dist
                delta_index_2 += 1
            old_x, old_y = x, y   # advance the reference position

    # Pack attractor coordinates into a plain 2D array for saving
    attr_arr = np.zeros((len(Attraction_points), 2))
    for i in range(len(Attraction_points)):
        attr_arr[i, 0] = Attraction_points[i][0][0]
        attr_arr[i, 1] = Attraction_points[i][0][1]

    return (snap_x, snap_y,
            delta_distances_0[:delta_index_0],
            delta_distances_1[:delta_index_1],
            delta_distances_2[:delta_index_2],
            attr_arr,
            missed_burrow_days)


# ==============================================================================
# CPU HELPER FUNCTIONS
# ==============================================================================

@njit
def calibrate_stage1_radius(mu_vel, sigma_vel, angular_sigma, alpha_walk, alpha_stop,
                              night_ends_c, night_begins_c, n_reps=500):
    """
    Estimate the mean net displacement a tortoise can achieve in one full active
    day, with actual walk/stop power-law alternation and a guided walk toward a
    target placed infinitely far away (never reached, so the tortoise always
    walks in the same direction when the walk state is active).

    Including stops is essential: the distribution of walk/stop durations
    directly determines what fraction of the day the tortoise is moving, so the
    calibrated radius automatically reflects the current alpha_walk / alpha_stop
    parameters.

    The walk/stop rules match this CPU simulation: each replica starts walking
    at dawn and no bout extends past 20:00.

    Returns the mean end-of-day displacement in cm — used directly as
    stage1_radius so the filter is self-consistent with the simulation dynamics.
    """
    total_disp = 0.0
    e_theta    = np.pi / 36.0
    target_x   = 1.0e9    # effectively infinite — never reached
    target_y   = 0.0

    for _ in range(n_reps):
        x     = 0.0
        y     = 0.0
        angle = np.random.uniform(-np.pi, np.pi)

        WALK            = True
        count_walk_time = 0
        count_stop_time = 0

        remaining_day_dawn = night_begins_c - night_ends_c
        max_walk_dawn = remaining_day_dawn
        if max_walk_dawn > Max_time_mov:
            max_walk_dawn = Max_time_mov
        walk_time = powerlaw_stop_interval(alpha_walk, min_time=1, max_time=max_walk_dawn)
        stop_time = 0.0

        for time_in_day in range(night_ends_c, night_begins_c):
            remaining_day = night_begins_c - time_in_day

            if WALK:
                if (walk_time - count_walk_time) > 0:
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
                    step_length = np.random.lognormal(mu_vel, sigma_vel)
                    while step_length > data_max:
                        step_length = np.random.lognormal(mu_vel, sigma_vel)
                    x    += step_length * np.cos(new_angle)
                    y    += step_length * np.sin(new_angle)
                    angle = new_angle
                    count_walk_time += 1
                else:
                    WALK            = False
                    count_walk_time = 0
                    max_stop = remaining_day
                    if max_stop > Max_time_freeze:
                        max_stop = Max_time_freeze
                    stop_time = powerlaw_stop_interval(alpha_stop, min_time=1, max_time=max_stop)
            else:
                if (stop_time - count_stop_time) > 0:
                    count_stop_time += 1
                else:
                    WALK            = True
                    count_stop_time = 0
                    max_walk = remaining_day
                    if max_walk > Max_time_mov:
                        max_walk = Max_time_mov
                    walk_time = powerlaw_stop_interval(alpha_walk, min_time=1, max_time=max_walk)

        total_disp += np.sqrt(x**2 + y**2)

    return total_disp / n_reps


def compute_stages_cpu(T, seconds_per_time_step,
                        day_exploration_end, day_brumation_begin, day_brumation_end):
    """
    Build an integer stage-index array of length T.
    The yearly calendar wraps modulo 365 so multi-year simulations are supported.
    Values: 0 = exploration, 1 = intermediate, 2 = pre-brumation.
    Days after day_brumation_end map back to stage 0 (exploration resumes).
    """
    tic_per_day = int(24*60*60 / seconds_per_time_step)
    stages = np.zeros(T, dtype=np.int64)
    for t in range(T):
        day = (t // tic_per_day) % 365
        if   day < day_exploration_end: stages[t] = 0
        elif day < day_brumation_begin: stages[t] = 1
        elif day < day_brumation_end:   stages[t] = 2
        else:                           stages[t] = 0   # post-brumation = exploration
    return stages


def compute_snapshot_times(T, tic_per_day, night_ends_val,
                            day_exploration_end, day_brumation_begin, day_brumation_end,
                            seconds_per_time_step,
                            n_refs=48, ref_spacing_minutes=30, base=1.5):
    """
    Precompute snapshot timesteps for an ensemble MSD averaged over multiple
    reference times. For each stage, n_refs reference times are placed on day 2
    of the stage, starting at 00:00, spaced ref_spacing_minutes apart (default
    48 x 30 min = 24 h). Lags are tau_k = round(base^k). The lag grid is the
    same for all reference times within a stage, capped by the last (most
    restrictive) reference time so every reference has the full lag range.

    Returns
    -------
    snapshot_times    : sorted int64 array of all unique absolute timesteps
    t_refs_per_stage  : list of 3 int64 arrays of shape (n_refs,)
    taus_per_stage    : list of 3 int64 arrays of lags (same for all refs)
    ref_snap_idxs     : list of 3 int32 arrays of shape (n_refs,)
    lag_snap_idxs     : list of 3 int32 arrays of shape (n_refs, n_lags)
    """
    ref_spacing_ticks = int(ref_spacing_minutes * 60 / seconds_per_time_step)

    stage_day_starts = [0,                  day_exploration_end, day_brumation_begin]
    stage_day_ends   = [day_exploration_end, day_brumation_begin, day_brumation_end  ]

    t_refs_per_stage = []
    taus_per_stage   = []
    all_snaps        = set()

    for d_start, d_end in zip(stage_day_starts, stage_day_ends):
        stage_t_start = d_start * tic_per_day
        stage_t_end   = min(d_end * tic_per_day, T)

        t_ref_0 = stage_t_start + tic_per_day
        t_refs  = np.array([t_ref_0 + k * ref_spacing_ticks for k in range(n_refs)],
                            dtype=np.int64)
        t_refs_per_stage.append(t_refs)

        for t_ref in t_refs:
            all_snaps.add(int(t_ref))

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

    snapshot_times = np.array(sorted(all_snaps), dtype=np.int64)
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


def accumulate_msd_single(snap_x, snap_y, ref_snap_idxs, lag_snap_idxs, taus_per_stage,
                           msd_sum_r2, msd_sum_r4, msd_count):
    """
    Accumulate MSD statistics from one run's snapshot positions.

    snap_x, snap_y : 1D float64 arrays of shape (n_snaps,)
    Modifies msd_sum_r2, msd_sum_r4, msd_count in-place. Every (tortoise,
    reference time) pair is counted as one sample.
    """
    for s in range(3):
        r_idxs = ref_snap_idxs[s]           # (n_refs,)
        l_idxs = lag_snap_idxs[s]           # (n_refs, n_lags)
        n_lags = len(taus_per_stage[s])
        n_refs = len(r_idxs)

        ref_x = snap_x[r_idxs]   # (n_refs,)
        ref_y = snap_y[r_idxs]

        for j in range(n_lags):
            lag_x = snap_x[l_idxs[:, j]]   # (n_refs,)
            lag_y = snap_y[l_idxs[:, j]]
            r2    = (lag_x - ref_x)**2 + (lag_y - ref_y)**2   # (n_refs,)

            msd_sum_r2[s][j] += np.sum(r2)
            msd_sum_r4[s][j] += np.sum(r2**2)
            msd_count[s][j]  += n_refs


def finalize_msd(msd_sum_r2, msd_sum_r4, msd_count, taus_per_stage):
    """
    Compute final MSD and SEM from accumulated sums.
    SEM is derived from the variance of r²: var(r²) = E[r⁴] - E[r²]².
    Because all (tortoise, reference) pairs are pooled and pairs from the same
    tortoise are correlated, this SEM is a lower bound.
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


def delta_distances_to_histogram(all_delta_distances, max_dist, n_bins=100):
    """
    Convert a flat array of raw displacement samples into a normalised frequency
    histogram using fixed bin edges [0, max_dist].

    frequencies : count_i / total  — fraction of samples in each bin
    errors      : sqrt(count_i) / total  — Poisson uncertainty

    Fixed bin edges are used so CPU and GPU histograms are directly comparable
    regardless of the number of tortoises simulated.
    Returns (None, None, None) if the input array is empty after zero filtering.
    """
    arr = np.array(all_delta_distances)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return None, None, None

    edges       = np.linspace(0.0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _   = np.histogram(arr, bins=edges)
    N_total     = len(arr)
    frequencies = counts / N_total
    errors      = np.sqrt(counts) / N_total

    return bin_centers, frequencies, errors


# ==============================================================================
# MAIN
# ==============================================================================

def main(T, seconds_per_time_step, mean_r, sigma_r, angular_sigma,
         alpha_stop, alpha_walk, simulation_runs, Delta_T,
         experiment_name, subfolder_name, n_hist_bins=100, traj_every=2118,
         stage2_radius_fraction=0.5, n_refs=48, ref_spacing_minutes=30.0):

    start_time     = time.perf_counter()
    tic_per_day    = int(24*60*60 / seconds_per_time_step)
    night_ends_l   = int( 8*60*60 / seconds_per_time_step)
    night_begins_l = int(20*60*60 / seconds_per_time_step)

    # ── Stage 1 radius calibration ────────────────────────────────────────────
    # Simulate walk/stop dynamics for one full day (500 reps) toward a very
    # distant target to estimate the mean reachable displacement (cm).
    print("Calibrating stage 1 radius (500 reps)...")
    stage1_radius_cm = calibrate_stage1_radius(
        mean_r, sigma_r, angular_sigma, alpha_walk, alpha_stop,
        night_ends_l, night_begins_l)
    stage1_radius2   = stage1_radius_cm ** 2
    print(f"  Mean daily displacement = {stage1_radius_cm:.1f} cm = {stage1_radius_cm/100:.2f} m  "
          f"→ stage1_radius = {stage1_radius_cm:.1f} cm")
    stage2_radius2 = stage1_radius2 * (stage2_radius_fraction ** 2)
    print(f"  stage2_radius = stage1_radius × {stage2_radius_fraction} = "
          f"{stage1_radius_cm * stage2_radius_fraction:.1f} cm = "
          f"{stage1_radius_cm * stage2_radius_fraction / 100:.2f} m")

    # Create output directories, creating parent dirs as needed
    os.makedirs(experiment_name, exist_ok=True)
    folder_name = os.path.join(experiment_name, subfolder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Precompute the stage array — identical for all runs
    stages = compute_stages_cpu(
        T, seconds_per_time_step,
        day_exploration_end, day_brumation_begin, day_brumation_end)

    # Precompute snapshot timesteps — identical for all runs
    snapshot_times, t_refs_per_stage, taus_per_stage, ref_snap_idxs, lag_snap_idxs = \
        compute_snapshot_times(
            T, tic_per_day, night_ends_l,
            day_exploration_end, day_brumation_begin, day_brumation_end,
            seconds_per_time_step,
            n_refs=n_refs,
            ref_spacing_minutes=ref_spacing_minutes)

    n_snaps     = len(snapshot_times)
    stage_names = ['exploration', 'intermediate', 'prebrumation']
    print(f"Snapshot times: {n_snaps} total  ({n_refs} refs × 30 min on day 2 per stage)")
    for s, (t_refs, taus) in enumerate(zip(t_refs_per_stage, taus_per_stage)):
        print(f"  Stage {s} ({stage_names[s]}): {len(t_refs)} refs, {len(taus)} lags each")

    # Incremental MSD accumulators — each run is added and then discarded,
    # so no (N_SIM, n_snaps) array is ever stored.
    msd_sum_r2 = [np.zeros(len(taus), dtype=np.float64) for taus in taus_per_stage]
    msd_sum_r4 = [np.zeros(len(taus), dtype=np.float64) for taus in taus_per_stage]
    msd_count  = [np.zeros(len(taus), dtype=np.int64)   for taus in taus_per_stage]

    # Per-stage displacement lists
    all_delta_distances_0 = []
    all_delta_distances_1 = []
    all_delta_distances_2 = []
    total_missed_days     = 0

    # Example trajectory arrays — only written during run 0 (save_traj=True).
    n_traj_pts      = T // traj_every + 1
    traj_x_example  = np.zeros(n_traj_pts)
    traj_y_example  = np.zeros(n_traj_pts)
    dummy_traj      = np.zeros(1)

    # Fixed histogram range [0, data_max * Delta_T] cm — the largest possible
    # displacement between samples. Same convention as the GPU version.
    max_dist  = 5.6667 * Delta_T

    # Numba requires int64 for comparisons inside the @njit kernel
    snap_times_numba = snapshot_times.astype(np.int64)

    for it in range(simulation_runs):

        if it == 0:
            t_x, t_y, t_every, do_traj = traj_x_example, traj_y_example, traj_every, True
        else:
            t_x, t_y, t_every, do_traj = dummy_traj, dummy_traj, traj_every, False

        snap_x, snap_y, delta_0, delta_1, delta_2, attr_arr, missed = simulate_single_run(
            T, mean_r, sigma_r, angular_sigma,
            alpha_stop, alpha_walk,
            Delta_T, stages, snap_times_numba, n_snaps,
            t_x, t_y, t_every, do_traj,
            stage1_radius2, stage2_radius2)

        # Accumulate MSD from this run immediately — no large snapshot array needed
        accumulate_msd_single(snap_x, snap_y, ref_snap_idxs, lag_snap_idxs,
                               taus_per_stage, msd_sum_r2, msd_sum_r4, msd_count)

        all_delta_distances_0.extend(delta_0.tolist())
        all_delta_distances_1.extend(delta_1.tolist())
        all_delta_distances_2.extend(delta_2.tolist())
        total_missed_days += missed

        if it == 0:
            np.savetxt(os.path.join(folder_name, 'Attractors.txt'),
                       attr_arr, delimiter=',')
            traj_out = np.column_stack([traj_x_example, traj_y_example])
            np.save(os.path.join(folder_name, 'example_trajectory.npy'), traj_out)
            np.save(os.path.join(folder_name, 'example_attractors.npy'), attr_arr)

        if (it + 1) % 100 == 0:
            print(f"  Completed {it+1}/{simulation_runs} runs")

    # Finalize MSD from accumulated sums
    msd_results = finalize_msd(msd_sum_r2, msd_sum_r4, msd_count, taus_per_stage)

    # --------------------------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------------------------

    np.save(os.path.join(folder_name, 'snapshot_times.npy'), snapshot_times)

    # Per-stage displacement histograms: bin_center, frequency, Poisson_error
    all_deltas_per_stage = [
        np.array(all_delta_distances_0),
        np.array(all_delta_distances_1),
        np.array(all_delta_distances_2)
    ]
    for s, all_deltas in enumerate(all_deltas_per_stage):
        name = stage_names[s]
        bin_centers, frequencies, hist_errors = delta_distances_to_histogram(
            all_deltas, max_dist, n_bins=n_hist_bins)
        if bin_centers is not None:
            hist_out = np.column_stack([bin_centers, frequencies, hist_errors])
            np.savetxt(os.path.join(folder_name, f'Delta_distances_histogram_{name}.csv'),
                       hist_out, delimiter=',',
                       header='distance,frequency,poisson_error', comments='')

    # Reference times and lag grids — written unconditionally so they are always
    # available for plotting even if MSD computation is skipped for a stage
    np.savetxt(os.path.join(folder_name, 't_refs.csv'),
               np.array([t_refs[0] for t_refs in t_refs_per_stage], dtype=np.int64),
               delimiter=',')
    for s, taus in enumerate(taus_per_stage):
        name = stage_names[s]
        np.savetxt(os.path.join(folder_name, f'taus_{name}.csv'),
                   taus.astype(np.float64), delimiter=',')

    # MSD results: two columns — mean squared displacement and standard error
    for s, (taus, msd, sem) in enumerate(msd_results):
        if taus is None:
            continue
        name    = stage_names[s]
        msd_out = np.column_stack([msd, sem])
        np.savetxt(os.path.join(folder_name, f'msd_{name}.csv'),
                   msd_out, delimiter=',',
                   header='msd,sem', comments='')

    elapsed = time.perf_counter() - start_time
    print(f"All simulations completed. Runtime: {elapsed:.1f} s")

    # Days in stages 1/2 where the tortoise never reached its target burrow.
    # A high count may indicate the radius is too large or Max_time_mov too short.
    # (homing_days and total_homing_days are computed for reference but are
    # not currently printed.)
    homing_days = (day_brumation_begin - day_exploration_end + day_brumation_end - day_brumation_begin)
    total_homing_days = homing_days * (T // int(24*60*60 / seconds_per_time_step) // 365 or 1) * simulation_runs
    print(f"Days burrow not reached (stages 1+2): {total_missed_days} "
          f"across {simulation_runs} tortoises")


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patagonian tortoise yearly cycle simulation — CPU version (nightly reset model)")
    parser.add_argument('--T',                     type=int,   default=18550395,
                        help="Total timesteps (default: one full year at 1.7 s/step)")
    parser.add_argument('--seconds_per_time_step', type=float, default=1.7,
                        help="Duration of each timestep in seconds")
    parser.add_argument('--mu_vel',                type=float, default=0.82,
                        help="Lognormal mu for step-length distribution (step in cm)")
    parser.add_argument('--sigma_vel',             type=float, default=0.83,
                        help="Lognormal sigma for step-length distribution")
    parser.add_argument('--angular_sigma',         type=float, default=0.5,
                        help="Std of angular noise in radians")
    parser.add_argument('--alpha_stop',            type=float, default=-2.0,
                        help="Power-law exponent for stop durations, P(t) ~ t^alpha")
    parser.add_argument('--alpha_walk',            type=float, default=-2.0,
                        help="Power-law exponent for walk durations, P(t) ~ t^alpha")
    parser.add_argument('--simulation_runs',       type=int,   default=1000,
                        help="Number of independent tortoise trajectories")
    parser.add_argument('--Delta_T',               type=int,   default=500,
                        help="Interval between displacement samples (timesteps)")
    parser.add_argument('--n_hist_bins',           type=int,   default=100,
                        help="Number of bins for delta-distance histograms")
    parser.add_argument('--traj_every',            type=int,   default=2118,
                        help="Ticks between trajectory samples for the example "
                             "tortoise (default 2118 ≈ 1 hour at 1.7 s/step)")
    parser.add_argument('--stage2_radius_fraction', type=float, default=0.5,
                        help="Stage 2 search radius as a fraction of the calibrated "
                             "stage 1 radius (default 0.5 = half the daily range)")
    parser.add_argument('--n_refs',                type=int,   default=48,
                        help="Number of reference times per stage for MSD averaging "
                             "(default 48 × 30 min = 24 h coverage on day 2)")
    parser.add_argument('--ref_spacing_minutes',   type=float, default=30.0,
                        help="Spacing between reference times in minutes (default 30)")
    parser.add_argument('--folder_name',           type=str,   default='Simulations',
                        help="Root output directory")
    parser.add_argument('--sub_folder_name',       type=str,   default='LogNormalstep_CPU',
                        help="Subdirectory within folder_name for this run's output")
    args = parser.parse_args()

    if args.seconds_per_time_step >= 24*60*60:
        raise ValueError("seconds_per_time_step must be less than one day.")

    main(
        T=args.T,
        seconds_per_time_step=args.seconds_per_time_step,
        mean_r=args.mu_vel,
        sigma_r=args.sigma_vel,
        angular_sigma=args.angular_sigma,
        alpha_stop=args.alpha_stop,
        alpha_walk=args.alpha_walk,
        simulation_runs=args.simulation_runs,
        Delta_T=args.Delta_T,
        experiment_name=args.folder_name,
        subfolder_name=args.sub_folder_name,
        n_hist_bins=args.n_hist_bins,
        traj_every=args.traj_every,
        stage2_radius_fraction=args.stage2_radius_fraction,
        n_refs=args.n_refs,
        ref_spacing_minutes=args.ref_spacing_minutes
    )