"""Import this FIRST, before numpy/scipy/sklearn/torch, in every entrypoint.

Without this, sklearn's internally-OpenMP-parallel routines (e.g. the
nearest-neighbor search SMOTE uses) each spawn one thread per core, and
every one of those threads then calls into Apple's Accelerate BLAS, which
ALSO spawns one thread per core — an N-times-N oversubscription. On this
10-core machine that produced a load average of ~90 and made a build_clients
call that normally takes ~90s effectively hang. Capping Accelerate/BLAS to
a single thread per OpenMP worker fixes it; the env vars must be set before
the relevant libraries are first imported, since most BLAS backends read
them once at load time.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
