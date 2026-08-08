# Building the old ttpy (for comparison)

Needed only for measurements: ttpy 2 never requires Fortran at any point. The
recipe is written down because "just build the old version" does not work on a
current machine, and that is worth recording.

## What gets in the way

1. `numpy.distutils` was removed in numpy >= 1.26 and does not work on
   Python >= 3.12 — Python 3.11 and numpy 1.24 are required.
2. `setuptools >= 60` shadows `distutils`, and the build dies with
   `ModuleNotFoundError: distutils.msvccompiler` — `setuptools < 60` is
   required.
3. `numpy.distutils` fails to find gfortran on its own and falls into the
   ARM/flang detector, dying with `TypeError: 'NoneType' object is not
   subscriptable` — the compiler has to be named explicitly:
   `config_fc --fcompiler=gnu95`.
4. The code f2py generates assigns an `int (*)(...)` to a variable of type
   `long int (*)(...)`. GCC >= 14 treats that as an error, not a warning.
5. Two submodules live on bitbucket and one on github; the clone has to be
   recursive.
6. `gfortran` may be absent from the system, and sudo may not be available.

## Recipe (verified on Ubuntu, Xeon 6767P, without root)

```bash
# 1. a compiler without sudo
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~ bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/bin/micromamba create -y -n ttlegacy -c conda-forge \
    python=3.11 "numpy=1.24" scipy six cython gfortran_linux-64 gcc_linux-64 \
    make openblas liblapack
E=~/micromamba/envs/ttlegacy
ln -sf $E/bin/x86_64-conda-linux-gnu-gfortran $E/bin/gfortran
ln -sf $E/bin/x86_64-conda-linux-gnu-gcc      $E/bin/gcc
$E/bin/python -m pip install "setuptools<60"

# 2. the sources, submodules included
git clone --recursive https://github.com/oseledets/ttpy ttpy-src
cd ttpy-src

# 3. the build
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib SETUPTOOLS_USE_DISTUTILS=stdlib
export CFLAGS="-Wno-incompatible-pointer-types -Wno-implicit-function-declaration -Wno-int-conversion -O2 -fPIC"
$E/bin/python setup.py config_fc --fcompiler=gnu95 build_ext --inplace
```

This produces `tt_f90`, `core_f90`, `amen_f90`, `tt_eigb`, `dyn_tt`, `maxvol`
and the two cython modules `rect_maxvol`. To run it, set
`PYTHONPATH=<path to ttpy-src>`.

Note item 4: suppressing `-Wincompatible-pointer-types` is not "we turned a
warning off" but a deliberate decision to ignore a mismatch between `int*` and
`long int*` in the f2py callback interface. The resulting build was used only
for speed measurements and to check three constructors, and every result was
verified against dense truth independently.

## What the comparison showed

| check | old ttpy | ttpy 2 |
|---|---|---|
| `qshift(3)` against `eye(8, k=-1)` | matched `k=+1` (transposed) | matched |
| `Toeplitz(x, kind='L')` against the lower triangular one | matched the transpose | matched |
| `IpaS(2, 0.5)` against `I + 0.5 S_{-1}` | matched neither it nor its transpose | matched |
| `qlaplace_dd` against the dense Laplacian | matched | matched |

For `d=2` the old package returns from `IpaS`

```
[[1.   0.   0.   0.5]      expected   [[1.   0.   0.   0. ]
 [0.5  1.   0.   0. ]                  [0.5  1.   0.   0. ]
 [0.   0.5  0.   0. ]                  [0.   0.5  1.   0. ]
 [0.   0.   0.   0. ]]                 [0.   0.   0.5  1. ]]
```

— a stray entry in the top right corner, a zero on the diagonal and a zero row.
The cause: the first core there is written in the `(r, i, j, r)` layout while the
middle ones use `(i, j, r, r)`. When every dimension equals two the mismatched
axes do not produce a shape error, which is how it survived this long.
