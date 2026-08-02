# Как собрать старый ttpy (для сравнения)

Нужно только для замеров: ttpy 2 ни в какой момент не требует Fortran.
Рецепт приведён потому, что «просто соберите старую версию» на современной
машине не работает, и это стоит зафиксировать.

## Что мешает

1. `numpy.distutils` удалён в numpy >= 1.26 и не работает на Python >= 3.12 —
   нужен Python 3.11 и numpy 1.24.
2. `setuptools >= 60` подменяет `distutils`, и сборка падает на
   `ModuleNotFoundError: distutils.msvccompiler` — нужен `setuptools < 60`.
3. `numpy.distutils` сам не находит gfortran и уходит в детектор ARM/flang,
   падая с `TypeError: 'NoneType' object is not subscriptable` — компилятор надо
   назвать явно: `config_fc --fcompiler=gnu95`.
4. Код, который генерирует f2py, присваивает `int (*)(...)` переменной типа
   `long int (*)(...)`. GCC >= 14 считает это ошибкой, а не предупреждением.
5. Два сабмодуля живут на bitbucket, один на github; клонировать надо
   рекурсивно.
6. `gfortran` в системе может отсутствовать, а sudo — не быть.

## Рецепт (проверен на Ubuntu, Xeon 6767P, без прав root)

```bash
# 1. компилятор без sudo
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~ bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/bin/micromamba create -y -n ttlegacy -c conda-forge \
    python=3.11 "numpy=1.24" scipy six cython gfortran_linux-64 gcc_linux-64 \
    make openblas liblapack
E=~/micromamba/envs/ttlegacy
ln -sf $E/bin/x86_64-conda-linux-gnu-gfortran $E/bin/gfortran
ln -sf $E/bin/x86_64-conda-linux-gnu-gcc      $E/bin/gcc
$E/bin/python -m pip install "setuptools<60"

# 2. исходники вместе с сабмодулями
git clone --recursive https://github.com/oseledets/ttpy ttpy-src
cd ttpy-src

# 3. сборка
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib SETUPTOOLS_USE_DISTUTILS=stdlib
export CFLAGS="-Wno-incompatible-pointer-types -Wno-implicit-function-declaration -Wno-int-conversion -O2 -fPIC"
$E/bin/python setup.py config_fc --fcompiler=gnu95 build_ext --inplace
```

Получаются `tt_f90`, `core_f90`, `amen_f90`, `tt_eigb`, `dyn_tt`, `maxvol`
и два cython-модуля `rect_maxvol`. Запуск: `PYTHONPATH=<путь к ttpy-src>`.

Обратите внимание на пункт 4: подавление `-Wincompatible-pointer-types` — это
не «предупреждение выключили», а сознательное игнорирование несоответствия
`int*` и `long int*` в интерфейсе обратного вызова f2py. Результаты сборки мы
использовали только для замеров скорости и для проверки трёх конструкторов,
причём каждый результат сверялся с плотной истиной независимо.

## Что показала сверка

| проверка | старый ttpy | ttpy 2 |
|---|---|---|
| `qshift(3)` против `eye(8, k=-1)` | совпал с `k=+1` (транспонирован) | совпал |
| `Toeplitz(x, kind='L')` против нижнетреугольной | совпал с транспонированной | совпал |
| `IpaS(2, 0.5)` против `I + 0.5 S_{-1}` | не совпал ни с ней, ни с транспонированной | совпал |
| `qlaplace_dd` против плотного лапласиана | совпал | совпал |

`IpaS` в старом пакете возвращает при `d=2`

```
[[1.   0.   0.   0.5]      ожидается  [[1.   0.   0.   0. ]
 [0.5  1.   0.   0. ]                  [0.5  1.   0.   0. ]
 [0.   0.5  0.   0. ]                  [0.   0.5  1.   0. ]
 [0.   0.   0.   0. ]]                 [0.   0.   0.5  1. ]]
```

то есть посторонний элемент в правом верхнем углу, ноль на диагонали и нулевая
строка. Причина: первое ядро там записано в раскладке `(r, i, j, r)`, а средние —
в `(i, j, r, r)`. При всех размерностях, равных двум, несоответствие осей не
приводит к ошибке формы, поэтому оно и дожило до наших дней.
