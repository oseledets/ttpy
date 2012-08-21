f2py --build-dir f2py_temp -c matrix_util.f90 ttals.f90 tt_eigb.pyf  dispmodule.f90 tt_eigb.f90 libprimme.a --fcompiler=gnu95  --f90flags="-O3 -ffree-line-length-none -fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fcheck=bounds  -fcheck=do  -fcheck=mem  -fcheck=recursion  -std=f2008  -pedantic  -fbacktrace" 

