f2py --build-dir f2py_temp  -c lobpcg.pyf lobpcg.f90 dispmodule.a libprimme.a --debug --f90flags="-fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=bounds  -fcheck=do  -fcheck=mem  -fcheck=recursion  -std=f2008  -pedantic  -fbacktrace" 

