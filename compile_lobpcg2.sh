f2py --build-dir f2py_temp  -c lobpcg2.pyf lobpcg2.f90 dispmodule.a libprimme.a --debug --f90flags="-ffree-line-length-none -fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=bounds  -fcheck=do  -fcheck=mem  -fcheck=recursion  -std=f2008  -pedantic  -fbacktrace" 

