rm tt_adapt_als.mod
gcc -O3 -c uchol_full.c
f2py -c -m amr_f90 tt-fort/tt_adapt_als.f90 --f90flags="-ffree-line-length-none" uchol_full.o

