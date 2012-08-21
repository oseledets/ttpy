.SUFFIXES: .c .f .f90 .o

CPU     =  mac-gfort
#FOPT    = -m64 -fopenmp #-fdefault-integer-8 #-xSSE4.2
DEPS    =  nan, timef, say, rnd, ptype, sort, trans, ort, mat, check, lr, maxvol, svd,  rawimg, \
	     bt, quad,  tt, ttaux, ttop, ttio,  tts, dmrg_fun, python_conv, tensor_util
          # d3, mimic, d3als, d3kryl, d3op, d3test, d3elp, 
include Makefile.in

OBJS    = $(DEPS:,=.o).o
MODS    = *.mod
OBJF	= $(OBJS)
OBJC	= 


tt_f90.so: mytt.a tt_f90.f90 cross.f90 tt_matrix.f90
	f2py tt_f90.f90 --build-dir f2py_temp -c   -m  tt_f90 mytt.a -lgomp --f90flags="${FOPT}"  
	f2py --build-dir f2py_temp  -c cross.pyf cross.f90 mytt.a -lgomp --f90flags="${FOPT}"
	f2py tt_matrix.f90 --build-dir f2py_temp  -c  -m tt_matrix_f90 mytt.a -lgomp --f90flags="${FOPT}"
mytt.a : $(OBJS)
	ar rc mytt.a $(OBJS)
.f.o:
		$(FORT) -c $<
.f90.o:
		$(FORT) -c $<
.c.o:
		$(CC) -c $<
clean:
		rm -f $(OBJF) $(OBJC) $(MODS) tt_f90.so tt_matrix_f90.so cross.so mytt.a
