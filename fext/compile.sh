#f2py tt_f90.f90 -c -m only: full_to_tt tt_norm tt_dealloc tt_to_full tt_add tt_compr2 \
#add2 : tt_f90 -llapack -lblas mytt.so; cp mytt.so ..
#f2py tt_f90.f90 --build-dir f2py_temp -c -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 -framework vecLib  mytt.so; cp mytt.so ..
#f2py tt_f90.f90 --build-dir f2py_temp -c  -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/ -lBLAS -lLAPACK mytt.so; cp mytt.so ..
rm tt_f90.so
#f2py tt_f90.f90 --build-dir f2py_temp -c  -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 -L/Library/Frameworks/EPD64.framework/Versions/7.2/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm mytt.so; cp mytt.so .. 
#f2py tt_f90.f90 --build-dir f2py_temp -c -m tt_f90 -llapack -lblas mytt.so; cp mytt.so ..
#f2py tt_f90.f90 --build-dir f2py_temp -c   -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 libgoto2.a mytt.a  
f2py tt_f90.f90 --build-dir f2py_temp -c   -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 mytt.a  
#f2py tt_f90.f90 --build-dir f2py_temp -c --f90flags="-fPIC"  -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 mytt.a  
#f2py tt_f90.f90 --build-dir f2py_temp -c   -m skip: sdv_to_arrays arrays_to_sdv : tt_f90 -L/Applications/MATLAB_R2011a.app/bin/maci64/ -lmwlapack -lmwblas  mytt.so; cp mytt.so ..  
#f2py --build-dir f2py_temp -m tt_f90 -h tt_f90.pyf tt_f90.f90 skip: sdv_to_arrays arrays_to_sdv 
