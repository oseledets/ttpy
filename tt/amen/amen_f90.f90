module amen_f90
  use ttamen_lib
  use python_conv_lib
  real(8), allocatable :: core(:)
  complex(8), allocatable :: zcore(:)
contains

  subroutine deallocate_result()
  if ( allocated(core) ) then
    deallocate(core)
  end if
  if ( allocated(zcore) ) then
      deallocate(zcore)
  end if
  end subroutine deallocate_result

 subroutine dtt_amen_wrapper(d, n, m, rA, psA, crA, ry, psy, cry, rx, psx, crx, tol, kickrank, nswp, &
                             local_iters, local_restart, trunc_norm, max_full_size, verb, local_prec)
  implicit none
  integer, intent(in) :: d
  integer, intent(in) :: n(d)
  integer, intent(inout) :: m(d)
  integer, intent(in) :: rA(d+1)
  integer, intent(in) :: psA(d+1)
  real(8), intent(in) :: crA(:)
  integer, intent(in) :: ry(d+1)
  integer, intent(in) :: psy(d+1)
  real(8), intent(in) :: cry(:)
  integer, intent(inout) :: rx(d+1)
  integer, intent(inout) :: psx(d+1)
  real(8), intent(in) :: crx(:)
  
  double precision,intent(in) :: tol
  integer,intent(in) :: kickrank, local_iters, local_restart, nswp, trunc_norm, verb, max_full_size
  character,intent(in) :: local_prec
  
  type(dtt) :: A, y, x
  integer :: nm(d)
  nm = n * m
  call arrays_to_sdv(nm,rA,d,psA,crA,A)
  call arrays_to_sdv(n,ry,d,psy,cry,y)
  call arrays_to_sdv(m,rx,d,psx,crx,x)
  call dtt_amen_solve(A, y, tol, x, kickrank, nswp, local_prec, local_iters, local_restart, trunc_norm, max_full_size, verb)
  call dealloc(A)
  call dealloc(y)
  call sdv_to_arrays(m,rx,d,psx,core,x)
  call dealloc(x)
  
 end subroutine dtt_amen_wrapper

 subroutine ztt_amen_wrapper(d, n, m, rA, psA, crA, ry, psy, cry, rx, psx, crx, tol, kickrank, nswp, &
                             local_iters, local_restart, trunc_norm, max_full_size, verb, local_prec)
  implicit none
  integer, intent(in) :: d
  integer, intent(in) :: n(d)
  integer, intent(inout) :: m(d)
  integer, intent(in) :: rA(d+1)
  integer, intent(in) :: psA(d+1)
  complex(8), intent(in) :: crA(:)
  integer, intent(in) :: ry(d+1)
  integer, intent(in) :: psy(d+1)
  complex(8), intent(in) :: cry(:)
  integer, intent(inout) :: rx(d+1)
  integer, intent(inout) :: psx(d+1)
  complex(8), intent(in) :: crx(:)
  
  double precision,intent(in) :: tol
  integer,intent(in) :: kickrank, local_iters, local_restart, nswp, trunc_norm, verb, max_full_size
  character,intent(in) :: local_prec
  
  type(ztt) :: A, y, x
  integer :: nm(d)
  nm = n * m
  call arrays_to_sdv(nm,rA,d,psA,crA,A)
  call arrays_to_sdv(n,ry,d,psy,cry,y)
  call arrays_to_sdv(m,rx,d,psx,crx,x)
  call ztt_amen_solve(A, y, tol, x, kickrank, nswp, local_prec, local_iters, local_restart, trunc_norm, max_full_size, verb)
  call dealloc(A)
  call dealloc(y)
  call sdv_to_arrays(m,rx,d,psx,zcore,x)
  call dealloc(x)
  
 end subroutine ztt_amen_wrapper
end module amen_f90
