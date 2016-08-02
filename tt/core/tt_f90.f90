module tt_f90
  use tt_lib
  use ttop_lib
  use time_lib
  use python_conv_lib
  real(8), allocatable :: core(:)
  complex(8), allocatable :: zcore(:)
contains
  

  subroutine dfull_to_tt(a,n,d,eps,rmax,r,ps)
    implicit none
    integer, intent(in), optional :: rmax
    integer, intent(in) :: d
    real(8), intent(in) :: eps
    integer, intent(in) :: n(:)
    integer, intent(out) :: r(d+1)
    integer, intent(out) :: ps(d+1)
    real(8), intent(in) :: a(:)
    type(dtt) :: tt
    integer :: n1(d)
    if (present(rmax) ) then
       call svd(n, a, tt, eps, rmax)
    else
       call svd(n, a, tt, eps)
    end if
    call sdv_to_arrays(n1,r,d,ps,core,tt)
    call dealloc(tt)
  end subroutine dfull_to_tt
  
  subroutine zfull_to_tt(a,n,d,eps,rmax,r,ps)
    implicit none
    integer, intent(in), optional :: rmax
    integer, intent(in) :: d
    real(8), intent(in) :: eps
    integer, intent(in) :: n(:)
    integer, intent(out) :: r(d+1)
    integer, intent(out) :: ps(d+1)
    complex(8), intent(in) :: a(:)
    type(ztt) :: tt
    integer :: n1(d)
    call svd(n,a,tt,eps,rmax)
    call sdv_to_arrays(n1,r,d,ps,zcore,tt)
    call dealloc(tt)
  end subroutine zfull_to_tt

  subroutine tt_dealloc()
  if ( allocated(core) ) then
    deallocate(core)
  end if
  if ( allocated(zcore) ) then
      deallocate(zcore)
  end if
  end subroutine tt_dealloc
  
  subroutine dtt_write_wrapper(n,r,d,ps,cr,crsize,fnam)
    use ttio_lib
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    character(len=*),intent(in) :: fnam
    integer, intent(in) :: crsize
    double precision, intent(in) :: cr(crsize)
    type(dtt) :: tt
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call write(tt,fnam)
    call dealloc(tt)
  end subroutine dtt_write_wrapper
  
  subroutine ztt_write_wrapper(n,r,d,ps,cr,crsize,fnam)
    use ttio_lib
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    character(len=*),intent(in) :: fnam
    integer, intent(in) :: crsize
    complex(8), intent(in) :: cr(crsize)
    type(ztt) :: tt
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call write(tt,fnam)
    call dealloc(tt)
  end subroutine ztt_write_wrapper
  
  subroutine dtt_read_wrapper(n,r,d,d0,ps,fnam)
    use ttio_lib
    implicit none
    integer, intent(in) :: d0
    integer, intent(out) :: d
    integer, intent(inout) :: n(d0)
    integer, intent(inout) :: r(d0+1)
    integer, intent(inout) :: ps(d0+1)
    character(len=*),intent(in) :: fnam
    type(dtt) :: tt
    call read(tt,fnam)
    call sdv_to_arrays(n,r,d,ps,core,tt)
    call dealloc(tt)
  end subroutine dtt_read_wrapper
  
  subroutine ztt_read_wrapper(n,r,d,d0,ps,fnam)
    use ttio_lib
    implicit none
    integer, intent(in) :: d0
    integer, intent(out) :: d
    integer, intent(inout) :: n(d0)
    integer, intent(inout) :: r(d0+1)
    integer, intent(inout) :: ps(d0+1)
    character(len=*),intent(in) :: fnam
    type(ztt) :: tt
    call read(tt,fnam)
    call sdv_to_arrays(n,r,d,ps,zcore,tt)
    call dealloc(tt)
  end subroutine ztt_read_wrapper
  
  
  !a should be preallocated, and filled by zeros
  subroutine dtt_to_full(n,r,d,ps,cr,crsize,a,asize)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    integer, intent(in) :: asize
    integer, intent(in) :: crsize
    double precision, intent(in) :: cr(crsize)
    double precision, intent(out) :: a(asize)
    type(dtt) :: tt
    a(1:asize)=0d0
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call full(tt,a)
    call dealloc(tt)
  end subroutine dtt_to_full
  
  subroutine ztt_to_full(n,r,d,ps,cr,crsize,a,asize)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    integer, intent(in) :: asize
    integer, intent(in) :: crsize
    complex(8), intent(in) :: cr(crsize)
    complex(8), intent(out) :: a(asize)
    type(ztt) :: tt
    a(1:asize)=(0d0,0d0)
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call full(tt,a)
    call dealloc(tt)
  end subroutine ztt_to_full

  subroutine dtt_add(n,d,r1,r2,ps1,ps2,core1,core2,rres,psres)
    implicit none
    integer, intent(in)  :: d
    integer, intent(in)  :: n(d)
    integer, intent(in)  :: r1(d+1)
    integer, intent(in)  :: r2(d+1)
    integer, intent(in)  :: ps1(d+1)
    integer, intent(in)  :: ps2(d+1)
    integer, intent(out) :: rres(d+1)
    integer, intent(out) :: psres(d+1)
    real(8),    intent(in) :: core1(:)
    real(8),    intent(in) :: core2(:)
    type(dtt) :: tt1, tt2
    integer :: n1(d)
    call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
    call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
    call axpy(1.d0,tt1,1.d0,tt2)
    !tt1 is the sum
    call sdv_to_arrays(n1,rres,d,psres,core,tt2)
    call dealloc(tt1)
    call dealloc(tt2)


  end subroutine dtt_add
  
  subroutine ztt_add(n,d,r1,r2,ps1,ps2,core1,core2,rres,psres)
    implicit none
    integer, intent(in)  :: d
    integer, intent(in)  :: n(d)
    integer, intent(in)  :: r1(d+1)
    integer, intent(in)  :: r2(d+1)
    integer, intent(in)  :: ps1(d+1)
    integer, intent(in)  :: ps2(d+1)
    integer, intent(out) :: rres(d+1)
    integer, intent(out) :: psres(d+1)
    complex(8),    intent(in) :: core1(:)
    complex(8),    intent(in) :: core2(:)
    complex(8) :: ONE
    type(ztt) :: tt1, tt2
    integer :: n1(d)
    ONE = (1d0, 0d0)
    call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
    call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
    call axpy(ONE,tt1,ONE,tt2)
    !tt1 is the sum
    call sdv_to_arrays(n1,rres,d,psres,zcore,tt2)
    call dealloc(tt1)
    call dealloc(tt2)
  end subroutine ztt_add


  subroutine dtt_compr2(n,d,r,ps,cr,eps,rmax)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(inout) :: r(d+1)
    integer, intent(inout) :: ps(d+1)
    real(8), intent(in) :: cr(:)
    real(8), intent(in) :: eps
    integer, intent(in) :: rmax
    type(dtt) :: tt
    integer :: n1(d)
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call svd(tt,eps,rmax)
    call sdv_to_arrays(n1,r,d,ps,core,tt)
    call dealloc(tt)
  end subroutine dtt_compr2
  
  subroutine ztt_compr2(n,d,r,ps,cr,eps,rmax)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(inout) :: r(d+1)
    integer, intent(inout) :: ps(d+1)
    complex(8), intent(in) :: cr(:)
    real(8), intent(in) :: eps
    integer, intent(in) :: rmax
    type(ztt) :: tt
    integer :: n1(d)
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    call svd(tt,eps,rmax)
    call sdv_to_arrays(n1,r,d,ps,zcore,tt)
    call dealloc(tt)
  end subroutine ztt_compr2

  subroutine dtt_nrm(n,d,r,ps,cr,nrm)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    real(8), intent(in) :: cr(:)
    real(8), intent(out) :: nrm
    type(dtt) :: tt
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    nrm=norm(tt)
    call dealloc(tt)
  end subroutine dtt_nrm
  
    subroutine ztt_nrm(n,d,r,ps,cr,nrm)
    implicit none
    integer, intent(in) :: d
    integer, intent(in) :: n(d)
    integer, intent(in) :: r(d+1)
    integer, intent(in) :: ps(d+1)
    complex(8), intent(in) :: cr(:)
    real(8), intent(out) :: nrm
    type(ztt) :: tt
    call arrays_to_sdv(n,r,d,ps,cr,tt)
    nrm=norm(tt)
    call dealloc(tt)

    end subroutine ztt_nrm
  
 subroutine dtt_dotprod(n,d,r1,r2,ps1,ps2,core1,core2,dt,dtsize)
    implicit none
    integer, intent(in)  :: d
    integer, intent(in)  :: dtsize
    integer, intent(in)  :: n(d)
    integer, intent(in)  :: r1(d+1)
    integer, intent(in)  :: r2(d+1)
    integer, intent(in)  :: ps1(d+1)
    integer, intent(in)  :: ps2(d+1)
    real(8), intent(in) :: core1(:)
    real(8), intent(in) :: core2(:)
    real(8), intent(out) :: dt(dtsize)
    type(dtt) :: tt1, tt2
    call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
    call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
    dt = dot(tt1,tt2)
    call dealloc(tt1)
    call dealloc(tt2)
  end subroutine dtt_dotprod
 
  subroutine ztt_dotprod(n,d,r1,r2,ps1,ps2,core1,core2,dt,dtsize)
    implicit none
    integer, intent(in)  :: d
    integer, intent(in)  :: dtsize
    integer, intent(in)  :: n(d)
    integer, intent(in)  :: r1(d+1)
    integer, intent(in)  :: r2(d+1)
    integer, intent(in)  :: ps1(d+1)
    integer, intent(in)  :: ps2(d+1)
    complex(8), intent(in) :: core1(:)
    complex(8), intent(in) :: core2(:)
    complex(8), intent(out) :: dt(dtsize)
    type(ztt) :: tt1, tt2
    call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
    call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
    dt = dot(tt1,tt2)
    call dealloc(tt1)
    call dealloc(tt2)
  end subroutine ztt_dotprod
    !Later on we will avoid allocation in + and hdm, where the result have a very specific
    !size, i.e., ranks core size can be precomputed
 
 subroutine dtt_hdm(n,d,r1,r2,ps1,ps2,core1,core2,rres,psres)
      integer, intent(in)  :: d
      integer, intent(in)  :: n(d)
      integer, intent(in)  :: r1(d+1)
      integer, intent(in)  :: r2(d+1)
      integer, intent(in)  :: ps1(d+1)
      integer, intent(in)  :: ps2(d+1)
      integer, intent(out) :: rres(d+1)
      integer, intent(out) :: psres(d+1)
      real(8),    intent(in) :: core1(:)
      real(8),    intent(in) :: core2(:)
      type(dtt) :: tt1, tt2, tt
      integer :: n1(d)
      call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
      call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
      call dtt_ha1(tt1,tt2,tt)
      call dealloc(tt1)
      call dealloc(tt2)
      call sdv_to_arrays(n1,rres,d,psres,core,tt)
      call dealloc(tt)
 end subroutine dtt_hdm

 subroutine ztt_hdm(n,d,r1,r2,ps1,ps2,core1,core2,rres,psres)
      integer, intent(in)  :: d
      integer, intent(in)  :: n(d)
      integer, intent(in)  :: r1(d+1)
      integer, intent(in)  :: r2(d+1)
      integer, intent(in)  :: ps1(d+1)
      integer, intent(in)  :: ps2(d+1)
      integer, intent(out) :: rres(d+1)
      integer, intent(out) :: psres(d+1)
      complex(8),    intent(in) :: core1(:)
      complex(8),    intent(in) :: core2(:)
      type(ztt) :: tt1, tt2, tt
      integer :: n1(d)
      call arrays_to_sdv(n,r1,d,ps1,core1,tt1)
      call arrays_to_sdv(n,r2,d,ps2,core2,tt2)
      call ztt_ha1(tt1,tt2,tt)
      call dealloc(tt1)
      call dealloc(tt2)
      call sdv_to_arrays(n1,rres,d,psres,zcore,tt)
      call dealloc(tt)
 end subroutine ztt_hdm

! Check, if we can call an external python function from Fortran




end module tt_f90
