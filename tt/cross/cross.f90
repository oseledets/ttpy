module cross
  use tt_lib
  use ttop_lib
  use time_lib
  use dmrgfun_lib
  use python_conv_lib
  double precision, allocatable :: core(:)
contains
  subroutine tt_cross(d,n,r,ps,fun,eps)
    integer, intent(in) :: d
    integer, intent(in)  :: n(d)
    integer, intent(out) :: r(d+1)
    integer, intent(out) :: ps(d+1)
    double precision, intent(in) :: eps
    type (dtt) :: tt,tt1
    external :: fun
    double precision :: fun
    integer :: ind(d),i,nit
    double precision val,t1,t2
    !fun is a function of a multiindex of length d
    !print *,'d=',d
    !print *,'n=',n
    !do i=1,d
    !   ind(i) = i
    !end do
    !val = fun(d,ind) !Hehe
    do i=1,d
       tt%n(i)=n(i)
    end do
    tt%l=1
    tt%m=d

    !Measure time to call fun
    !nit=10000
    !t1=timef()
    !do i=1,nit
    !   val=fun(d,ind)
    !end do
    !t2=timef()
    !print *,'Time to call a python fun from fortran:',(t2-t1)/nit
    !t1=timef()
    !do i=1,nit
    !   val=fun1(d,ind)
    !end do
    !t2=timef()
    !print *,'Time to call fortran from fortran',(t2-t1)/(nit)
    !Initialize tt
    call dtt_ones(tt)

    !Now we have to send fun to dmrg
    !call dtt_dmrgf(tt, eps, maxiter, coresize, kick, dfun)

    !t1=timef()
    call dtt_dmrgf(tt,eps,fun)
    call sdv_to_arrays(n,r,d,ps,core,tt)
    call dealloc(tt)
    !t2=timef()
    !print *,'Time for Python callback',t2-t1
    !tt1%l=1
    !tt1%m=d
    !do i=1,d
    !   tt1%n(i)=n(i)
    !end do

    !call dtt_ones(tt1)
    !t1=timef()
    !call dtt_dmrgf(tt1,eps,fun1)
    !t2=timef()
    !print *,'Time for Fortran fun:',t2-t1
  end subroutine tt_cross

  subroutine cross_dealloc()
     deallocate(core)
  end subroutine cross_dealloc
  !   function fun1(d,x) result(res)
  !  integer, intent(in) :: d
  !  integer, intent(in) :: x(d)
  !  double precision :: res
  !  res=sum(x(1:d))
  !end function fun1


end module cross
