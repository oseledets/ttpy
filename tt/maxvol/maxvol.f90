      subroutine dmaxvol(a, n, r, ind, nswp, tol)
      implicit none
      integer, intent(in) :: r
      integer, intent(in) :: n
      real(8), intent(in) :: a(n,r)
      integer, intent(out) :: ind(r)
      integer tmp_ind(r),p(n),ipiv(r)
      real(8) ::  ba(r,n),c(n,r), u(n), v(r)
      real(8), intent(in) :: tol
      logical not_converged
      integer, intent(in) ::  nswp
      integer info,i,j,swp, big_ind,i0,j0,tmp
      integer idamax
      external idamax
      !tol=5e-2
      !nswp=20
      !Generate initial approximation
      ba = transpose(a)
      call dcopy(n*r,a,1,c,1)
      do i = 1,n
      p(i) = i
      end do
      call dgetrf(n,r,c,n,ipiv,info) 
      do i = 1,r
      j = ipiv(i)
      if ( j .ne. i ) then
          tmp = p(i)
          p(i) = p(j)
          p(j) = tmp
      end if 
      end do
      ind(1:r) = p(1:r)
      if (info .ne. 0) then
          print *, 'Maxvol failed at dgetrf'
      end if
      do i = 1,r
      tmp_ind(i) = i
      end do
      call dgetrs('t',r,n,c,n,tmp_ind,ba,r,info)

      if (info .ne. 0) then
          print *,'Maxvol failed at dgetrs'
      end if
      not_converged = .true.
      swp = 1
      !Now start the main iteration
      do while (not_converged .and. swp <= nswp) 
      big_ind = idamax(r*n,ba,1)
      j0 = mod(big_ind - 1,r) + 1 ! This value is from 1 to r 
      i0 = (big_ind - j0)/ r + 1 ! This value seems OK: If it is smaller, it goes 
      if ( i0 > n ) then 
          print *,'aarrgh'
          print *,'big_ind=',big_ind, 'i0=',i0,'j0=',j0,'n=',n,'r=',r
      end if 

      if ( dabs(ba(j0,i0)) <= 1 + tol ) then
          not_converged = .false.
      else 

          u(1:n) = ba(j0,:)
          v(1:r) = ba(:,ind(j0)) - ba(:,i0)
          u(1:n) = u(1:n) / ba(j0,i0)
          call dger(r,n,1.d0,v,1,u,1,ba,r)
          swp = swp + 1
          ind(j0) = i0

      end if

      end do

      end subroutine dmaxvol

      subroutine zmaxvol(a, n, r, ind, nswp, tol)
      implicit none
      integer, intent(in) :: r
      integer, intent(in) :: n
      complex(8), intent(in) :: a(n,r)
      integer, intent(out) :: ind(r)
      integer tmp_ind(r),p(n),ipiv(r)
      complex(8) ::  ba(r,n),c(n,r), u(n), v(r)
      real(8), intent(in) :: tol
      logical not_converged
      integer, intent(in) ::  nswp
      integer info,i,j,swp, big_ind,i0,j0,tmp
      complex(8) :: ONE
      integer izamax
      external izamax
      !tol=5e-2
      !nswp=20
      !Generate initial approximation
      ONE = (1d0, 0d0)
      ba = transpose(a)
      call zcopy(n*r,a,1,c,1)
      do i = 1,n
      p(i) = i
      end do
      call zgetrf(n,r,c,n,ipiv,info) 
      do i = 1,r
      j = ipiv(i)
      if ( j .ne. i ) then
          tmp = p(i)
          p(i) = p(j)
          p(j) = tmp
      end if 
      end do
      ind(1:r) = p(1:r)
      if (info .ne. 0) then
          print *, 'Maxvol failed at zgetrf'
      end if
      do i = 1,r
      tmp_ind(i) = i
      end do
      call zgetrs('T',r,n,c,n,tmp_ind,ba,r,info)

      if (info .ne. 0) then
          print *,'Maxvol failed at zgetrs'
      end if
      not_converged = .true.
      swp = 1
      !Now start the main iteration
      do while (not_converged .and. swp <= nswp) 
      big_ind = izamax(r*n,ba,1)
      j0 = mod(big_ind - 1,r) + 1 ! This value is from 1 to r 
      i0 = (big_ind - j0)/ r + 1 ! This value seems OK: If it is smaller, it goes 
      if ( i0 > n ) then 
          print *,'aarrgh'
          print *,'big_ind=',big_ind, 'i0=',i0,'j0=',j0,'n=',n,'r=',r
      end if 

      if ( zabs(ba(j0,i0)) <= 1 + tol ) then
          not_converged = .false.
      else 

          u(1:n) = ba(j0,:)
          v(1:r) = ba(:,ind(j0)) - ba(:,i0)
          u(1:n) = u(1:n) / ba(j0,i0)
          call zgeru(r,n,ONE,v,1,u,1,ba,r)
          swp = swp + 1
          ind(j0) = i0

      end if

      end do

      end subroutine zmaxvol
