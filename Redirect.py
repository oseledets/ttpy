import threading, time, sys, os, fcntl

__all__ = ['start_redirect', 'stop_redirect']

# A thread that will redirect the c++ stdout to the ipython notebook
class T(threading.Thread):

  def __init__(self):
    threading.Thread.__init__(self)
    self.go_on = True
    
    # copy the c++ stdout handler for later
    self.oldhandle = os.dup(1)

  def stop(self):
    self.go_on = False

  def run(self):
# create a pipe and glue the c++ stdout to its write end
# make the read end non-blocking
    piper, pipew = os.pipe()
    fcntl.fcntl(piper,fcntl.F_SETFL,os.O_NONBLOCK)
    os.dup2(pipew,1)
    os.close(pipew)
    import ipdb; ipdb.set_trace()
    while self.go_on:
    # give the system 1 second to fill up the pipe
        time.sleep(1e-3)
    try:
        # read out the pipe and write it to the screen
        # if the pipe was empty it return an error (hence the try-except)
        sys.stdout.write(os.read(piper, 10000))
        sys.stdout.flush()
    except:
        pass

# when we want to stop the thread put back the c++ stdout where it was
# clear the pipe
    os.dup2(self.oldhandle, 1)
    os.close(piper)
    return

# flag to know if the thread was started
started = False

# start the redirection
def start_redirect():
  global started
  global a
  if started:
    print "Already redirected c++ output"
  else:
    print "Starting"
    started = True
    # start a new redirection thread
    a = T()
    a.start()
    
# stop the redirection
def stop_redirect():
  global started
  global a
  if started:
    started = False
    a.stop()
