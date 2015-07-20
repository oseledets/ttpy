import os
import sys
import nose
import tt
verbose = True
nose_argv = sys.argv
nose_argv += ['--detailed-errors', '--exe']
if verbose:
    nose_argv.append('-v')
initial_dir = os.getcwd()
tt_file = os.path.abspath(tt.__file__)
tt_dir = os.path.dirname(tt_file)
os.chdir(tt_dir)
try:
    nose.run(argv=nose_argv)
finally:
    os.chdir(initial_dir)
