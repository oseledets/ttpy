   valgrind --tool=memcheck --suppressions=valgrind-python.supp \
       python -E -tt test_eigb.py -u bsddb,network

