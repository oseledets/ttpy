from numpy.distutils import customized_fcompiler


def get_extra_fflags():
    fflags = []
    fcompiler = customized_fcompiler()
    if fcompiler.compiler_type in ('g95', 'gnu', 'gnu95'):
        if fcompiler.get_version() >= '10':
            fflags.append('-fallow-argument-mismatch')
    return fflags
