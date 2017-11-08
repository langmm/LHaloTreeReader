from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist
from subprocess import Popen, PIPE
import copy
import numpy
import os
import sys
import warnings
import platform

_package_name = 'pylht'
PY_MAJOR_VERSION = sys.version_info[0]
PY2 = (PY_MAJOR_VERSION == 2)

# Check for ReadTheDocs/coverage flags
RTDFLAG = bool(os.environ.get('READTHEDOCS', None) == 'True')
COVFLAG = bool(os.environ.get(_package_name.upper() + '_COVERAGE', None) == 'True')
PRFFLAG = bool(os.environ.get(_package_name.upper() + '_PROFILE', None) == 'True')
OSXFLAG = (platform.system() == 'Darwin')

# Check for Cython
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError('Cython is a required dependency of ' + _package_name)

ext_options = dict(language="c",
                   include_dirs=[numpy.get_include()],
                   libraries=[],
                   extra_link_args=["-lm"],
                   extra_compile_args=["-std=c99", "-fno-strict-aliasing", "-g",
                                       "-D_POSIX_SOURCE=200809L",
                                       "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE"])
cyt_options = {}
if not OSXFLAG:
    ext_options['extra_link_args'].append("-lrt")


# Needed for line_profiler/coverage - disabled for production code
if not RTDFLAG and (COVFLAG or PRFFLAG):
    try:
        from Cython.Compiler.Options import directive_defaults
    except ImportError:
        # Update to cython
        from Cython.Compiler.Options import get_directive_defaults
        directive_defaults = get_directive_defaults()
    directive_defaults['profile'] = True
    directive_defaults['linetrace'] = True
    directive_defaults['binding'] = True
    cyt_options['compiler_directives'] = {'linetrace': True,
                                          'profile': True,
                                          'binding': True}
    ext_options['define_macros'] = [('CYTHON_TRACE', 1),
                                    ('CYTHON_TRACE_NOGIL', 1)]

def call_subprocess(args):
    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    exit_code = p.returncode
    if exit_code != 0:
        return None
    ret = output.decode().strip().split()
    if ret[0] == 'c':
        return ret[1:]
    return ret


def get_mpi_args(mpi_executable, compile_argument, link_argument):
    compile_args = call_subprocess([mpi_executable, compile_argument])
    link_args = call_subprocess([mpi_executable, link_argument])
    if compile_args is None:
        return None
    if len(compile_args) > 1 and compile_args[0] == 'clang':
        compile_args = compile_args[1:]
    if len(link_args) > 1 and link_args[0] == 'clang':
        link_args = link_args[1:]
    return compile_args, link_args


# Add MPI libraries
ext_options_mpi = copy.deepcopy(ext_options)
if RTDFLAG:
    ext_options['libraries'] = []
    ext_options['extra_link_args'] = []
    ext_options['extra_compile_args'].append('-DREADTHEDOCS')


# Set coverage options in .coveragerc
cov_installed = False
try:
    from coverage.config import HandyConfigParser
    cov_installed = True
except ImportError:
    pass
if cov_installed:
    # Read options
    covrc = '.coveragerc'
    cp = HandyConfigParser("")
    cp.read(covrc)
    # Exclude rules for all files
    if not cp.has_section('report'):
        cp.add_section('report')
    if cp.has_option('report', 'exclude_lines'):
        excl_list = cp.getlist('report', 'exclude_lines')
    else:
        excl_list = []
    # Exclude rules for Cython files
    if not cp.has_section('Cython.Coverage'):
        cp.add_section('Cython.Coverage')
    if cp.has_option('Cython.Coverage', 'exclude_lines'):
        cy_excl_list = cp.getlist('Cython.Coverage', 'exclude_lines')
    else:
        cy_excl_list = []
    # Funcs to add/rm rules
    def add_excl_rule(excl_list, new_rule):
        if new_rule not in excl_list:
            excl_list.append(new_rule)
        return excl_list
    def rm_excl_rule(excl_list, new_rule):
        if new_rule in excl_list:
            excl_list.remove(new_rule)
        return excl_list
    # Python version
    verlist = [2, 3]
    for v in verlist:
        vincl = 'pragma: Python %d' % v
        if PY_MAJOR_VERSION == v:
            excl_list = rm_excl_rule(excl_list, vincl)
        else:
            excl_list = add_excl_rule(excl_list, vincl)
    # Add new rules
    for r in excl_list:
        if r not in cy_excl_list:
            cy_excl_list.append(r)
    cp.set('report', 'exclude_lines', '\n'+'\n'.join(excl_list))
    cp.set('Cython.Coverage', 'exclude_lines', '\n'+'\n'.join(cy_excl_list))
    # Write
    with open(covrc, 'w') as fd:
        cp.write(fd)

# Cythonize modules
ext_modules = []

def make_cpp(cpp_file):
    if not os.path.isfile(cpp_file):
        open(cpp_file, 'a').close()
        assert(os.path.isfile(cpp_file))

ext_modules += [
    Extension(_package_name,
              sources=[os.path.join(_package_name, "py_read_lhalotree.pyx"),
                       os.path.join(_package_name, "read_lhalotree.c"),
                       os.path.join(_package_name, "utils.h")],
              **ext_options)]

class sdist(_sdist):
    # subclass setuptools source distribution builder to ensure cython
    # generated C files are included in source distribution and readme
    # is converted from markdown to restructured text.  See
    # http://stackoverflow.com/a/18418524/1382869
    def run(self):
        # Make sure the compiled Cython files in the distribution are
        # up-to-date

        try:
            import pypandoc
        except ImportError:
            raise RuntimeError(
                'Trying to create a source distribution without pypandoc. '
                'The readme will not render correctly on pypi without '
                'pypandoc so we are exiting.'
            )
        from Cython.Build import cythonize
        cythonize(ext_modules, **cyt_options)
        _sdist.run(self)

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (ImportError, IOError):
    with open('README.md') as file:
        long_description = file.read()

setup(name=_package_name,
      packages=find_packages(),
      include_package_data=True,
      version='0.2.5.dev0',
      description='Cython wrapper for C routines that read LHaloTree files',
      long_description=long_description,
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      url='https://github.com/langmm/LHaloTreeReader',
      keywords=['merger tree', 'halo tree', 'LHaloTree'],
      classifiers=["Programming Language :: Python",
                   "Programming Language :: C",
                   "Operating System :: OS Independent",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Development Status :: 3 - Alpha"],
      license='MIT',
      zip_safe=False,
      cmdclass={'build_ext': build_ext, 'sdist': sdist},
      ext_modules=cythonize(ext_modules, **cyt_options))

