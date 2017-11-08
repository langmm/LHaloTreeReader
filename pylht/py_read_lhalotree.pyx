from cpython cimport PyObject, Py_INCREF
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
np.import_array()


cdef get_lhalotree_dtype():
    r"""Get the numpy dtype corresponding to the lhalotree C struct.

    Returns:
        np.dtype: Corresponding structured data type.

    """
    cdef lhalotree _tmp
    dt = np.asarray(<lhalotree[:1]>(&_tmp)).dtype
    return dt


cdef int get_lhalotree_typenum(dtype):
    r"""Get the C-API typenum for PyArray_Descr corresponding to the numpy user
    defined lhalotree type.

    Args:
        dtype (np.dtype): Numpy data type to get type_num for.

    Returns:
        int: Type number for the registered PyArray_Descr.

    """
    dt = get_lhalotree_dtype()
    cdef PyArray_Descr* descr
    cdef int out = np.PyArray_DescrConverter(<PyObject*>dtype.str, &descr)
    cdef int type_num = np.PyArray_RegisterDataType(descr)
    return type_num


# Adapted from
# http://gael-varoquaux.info/programming/cython-example-of-exposing-c-computed-
# arrays-in-python-without-data-copies.html
cdef class LHaloTreeArray:
    cdef void* data_ptr
    cdef int size
    cdef int type_num

    cdef void set_data(self, int size, void* data_ptr, int type_num):
        r"""Set the data of the array.

        Args:
            size (int): Number of lhalotree structure elements in the array.
            data_ptr (void*): Pointer to C array of structures.

        """
        self.size = size
        self.type_num = type_num
        self.data_ptr = data_ptr

    def __array__(self):
        r"""Here we use the __array__ method, that is called when numpy
        tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape, self.type_num,
                                               self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """Frees the array. This is called by Python when all the
        references to the object are gone."""
        free(<void*>self.data_ptr)


def read_single_lhalotree(filename, treenum, dtype=None):
    r"""Read a single LHaloTree from the file.

    Args:
        filename (str): Full path to the file that a LHaloTree should be read
            from.
        treenum (int): Index of the tree that should be read. (Starts at 0)

    Returns:
        np.ndarray: Single element structure array with fields for the returned
            tree.

    """
    # TODO: Python 2/3 compat
    if dtype is None:
        dtype = get_lhalotree_dtype()
    else:
        dtype = np.dtype(dtype)
    cdef int type_num = get_lhalotree_typenum(dtype)
    cdef char* c_filename = filename.encode()
    cdef int32_t c_treenum = <int32_t>treenum
    # Pass header size and struct size (user options)
    # Pass reader for header that takes filename, returns headersize
    # and dictionary with header info minimum has to contain array with
    # ntree elements of nhalos
    # Pass NULL, -1 to ntree
    cdef void* tree = read_single_lhalotree(c_filename, c_treenum)
    array_wrapper = LHaloTreeArray()
    array_wrapper.set_data(1, <void*>tree, type_num)
    ndarray = np.array(array_wrapper, copy=False)
    ndarray.base = <PyObject*> array_wrapper
    Py_INCREF(array_wrapper)
    return ndarray
    
