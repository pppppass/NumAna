#include "fft.h"

static PyObject* wrapper_trans_dft(PyObject* self, PyObject* args)
{
    int size;
    PyObject* vec_obj;
    if (!PyArg_ParseTuple(
        args, "iO!",
        &size,
        &PyArray_Type, &vec_obj
    ))
        return NULL;
    
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!vec_arr)
        return NULL;
    
    double* vec = PyArray_DATA(vec_arr);

    double* work = malloc((1<<size) * sizeof(double));

    trans_dft(size, vec, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(vec_arr);
    Py_DECREF(vec_arr);
    
    return Py_BuildValue("O", vec_obj);
}

static PyObject* wrapper_trans_fft(PyObject* self, PyObject* args)
{
    int size;
    PyObject* vec_obj;
    if (!PyArg_ParseTuple(
        args, "iO!",
        &size,
        &PyArray_Type, &vec_obj
    ))
        return NULL;
    
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!vec_arr)
        return NULL;
    
    double* vec = PyArray_DATA(vec_arr);

    trans_fft(size, vec);

    PyArray_ResolveWritebackIfCopy(vec_arr);
    Py_DECREF(vec_arr);
    
    return Py_BuildValue("O", vec_obj);
}

static PyObject* wrapper_trans_ifft(PyObject* self, PyObject* args)
{
    int size;
    PyObject* vec_obj;
    if (!PyArg_ParseTuple(
        args, "iO!",
        &size,
        &PyArray_Type, &vec_obj
    ))
        return NULL;
    
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!vec_arr)
        return NULL;
    
    double* vec = PyArray_DATA(vec_arr);

    trans_ifft(size, vec);

    PyArray_ResolveWritebackIfCopy(vec_arr);
    Py_DECREF(vec_arr);
    
    return Py_BuildValue("O", vec_obj);
}

static PyObject* wrapper_solve_diff_3(PyObject* self, PyObject* args)
{
    int size;
    double alpha, beta, gamma;
    PyObject* vec_obj;
    if (!PyArg_ParseTuple(
        args, "iO!ddd",
        &size,
        &PyArray_Type, &vec_obj,
        &alpha, &beta, &gamma
    ))
        return NULL;
    
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!vec_arr)
        return NULL;
    
    double* vec = PyArray_DATA(vec_arr);

    solve_diff_3(size, vec, alpha, beta, gamma);

    PyArray_ResolveWritebackIfCopy(vec_arr);
    Py_DECREF(vec_arr);
    
    return Py_BuildValue("O", vec_obj);
}

static PyObject* wrapper_solve_spec_3(PyObject* self, PyObject* args)
{
    int size;
    double alpha, beta, gamma;
    PyObject* vec_obj;
    if (!PyArg_ParseTuple(
        args, "iO!ddd",
        &size,
        &PyArray_Type, &vec_obj,
        &alpha, &beta, &gamma
    ))
        return NULL;
    
    PyArrayObject* vec_arr = (PyArrayObject*)PyArray_FROM_OTF(vec_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!vec_arr)
        return NULL;
    
    double* vec = PyArray_DATA(vec_arr);

    solve_spec_3(size, vec, alpha, beta, gamma);

    PyArray_ResolveWritebackIfCopy(vec_arr);
    Py_DECREF(vec_arr);
    
    return Py_BuildValue("O", vec_obj);
}

static PyMethodDef methods[] = 
{
    {"trans_dft", wrapper_trans_dft, METH_VARARGS, NULL},
    {"trans_fft", wrapper_trans_fft, METH_VARARGS, NULL},
    {"trans_ifft", wrapper_trans_ifft, METH_VARARGS, NULL},
    {"solve_diff_3", wrapper_solve_diff_3, METH_VARARGS, NULL},
    {"solve_spec_3", wrapper_solve_spec_3, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "fft", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_fft(void)
{
    import_array();
    return PyModule_Create(&table);
}
