#include "rand.h"

static PyObject* wrapper_gen_rand_gauss_box(PyObject* self, PyObject* args)
{
    int num, seed;
    PyObject* x_obj, * y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!i",
        &num,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj,
        &seed
    ))
        return NULL;
    
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!x_arr || !y_arr)
        return NULL;

    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr);

    srand48(seed);

    gen_rand_gauss_box(num, x, y);

    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);

    return Py_BuildValue("OO", x_obj, y_obj);
}

static PyObject* wrapper_gen_rand_gauss_rej(PyObject* self, PyObject* args)
{
    int num, seed;
    PyObject* x_obj, * y_obj;

    if(!PyArg_ParseTuple(
        args, "iO!O!i",
        &num,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj,
        &seed
    ))
        return NULL;
    
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!x_arr || !y_arr)
        return NULL;

    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr);

    srand48(seed);

    gen_rand_gauss_rej(num, x, y);

    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);

    return Py_BuildValue("OO", x_obj, y_obj);
}

static PyMethodDef methods[] = 
{
    {"gen_rand_gauss_box", wrapper_gen_rand_gauss_box, METH_VARARGS, NULL},
    {"gen_rand_gauss_rej", wrapper_gen_rand_gauss_rej, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "rand", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_rand(void)
{
    import_array();
    return PyModule_Create(&table);
}
