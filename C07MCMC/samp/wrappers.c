#include "samp.h"

static PyObject* wrapper_driver_samp_ising_single_2d(PyObject* self, PyObject* args)
{
    int n;
    double t, h;
    int iter, rep, size_buf, seed, thread;
    PyObject* sites_obj;

    if(!PyArg_ParseTuple(
        args, "iddiiiiiO!",
        &n, &t, &h, &iter, &rep, &size_buf, &seed, &thread,
        &PyArray_Type, &sites_obj
    ))
        return NULL;
    
    PyArrayObject* sites_arr = (PyArrayObject*)PyArray_FROM_OTF(sites_obj, NPY_INT, NPY_ARRAY_INOUT_ARRAY2);
    
    if (!sites_arr)
        return NULL;

    int* q = PyArray_DATA(sites_arr);

    omp_set_num_threads(thread);

    driver_samp_ising_single_2d(n, t, h, iter, rep, size_buf, seed, q);

    PyArray_ResolveWritebackIfCopy(sites_arr);
    Py_DECREF(sites_arr);

    return Py_BuildValue("");
}

static PyObject* wrapper_driver_samp_ising_2d(PyObject* self, PyObject* args)
{
    int n;
    double t, h;
    int iter, start, rep, size_buf, seed, thread;

    if(!PyArg_ParseTuple(
        args, "iddiiiiii",
        &n, &t, &h, &iter, &start, &rep, &size_buf, &seed, &thread
    ))
        return NULL;

    omp_set_num_threads(thread);

    double m1 = 0.0, m2 = 0.0, ma1 = 0.0, ma2 = 0.0, u1 = 0.0, u2 = 0.0, c1 = 0.0, c2 = 0.0;

    driver_samp_ising_2d(n, t, h, iter, start, rep, size_buf, seed, &m1, &m2, &ma1, &ma2, &u1, &u2, &c1, &c2);

    return Py_BuildValue("dddddddd", m1, m2, ma1, ma2, u1, u2, c1, c2);
}

static PyObject* wrapper_driver_samp_ising_kin_2d(PyObject* self, PyObject* args)
{
    int n;
    double t, h;
    int iter, start, rep, size_buf, seed, thread;

    if(!PyArg_ParseTuple(
        args, "iddiiiiii",
        &n, &t, &h, &iter, &start, &rep, &size_buf, &seed, &thread
    ))
        return NULL;

    omp_set_num_threads(thread);

    double m1 = 0.0, m2 = 0.0, ma1 = 0.0, ma2 = 0.0, u1 = 0.0, u2 = 0.0, c1 = 0.0, c2 = 0.0;

    driver_samp_ising_kin_2d(n, t, h, iter, start, rep, size_buf, seed, &m1, &m2, &ma1, &ma2, &u1, &u2, &c1, &c2);

    return Py_BuildValue("dddddddd", m1, m2, ma1, ma2, u1, u2, c1, c2);
}

static PyObject* wrapper_driver_samp_ising_kin_3d(PyObject* self, PyObject* args)
{
    int n;
    double t, h;
    int iter, start, rep, size_buf, seed, thread;

    if(!PyArg_ParseTuple(
        args, "iddiiiiii",
        &n, &t, &h, &iter, &start, &rep, &size_buf, &seed, &thread
    ))
        return NULL;

    omp_set_num_threads(thread);

    double m1 = 0.0, m2 = 0.0, ma1 = 0.0, ma2 = 0.0, u1 = 0.0, u2 = 0.0, c1 = 0.0, c2 = 0.0;

    driver_samp_ising_kin_3d(n, t, h, iter, start, rep, size_buf, seed, &m1, &m2, &ma1, &ma2, &u1, &u2, &c1, &c2);

    return Py_BuildValue("dddddddd", m1, m2, ma1, ma2, u1, u2, c1, c2);
}

static PyMethodDef methods[] = 
{
    {"driver_samp_ising_2d", wrapper_driver_samp_ising_2d, METH_VARARGS, NULL},
    {"driver_samp_ising_single_2d", wrapper_driver_samp_ising_single_2d, METH_VARARGS, NULL},
    {"driver_samp_ising_kin_2d", wrapper_driver_samp_ising_kin_2d, METH_VARARGS, NULL},
    {"driver_samp_ising_kin_3d", wrapper_driver_samp_ising_kin_3d, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "samp", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_samp(void)
{
    import_array();
    return PyModule_Create(&table);
}
