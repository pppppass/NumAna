#include "intg.h"

static PyObject* wrapper_intg_ode1_lor(PyObject* self, PyObject* args)
{
    int num_step;
    double step, x0, y0, z0, sigma, rho, beta;
    PyObject* x_obj, * y_obj, * z_obj;
    if (!PyArg_ParseTuple(
        args, "idddddddO!O!O!",
        &num_step, &step,
        &x0, &y0, &z0,
        &sigma, &rho, &beta,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj
    ))
        return NULL;
    
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_arr || !y_arr || !z_arr)
        return NULL;
    
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr);

    intg_ode1_lor(num_step, step, x0, y0, z0, sigma, rho, beta, x, y, z);

    PyArray_ResolveWritebackIfCopy(z_arr);
    Py_DECREF(z_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    
    return Py_BuildValue("OOO", x_obj, y_obj, z_obj);
}

static PyObject* wrapper_intg_ode4_lor(PyObject* self, PyObject* args)
{
    int num_step;
    double step, x0, y0, z0, sigma, rho, beta;
    PyObject* x_obj, * y_obj, * z_obj;
    if (!PyArg_ParseTuple(
        args, "idddddddO!O!O!",
        &num_step, &step,
        &x0, &y0, &z0,
        &sigma, &rho, &beta,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj
    ))
        return NULL;
    
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_arr || !y_arr || !z_arr)
        return NULL;
    
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr);

    intg_ode4_lor(num_step, step, x0, y0, z0, sigma, rho, beta, x, y, z);

    PyArray_ResolveWritebackIfCopy(z_arr);
    Py_DECREF(z_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    
    return Py_BuildValue("OOO", x_obj, y_obj, z_obj);
}

static PyMethodDef methods[] = 
{
    {"intg_ode1_lor", wrapper_intg_ode1_lor, METH_VARARGS, NULL},
    {"intg_ode4_lor", wrapper_intg_ode4_lor, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "intg", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_intg(void)
{
    import_array();
    return PyModule_Create(&table);
}
