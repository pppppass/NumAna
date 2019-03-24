#include "intp.h"

static PyObject* wrapper_intp_newton(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    PyObject* x_node_obj, * coef_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!iO!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &coef_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !coef_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* coef = PyArray_DATA(coef_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_newton(num_node, x_node, coef, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(coef_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_calc_newton_array(PyObject* self, PyObject* args)
{
    int num_node;
    PyObject* x_node_obj, * y_node_obj, * coef_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &PyArray_Type, & coef_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_INT, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !coef_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    double* coef = PyArray_DATA(coef_arr);

    calc_nowton_array(num_node, x_node, y_node, coef);

    PyArray_ResolveWritebackIfCopy(coef_arr);
    Py_DECREF(coef_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", coef_obj);
}

static PyMethodDef methods[] = 
{
    {"wrapper_intp_newton", wrapper_intp_newton, METH_VARARGS, NULL},
    {"wrapper_calc_newton_array", wrapper_calc_newton_array, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "intp", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_intp(void)
{
    import_array();
    return PyModule_Create(&table);
}
