#include "intp.h"

static PyObject* wrapper_calc_newt_arr(PyObject* self, PyObject* args)
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
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !coef_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    double* coef = PyArray_DATA(coef_arr);

    calc_newt_arr(num_node, x_node, y_node, coef);

    PyArray_ResolveWritebackIfCopy(coef_arr);
    Py_DECREF(coef_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", coef_obj);
}

static PyObject* wrapper_intp_newt(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    PyObject* x_node_obj, * coef_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!iO!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &coef_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* coef_arr = (PyArrayObject*)PyArray_FROM_OTF(coef_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !coef_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* coef = PyArray_DATA(coef_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_newt(num_node, x_node, coef, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(coef_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_intp_lagr(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    PyObject* x_node_obj, * y_node_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!iO!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_lagr(num_node, x_node, y_node, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_intp_lin_unif(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    double x_low, x_high;
    PyObject* y_node_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iddO!iO!O!",
        &num_node, &x_low, &x_high, &PyArray_Type, &y_node_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!y_node_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* y_node = PyArray_DATA(y_node_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_lin_unif(num_node, x_low, x_high, y_node, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(y_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_intp_lin(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    PyObject* x_node_obj, * y_node_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!iO!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_lin(num_node, x_node, y_node, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_intp_cub(PyObject* self, PyObject* args)
{
    int num_node, num_req;
    PyObject* x_node_obj, * y_node_obj, * d_y_node_obj, * x_req_obj, * y_req_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!O!iO!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &PyArray_Type, &d_y_node_obj, &num_req, &PyArray_Type, &x_req_obj, &PyArray_Type, &y_req_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* d_y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(d_y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_req_arr = (PyArrayObject*)PyArray_FROM_OTF(x_req_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_req_arr = (PyArrayObject*)PyArray_FROM_OTF(y_req_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !d_y_node_arr || !x_req_arr || !y_req_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    const double* d_y_node = PyArray_DATA(d_y_node_arr);
    const double* x_req = PyArray_DATA(x_req_arr);
    double* y_req = PyArray_DATA(y_req_arr);

    intp_cub(num_node, x_node, y_node, d_y_node, num_req, x_req, y_req);

    PyArray_ResolveWritebackIfCopy(y_req_arr);
    Py_DECREF(y_req_arr);
    Py_DECREF(x_req_arr);
    Py_DECREF(d_y_node_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", y_req_obj);
}

static PyObject* wrapper_calc_spl_cub_d_y_nat(PyObject* self, PyObject* args)
{
    int num_node;
    PyObject* x_node_obj, * y_node_obj, * d_y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!O!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &PyArray_Type, &d_y_node_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* d_y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(d_y_node_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !d_y_node_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    double* d_y_node = PyArray_DATA(d_y_node_arr);

    double* work = malloc((3*num_node+1) * sizeof(double));

    calc_spl_cub_d_y_nat(num_node, x_node, y_node, d_y_node, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(d_y_node_arr);
    Py_DECREF(d_y_node_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", d_y_node_obj);
}

static PyObject* wrapper_calc_spl_cub_d_y_coe(PyObject* self, PyObject* args)
{
    int num_node;
    double d_y_low, d_y_high;
    PyObject* x_node_obj, * y_node_obj, * d_y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!ddO!",
        &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj, &d_y_low, &d_y_high, &PyArray_Type, &d_y_node_obj
    ))
        return NULL;
    
    PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* d_y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(d_y_node_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!x_node_arr || !y_node_arr || !d_y_node_arr)
        return NULL;
    
    const double* x_node = PyArray_DATA(x_node_arr);
    const double* y_node = PyArray_DATA(y_node_arr);
    double* d_y_node = PyArray_DATA(d_y_node_arr);

    double* work = malloc((3*num_node-1) * sizeof(double));

    calc_spl_cub_d_y_coe(num_node, x_node, y_node, d_y_low, d_y_high, d_y_node, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(d_y_node_arr);
    Py_DECREF(d_y_node_arr);
    Py_DECREF(y_node_arr);
    Py_DECREF(x_node_arr);
    
    return Py_BuildValue("O", d_y_node_obj);
}

static PyMethodDef methods[] = 
{
    {"calc_newt_arr", wrapper_calc_newt_arr, METH_VARARGS, NULL},
    {"intp_newt", wrapper_intp_newt, METH_VARARGS, NULL},
    {"intp_lagr", wrapper_intp_lagr, METH_VARARGS, NULL},
    {"intp_lin_unif", wrapper_intp_lin_unif, METH_VARARGS, NULL},
    {"intp_lin", wrapper_intp_lin, METH_VARARGS, NULL},
    {"intp_cub", wrapper_intp_cub, METH_VARARGS, NULL},
    {"calc_spl_cub_d_y_nat", wrapper_calc_spl_cub_d_y_nat, METH_VARARGS, NULL},
    {"calc_spl_cub_d_y_coe", wrapper_calc_spl_cub_d_y_coe, METH_VARARGS, NULL},
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
