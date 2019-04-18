#include "quad.h"

static PyObject* wrapper_quad_mid_unif(PyObject* self, PyObject* args)
{
    int num_node;
    double x_low, x_high;
    PyObject* y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iddO!",
        &num_node, &x_low, &x_high, &PyArray_Type, &y_node_obj
    ))
        return NULL;
    
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!y_node_arr)
        return NULL;
    
    const double* y_node = PyArray_DATA(y_node_arr);

    double r = quad_mid_unif(num_node, x_low, x_high, y_node);

    Py_DECREF(y_node_arr);
    
    return Py_BuildValue("d", r);
}

static PyObject* wrapper_quad_trap_unif(PyObject* self, PyObject* args)
{
    int num_node;
    double x_low, x_high;
    PyObject* y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iddO!",
        &num_node, &x_low, &x_high, &PyArray_Type, &y_node_obj
    ))
        return NULL;
    
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!y_node_arr)
        return NULL;
    
    const double* y_node = PyArray_DATA(y_node_arr);

    double ret = quad_trap_unif(num_node, x_low, x_high, y_node);

    Py_DECREF(y_node_arr);
    
    return Py_BuildValue("d", ret);
}

static PyObject* wrapper_quad_simp_unif(PyObject* self, PyObject* args)
{
    int num_node;
    double x_low, x_high;
    PyObject* y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iddO!",
        &num_node, &x_low, &x_high, &PyArray_Type, &y_node_obj
    ))
        return NULL;
    
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!y_node_arr)
        return NULL;
    
    const double* y_node = PyArray_DATA(y_node_arr);

    double ret = quad_simp_unif(num_node, x_low, x_high, y_node);

    Py_DECREF(y_node_arr);
    
    return Py_BuildValue("d", ret);
}

static PyObject* wrapper_quad_romb_unif(PyObject* self, PyObject* args)
{
    int num_node, num_order;
    double x_low, x_high;
    PyObject* y_node_obj;
    if (!PyArg_ParseTuple(
        args, "iiddO!",
        &num_node, &num_order, &x_low, &x_high, &PyArray_Type, &y_node_obj
    ))
        return NULL;
    
    PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!y_node_arr)
        return NULL;
    
    const double* y_node = PyArray_DATA(y_node_arr);

    double* work = malloc((num_order+1) * sizeof(double));

    double ret = quad_romb_unif(num_node, num_order, x_low, x_high, y_node, work);

    free(work);

    Py_DECREF(y_node_arr);
    
    return Py_BuildValue("d", ret);
}

static PyObject* wrapper_calc_lagu_para(PyObject* self, PyObject* args)
{
    int order;
    PyObject* zero_obj, * weight_obj;
    if (!PyArg_ParseTuple(
        args, "iO!O!",
        &order, &PyArray_Type, &zero_obj, &PyArray_Type, &weight_obj
    ))
        return NULL;
    
    PyArrayObject* zero_arr = (PyArrayObject*)PyArray_FROM_OTF(zero_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* weight_arr = (PyArrayObject*)PyArray_FROM_OTF(weight_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!zero_arr || !weight_arr)
        return NULL;
    
    double* zero = PyArray_DATA(zero_arr);
    double* weight = PyArray_DATA(weight_arr);

    // double* work = malloc(order * sizeof(double));

    // calc_lagu_zero(order, zero, work);
    calc_lagu_para(order, zero, weight);

    // free(work);

    PyArray_ResolveWritebackIfCopy(weight_arr);
    Py_DECREF(weight_arr);
    PyArray_ResolveWritebackIfCopy(zero_arr);
    Py_DECREF(zero_arr);
    
    return Py_BuildValue("OO", zero_obj, weight_obj);
}

static PyObject* wrapper_calc_lege_para(PyObject* self, PyObject* args)
{
    int order;
    double x_low, x_high;
    PyObject* zero_obj, * weight_obj;
    if (!PyArg_ParseTuple(
        args, "iddO!O!",
        &order, &x_low, &x_high, &PyArray_Type, &zero_obj, &PyArray_Type, &weight_obj
    ))
        return NULL;
    
    PyArrayObject* zero_arr = (PyArrayObject*)PyArray_FROM_OTF(zero_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* weight_arr = (PyArrayObject*)PyArray_FROM_OTF(weight_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!zero_arr || !weight_arr)
        return NULL;
    
    double* zero = PyArray_DATA(zero_arr);
    double* weight = PyArray_DATA(weight_arr);

    // double* work = malloc(order * sizeof(double));

    // calc_lagu_zero(order, zero, work);
    calc_lege_para(order, x_low, x_high, zero, weight);

    // free(work);

    PyArray_ResolveWritebackIfCopy(weight_arr);
    Py_DECREF(weight_arr);
    PyArray_ResolveWritebackIfCopy(zero_arr);
    Py_DECREF(zero_arr);
    
    return Py_BuildValue("OO", zero_obj, weight_obj);
}

// static PyObject* wrapper_quad_lagu(PyObject* self, PyObject* args)
// {
//     int num_node;
//     PyObject* x_node_obj, * y_node_obj;
//     if (!PyArg_ParseTuple(
//         args, "iO!O!",
//         &num_node, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj
//     ))
//         return NULL;
    
//     PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

//     if (!x_node_arr || !y_node_arr)
//         return NULL;
    
//     const double* x_node = PyArray_DATA(x_node_arr);
//     const double* y_node = PyArray_DATA(y_node_arr);

//     double r = quad_lagu(num_node, x_node, y_node);

//     Py_DECREF(x_node_arr);
//     Py_DECREF(y_node_arr);
    
//     return Py_BuildValue("d", r);
// }

// static PyObject* wrapper_calc_lege_zero(PyObject* self, PyObject* args)
// {
//     int order;
//     double x_low, x_high;
//     PyObject* zero_obj;
//     if (!PyArg_ParseTuple(
//         args, "iddO!",
//         &order, &x_low, &x_high, &PyArray_Type, &zero_obj
//     ))
//         return NULL;
    
//     PyArrayObject* zero_arr = (PyArrayObject*)PyArray_FROM_OTF(zero_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

//     if (!zero_arr)
//         return NULL;
    
//     double* zero = PyArray_DATA(zero_arr);

//     double* work = malloc(order * sizeof(double));

//     calc_lege_zero(order, x_low, x_high, zero, work);

//     free(work);

//     PyArray_ResolveWritebackIfCopy(zero_arr);
//     Py_DECREF(zero_arr);
    
//     return Py_BuildValue("O", zero_obj);
// }

// static PyObject* wrapper_quad_lege(PyObject* self, PyObject* args)
// {
//     int num_node;
//     double x_low, x_high;
//     PyObject* x_node_obj, * y_node_obj;
//     if (!PyArg_ParseTuple(
//         args, "iddO!O!",
//         &num_node, &x_low, &x_high, &PyArray_Type, &x_node_obj, &PyArray_Type, &y_node_obj
//     ))
//         return NULL;
    
//     PyArrayObject* x_node_arr = (PyArrayObject*)PyArray_FROM_OTF(x_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     PyArrayObject* y_node_arr = (PyArrayObject*)PyArray_FROM_OTF(y_node_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

//     if (!x_node_arr || !y_node_arr)
//         return NULL;
    
//     const double* x_node = PyArray_DATA(x_node_arr);
//     const double* y_node = PyArray_DATA(y_node_arr);

//     double r = quad_lege(num_node, x_low, x_high, x_node, y_node);

//     Py_DECREF(x_node_arr);
//     Py_DECREF(y_node_arr);
    
//     return Py_BuildValue("d", r);
// }

static PyMethodDef methods[] = 
{
    {"quad_mid_unif", wrapper_quad_mid_unif, METH_VARARGS, NULL},
    {"quad_trap_unif", wrapper_quad_trap_unif, METH_VARARGS, NULL},
    {"quad_simp_unif", wrapper_quad_simp_unif, METH_VARARGS, NULL},
    {"quad_romb_unif", wrapper_quad_romb_unif, METH_VARARGS, NULL},
    // {"calc_lagu_zero", wrapper_calc_lagu_zero, METH_VARARGS, NULL},
    // // {"quad_lagu", wrapper_quad_lagu, METH_VARARGS, NULL},
    {"calc_lagu_para", wrapper_calc_lagu_para, METH_VARARGS, NULL},
    // {"calc_lege_zero", wrapper_calc_lege_zero, METH_VARARGS, NULL},
    // {"quad_lege", wrapper_quad_lege, METH_VARARGS, NULL},
    {"calc_lege_para", wrapper_calc_lege_para, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "quad", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_quad(void)
{
    import_array();
    return PyModule_Create(&table);
}
