#include "opt.h"

static PyObject* wrapper_calc_val_sos_2d_dist(PyObject* self, PyObject* args)
{
    int num_node, num_res;
    PyObject* res_i_obj, * res_j_obj, * x_obj, * y_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!O!O!",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &w_obj
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!res_i_arr || !res_j_arr || !x_arr || !y_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr);
    const double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * w = PyArray_DATA(w_arr);

    double f = calc_val_sos_2d_dist(num_node, num_res, res_i, res_j, x, y, w);

    Py_DECREF(w_arr);
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("d", f);
}

static PyObject* wrapper_opt_gauss_sos_2d_grad_nest(PyObject* self, PyObject* args)
{
    int num_node, num_res, num_class, num_iter;
    double eta;
    PyObject* res_i_obj, * res_j_obj, * class_node_obj, * x_obj, * y_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!iO!O!O!O!di",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj,
        &num_class,
        &PyArray_Type, &class_node_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &w_obj,
        &eta, &num_iter
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* class_node_arr = (PyArrayObject*)PyArray_FROM_OTF(class_node_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !class_node_arr || !x_arr || !y_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr);
    const int* class_node = PyArray_DATA(class_node_arr);
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * w = PyArray_DATA(w_arr);

    double* work = malloc((4*num_node+3*num_class) * sizeof(double));

    opt_gauss_sos_2d_grad_nest(num_node, num_res, res_i, res_j, num_class, class_node, x, y, w, eta, num_iter, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(w_arr);
    Py_DECREF(w_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    Py_DECREF(class_node_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("OOO", x_arr, y_arr, w_arr);
}

static PyObject* wrapper_calc_func_2d_dist(PyObject* self, PyObject* args)
{
    int num_node, num_res;
    PyObject* res_i_obj, * res_j_obj, * x_obj, * y_obj, * w_obj, * e_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!O!O!O!",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &w_obj,
        &PyArray_Type, &e_obj
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* e_arr = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !x_arr || !y_arr || !w_arr || !e_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr);
    const double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * w = PyArray_DATA(w_arr);
    double* e = PyArray_DATA(e_arr);

    calc_func_2d_dist(num_node, num_res, res_i, res_j, x, y, w, e);

    PyArray_ResolveWritebackIfCopy(e_arr);
    Py_DECREF(e_arr);
    Py_DECREF(w_arr);
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("O", e_obj);
}

static PyObject* wrapper_opt_gauss_2d_newt(PyObject* self, PyObject* args)
{
    int num_node, num_res, num_class, num_iter;
    double eta;
    PyObject* res_i_obj, * res_j_obj, * class_node_obj, * x_obj, * y_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!iO!O!O!O!di",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj,
        &num_class,
        &PyArray_Type, &class_node_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &w_obj,
        &eta, &num_iter
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* class_node_arr = (PyArrayObject*)PyArray_FROM_OTF(class_node_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !class_node_arr || !x_arr || !y_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr);
    const int* class_node = PyArray_DATA(class_node_arr);
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * w = PyArray_DATA(w_arr);

    double* work = malloc(((num_res+3)*(2*num_node+num_class)) * sizeof(double));
    
    opt_gauss_2d_newt(num_node, num_res, res_i, res_j, num_class, class_node, x, y, w, eta, num_iter, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(w_arr);
    Py_DECREF(w_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    Py_DECREF(class_node_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("OOO", x_arr, y_arr, w_arr);
}

static PyObject* wrapper_calc_val_sos_3d_dist(PyObject* self, PyObject* args)
{
    int num_node, num_res;
    PyObject* res_i_obj, * res_j_obj, * res_k_obj, * x_obj, * y_obj, * z_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!O!O!O!O!",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj, &PyArray_Type, &res_k_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj, &PyArray_Type, &w_obj
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_k_arr = (PyArrayObject*)PyArray_FROM_OTF(res_k_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!res_i_arr || !res_j_arr || !res_k_arr || !x_arr || !y_arr || !z_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr), * res_k = PyArray_DATA(res_k_arr);
    const double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr), * w = PyArray_DATA(w_arr);

    double f = calc_val_sos_3d_dist(num_node, num_res, res_i, res_j, res_k, x, y, z, w);

    Py_DECREF(w_arr);
    Py_DECREF(z_arr);
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    Py_DECREF(res_k_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("d", f);
}

static PyObject* wrapper_opt_gauss_sos_3d_grad_nest(PyObject* self, PyObject* args)
{
    int num_node, num_res, num_class, num_iter;
    double eta;
    PyObject* res_i_obj, * res_j_obj, * res_k_obj, * class_node_obj, * x_obj, * y_obj, * z_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!iO!O!O!O!O!di",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj, &PyArray_Type, &res_k_obj,
        &num_class,
        &PyArray_Type, &class_node_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj, &PyArray_Type, &w_obj,
        &eta, &num_iter
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_k_arr = (PyArrayObject*)PyArray_FROM_OTF(res_k_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* class_node_arr = (PyArrayObject*)PyArray_FROM_OTF(class_node_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !res_k_arr || !class_node_arr || !x_arr || !y_arr || !z_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr), * res_k = PyArray_DATA(res_k_arr);
    const int* class_node = PyArray_DATA(class_node_arr);
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr), * w = PyArray_DATA(w_arr);

    double* work = malloc((6*num_node+3*num_class) * sizeof(double));

    opt_gauss_sos_3d_grad_nest(num_node, num_res, res_i, res_j, res_k, num_class, class_node, x, y, z, w, eta, num_iter, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(w_arr);
    Py_DECREF(w_arr);
    PyArray_ResolveWritebackIfCopy(z_arr);
    Py_DECREF(z_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    Py_DECREF(class_node_arr);
    Py_DECREF(res_k_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("OOOO", x_arr, y_arr, z_arr, w_arr);
}

static PyObject* wrapper_calc_func_3d_dist(PyObject* self, PyObject* args)
{
    int num_node, num_res;
    PyObject* res_i_obj, * res_j_obj, * res_k_obj, * x_obj, * y_obj, * z_obj, * w_obj, * e_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!O!O!O!O!O!",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj, &PyArray_Type, &res_k_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj, &PyArray_Type, &w_obj,
        &PyArray_Type, &e_obj
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_k_arr = (PyArrayObject*)PyArray_FROM_OTF(res_k_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* e_arr = (PyArrayObject*)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !res_k_arr || !x_arr || !y_arr || !z_arr || !w_arr || !e_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr), * res_k = PyArray_DATA(res_k_arr);
    const double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr), * w = PyArray_DATA(w_arr);
    double* e = PyArray_DATA(e_arr);

    calc_func_3d_dist(num_node, num_res, res_i, res_j, res_k, x, y, z, w, e);

    PyArray_ResolveWritebackIfCopy(e_arr);
    Py_DECREF(e_arr);
    Py_DECREF(w_arr);
    Py_DECREF(z_arr);
    Py_DECREF(y_arr);
    Py_DECREF(x_arr);
    Py_DECREF(res_k_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("O", e_obj);
}

static PyObject* wrapper_opt_gauss_3d_newt(PyObject* self, PyObject* args)
{
    int num_node, num_res, num_class, num_iter;
    double eta;
    PyObject* res_i_obj, * res_j_obj, * res_k_obj, * class_node_obj, * x_obj, * y_obj, * z_obj, * w_obj;
    if (!PyArg_ParseTuple(
        args, "iiO!O!O!iO!O!O!O!O!di",
        &num_node, &num_res,
        &PyArray_Type, &res_i_obj, &PyArray_Type, &res_j_obj, &PyArray_Type, &res_k_obj,
        &num_class,
        &PyArray_Type, &class_node_obj,
        &PyArray_Type, &x_obj, &PyArray_Type, &y_obj, &PyArray_Type, &z_obj, &PyArray_Type, &w_obj,
        &eta, &num_iter
    ))
        return NULL;
    
    PyArrayObject* res_i_arr = (PyArrayObject*)PyArray_FROM_OTF(res_i_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_j_arr = (PyArrayObject*)PyArray_FROM_OTF(res_j_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* res_k_arr = (PyArrayObject*)PyArray_FROM_OTF(res_k_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* class_node_arr = (PyArrayObject*)PyArray_FROM_OTF(class_node_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);

    if (!res_i_arr || !res_j_arr || !res_k_arr || !class_node_arr || !x_arr || !y_arr || !z_arr || !w_arr)
        return NULL;
    
    const int* res_i = PyArray_DATA(res_i_arr), * res_j = PyArray_DATA(res_j_arr), * res_k = PyArray_DATA(res_k_arr);
    const int* class_node = PyArray_DATA(class_node_arr);
    double* x = PyArray_DATA(x_arr), * y = PyArray_DATA(y_arr), * z = PyArray_DATA(z_arr), * w = PyArray_DATA(w_arr);

    double* work = malloc(((num_res+3)*(3*num_node+num_class)) * sizeof(double));
    
    opt_gauss_3d_newt(num_node, num_res, res_i, res_j, res_k, num_class, class_node, x, y, z, w, eta, num_iter, work);

    free(work);

    PyArray_ResolveWritebackIfCopy(w_arr);
    Py_DECREF(w_arr);
    PyArray_ResolveWritebackIfCopy(z_arr);
    Py_DECREF(z_arr);
    PyArray_ResolveWritebackIfCopy(y_arr);
    Py_DECREF(y_arr);
    PyArray_ResolveWritebackIfCopy(x_arr);
    Py_DECREF(x_arr);
    Py_DECREF(class_node_arr);
    Py_DECREF(res_k_arr);
    Py_DECREF(res_j_arr);
    Py_DECREF(res_i_arr);
    
    return Py_BuildValue("OOOO", x_arr, y_arr, z_arr, w_arr);
}

static PyMethodDef methods[] = 
{
    {"calc_val_sos_2d_dist", wrapper_calc_val_sos_2d_dist, METH_VARARGS, NULL},
    {"opt_gauss_sos_2d_grad_nest", wrapper_opt_gauss_sos_2d_grad_nest, METH_VARARGS, NULL},
    {"calc_func_2d_dist", wrapper_calc_func_2d_dist, METH_VARARGS, NULL},
    {"opt_gauss_2d_newt", wrapper_opt_gauss_2d_newt, METH_VARARGS, NULL},
    {"calc_val_sos_3d_dist", wrapper_calc_val_sos_3d_dist, METH_VARARGS, NULL},
    {"opt_gauss_sos_3d_grad_nest", wrapper_opt_gauss_sos_3d_grad_nest, METH_VARARGS, NULL},
    {"calc_func_3d_dist", wrapper_calc_func_3d_dist, METH_VARARGS, NULL},
    {"opt_gauss_3d_newt", wrapper_opt_gauss_3d_newt, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "opt", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_opt(void)
{
    import_array();
    return PyModule_Create(&table);
}
