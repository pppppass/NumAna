#include "intp.h"

static PyObject* wrapper_test(PyObject* self, PyObject* args)
{
    int n;

    PyObject* vec_obj = NULL;
    if (!PyArg_ParseTuple(
        args, "i",
        &n
    ))
        return NULL;
    
    double r = test(n);
    
    return Py_BuildValue("d", r);
}

static PyMethodDef methods[] = 
{
    {"wrapper_test", wrapper_test, METH_VARARGS, NULL},
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
