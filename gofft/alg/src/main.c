#include "include.h"
#include <math.h>

static PyObject* dsp_goertzel(PyObject* self, PyObject* args)
{
    PyArrayObject *in_data;
    double k;

    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &in_data, &k))
        return NULL;
    if (in_data == NULL) return NULL;

    // Ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    in_data = PyArray_GETCONTIGUOUS(in_data);
    double *data = (double *)PyArray_DATA(in_data);

    // create output dimensions
    // last axis is replaced by 2 for (x, y, mag)
    npy_intp out_dim[NPY_MAXDIMS];
    int n_dim = PyArray_NDIM(in_data);
    memcpy(out_dim, PyArray_DIMS(in_data), n_dim*sizeof(npy_intp));
    long int data_len = out_dim[n_dim-1];
    npy_intp n_data = 1;
    for (int k = 0; k < n_dim-1; k++)
        n_data *= out_dim[k];
    out_dim[n_dim-1] = 2;

    PyObject *output = PyArray_SimpleNew(n_dim, out_dim, NPY_DOUBLE);
    double *out_res = (double *)PyArray_DATA((PyArrayObject *)output);

    for (npy_intp i_data = 0; i_data < n_data; i_data++)
    {
        goertzel(data, data_len, k, out_res);
        data += data_len;
        out_res += 2;
    }

    // Decrease the reference count of ap.
    Py_DECREF(in_data);
    return output;
}

static PyObject* dsp_goertzel_m(PyObject* self, PyObject* args)
{
    PyArrayObject *in_data;
    PyArrayObject *in_ft;
    int filter_size;
    int fs;

    if (!PyArg_ParseTuple(args, "O!iO!i",
        &PyArray_Type, &in_data, &fs, &PyArray_Type, &in_ft, &filter_size))
        return NULL;
    if (in_data == NULL) return NULL;
    if (in_ft == NULL) return NULL;

    in_data = PyArray_GETCONTIGUOUS(in_data);

    double *data = (double *)PyArray_DATA(in_data);
    long int data_len = (long int)PyArray_DIM(in_data, 0);
    double *ft = (double *)PyArray_DATA(in_ft);
    int ft_num = (int)PyArray_DIM(in_ft, 0);

    PyObject *output = PyArray_SimpleNew(1, PyArray_DIMS(in_ft), NPY_DOUBLE);
    double *mag = (double *)PyArray_DATA((PyArrayObject *)output);

    goertzel_m(data, data_len, fs, ft, ft_num, filter_size, mag);

    Py_DECREF(in_data);
    return output;
}

static PyObject* dsp_goertzel_rng(PyObject* self, PyObject* args)
{
    PyArrayObject *ap;
    int filter_size, fs;
    double ft;
    double rng;
    long data_len;
    double *data;

    double magnitude;

    if (!PyArg_ParseTuple(args, "O!idid",
        &PyArray_Type, &ap, &fs, &ft, &filter_size, &rng)) {
        return NULL;
    }
    if (ap == NULL) return NULL;

    ap = PyArray_GETCONTIGUOUS(ap);

    data = (double *)PyArray_DATA(ap);
    data_len = (long)PyArray_DIM(ap, 0);

    magnitude = goertzel_rng(data, data_len, fs, ft, filter_size, rng);

    Py_DECREF(ap);
    return Py_BuildValue("d", magnitude);
}

/* Set up the methods table */
static PyMethodDef methods[] = {
    {
        "goertzel", dsp_goertzel, // Python name, C name
        METH_VARARGS, // input parameters
        "Goertzel algorithm." // doc string
    },
    {
        "goertzel_m", dsp_goertzel_m,
        METH_VARARGS,
        "Goertzel algorithm for multiple target frequency."
    },
    {
        "goertzel_rng", dsp_goertzel_rng,
        METH_VARARGS,
        "Goertzel algorithm for specific frequency range."
    },
    {NULL, NULL, 0, NULL} // sentinel
};

/* Initialize module */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "dsp_ext",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyMODINIT_FUNC PyInit_dsp_ext(void)
{
    import_array(); // Must be called for NumPy.
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initdsp_ext(void)
{
    (void)Py_InitModule("dsp_ext", methods);
    import_array(); // Must be called for NumPy.
}
#endif
