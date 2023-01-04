#include "include.h"
#include <math.h>

static PyObject* dsp_goertzel_template(
    PyObject* self, PyObject* args, goertzel_fun_t *fun, goertzel_fun_t *fun_cx)
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
    // last axis is removed, replaced by complex data i&q
    npy_intp out_dim[NPY_MAXDIMS];
    int n_dim = PyArray_NDIM(in_data);
    memcpy(out_dim, PyArray_DIMS(in_data), n_dim*sizeof(npy_intp));
    long int data_len = out_dim[n_dim-1];
    npy_intp n_data = 1;
    for (int k = 0; k < n_dim-1; k++)
        n_data *= out_dim[k];

    PyObject *output = PyArray_SimpleNew(n_dim-1, out_dim, NPY_COMPLEX128);
    double *out_res = (double *)PyArray_DATA((PyArrayObject *)output);

    if (PyArray_ISCOMPLEX(in_data) && fun_cx != NULL)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            fun_cx(data, data_len, k, out_res);
            data += data_len*2;
            out_res += 2;
        }
    else if (!PyArray_ISCOMPLEX(in_data) && fun != NULL)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            fun(data, data_len, k, out_res);
            data += data_len;
            out_res += 2;
        }

    // Decrease the reference count of ap.
    Py_DECREF(in_data);
    return output;
}

static PyObject* dsp_goertzel_dft_template(
    PyObject* self, PyObject* args, goertzel_fun_t *fun, goertzel_fun_t *fun_cx)
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
    int n_dim = PyArray_NDIM(in_data);
    npy_intp *out_dim = PyArray_DIMS(in_data);
    long int data_len = out_dim[n_dim-1];
    npy_intp n_data = 1;
    for (int k = 0; k < n_dim-1; k++)
        n_data *= out_dim[k];

    PyObject *output = PyArray_SimpleNew(n_dim, out_dim, NPY_COMPLEX128);
    double *out_res = (double *)PyArray_DATA((PyArrayObject *)output);

    if (PyArray_ISCOMPLEX(in_data) && fun_cx != NULL)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            fun_cx(data, data_len, k, out_res);
            data += data_len*2;
            out_res += data_len*2;
        }
    else if (!PyArray_ISCOMPLEX(in_data) && fun != NULL)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            fun(data, data_len, k, out_res);
            data += data_len;
            out_res += data_len*2;
        }

    // Decrease the reference count of ap.
    Py_DECREF(in_data);
    return output;
}

static PyObject* dsp_goertzel(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel, &goertzel_cx);
}

static PyObject* dsp_goertzel_rad2(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad2, NULL);
}

static PyObject* dsp_goertzel_rad2_sse(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad2_sse, NULL);
}

static PyObject* dsp_goertzel_rad4(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad4, NULL);
}

static PyObject* dsp_goertzel_rad4_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad4_avx, NULL);
}

static PyObject* dsp_goertzel_rad8_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad8_avx, NULL);
}

static PyObject* dsp_goertzel_rad12_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad12_avx, NULL);
}

static PyObject* dsp_goertzel_rad16_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad16_avx, NULL);
}

static PyObject* dsp_goertzel_rad20_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad20_avx, NULL);
}

static PyObject* dsp_goertzel_rad24_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, &goertzel_rad24_avx, NULL);
}

static PyObject* dsp_goertzel_dft(PyObject* self, PyObject* args)
{
    return dsp_goertzel_dft_template(
        self, args, &goertzel_dft, &goertzel_dft_cx);
}

static PyObject* dsp_goertzel_dft_rad2(PyObject* self, PyObject* args)
{
    return dsp_goertzel_dft_template(self, args, &goertzel_dft_rad2, NULL);
}

static PyObject* dsp_goertzel_dft_rad2_sse(PyObject* self, PyObject* args)
{
    return dsp_goertzel_dft_template(self, args, &goertzel_dft_rad2_sse, NULL);
}

/* Set up the methods table */
static PyMethodDef methods[] = {
    {
        "goertzel", dsp_goertzel, // Python name, C name
        METH_VARARGS, // input parameters
        "Goertzel algorithm." // doc string
    },
    {
        "goertzel_cx", dsp_goertzel_rad2,
        METH_VARARGS,
        "Goertzel with complex input algorithm."
    },
    {
        "goertzel_rad2", dsp_goertzel_rad2,
        METH_VARARGS,
        "Goertzel radix-2 algorithm."
    },
    {
        "goertzel_rad2_sse", dsp_goertzel_rad2_sse,
        METH_VARARGS,
        "Goertzel radix-2 algorithm using SSE instructions."
    },
    {
        "goertzel_rad4", dsp_goertzel_rad4,
        METH_VARARGS,
        "Goertzel radix-4 algorithm."
    },
    {
        "goertzel_rad4_avx", dsp_goertzel_rad4_avx,
        METH_VARARGS,
        "Goertzel radix-4 algorithm using AVX instructions."
    },
    {
        "goertzel_rad8_avx", dsp_goertzel_rad8_avx,
        METH_VARARGS,
        "Goertzel radix-8 algorithm using AVX instructions."
    },
    {
        "goertzel_rad12_avx", dsp_goertzel_rad12_avx,
        METH_VARARGS,
        "Goertzel radix-12 algorithm using AVX instructions."
    },
    {
        "goertzel_rad16_avx", dsp_goertzel_rad16_avx,
        METH_VARARGS,
        "Goertzel radix-16 algorithm using AVX instructions."
    },
    {
        "goertzel_rad20_avx", dsp_goertzel_rad20_avx,
        METH_VARARGS,
        "Goertzel radix-20 algorithm using AVX instructions."
    },
    {
        "goertzel_rad24_avx", dsp_goertzel_rad24_avx,
        METH_VARARGS,
        "Goertzel radix-24 algorithm using AVX instructions."
    },
    {
        "goertzel_dft", dsp_goertzel_dft,
        METH_VARARGS,
        "Goertzel algorithm to compute dft."
    },
    {
        "goertzel_dft_rad2", dsp_goertzel_dft_rad2,
        METH_VARARGS,
        "Goertzel radix-2 algorithm to compute dft."
    },
    {
        "goertzel_dft_rad2_sse", dsp_goertzel_dft_rad2_sse,
        METH_VARARGS,
        "Goertzel radix-2 algorithm using SSE instructions to compute dft."
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
    if (!m) return NULL;
    return m;
}
#else
PyMODINIT_FUNC initdsp_ext(void)
{
    (void)Py_InitModule("dsp_ext", methods);
    import_array(); // Must be called for NumPy.
}
#endif
