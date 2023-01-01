#include "include.h"
#include <math.h>

static PyObject* dsp_goertzel_template(
    PyObject* self, PyObject* args, int fun_index)
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

    if (fun_index == 0)
    {
        if (PyArray_ISCOMPLEX(in_data))
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                goertzel_cx(data, data_len, k, out_res);
                data += data_len*2;
                out_res += 2;
            }
        else
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                goertzel(data, data_len, k, out_res);
                data += data_len;
                out_res += 2;
            }
    }
    else if (fun_index == 1)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            goertzel_rad2(data, data_len, k, out_res);
            data += data_len;
            out_res += 2;
        }
    else if (fun_index == 2)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            goertzel_rad2_sse(data, data_len, k, out_res);
            data += data_len;
            out_res += 2;
        }
    else if (fun_index == 3)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            goertzel_rad4(data, data_len, k, out_res);
            data += data_len;
            out_res += 2;
        }
    else if (fun_index == 4)
        for (npy_intp i_data = 0; i_data < n_data; i_data++)
        {
            goertzel_rad4_avx(data, data_len, k, out_res);
            data += data_len;
            out_res += 2;
        }

    // Decrease the reference count of ap.
    Py_DECREF(in_data);
    return output;
}

static PyObject* dsp_goertzel(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, 0);
}

static PyObject* dsp_goertzel_rad2(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, 1);
}

static PyObject* dsp_goertzel_rad2_sse(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, 2);
}

static PyObject* dsp_goertzel_rad4(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, 3);
}

static PyObject* dsp_goertzel_rad4_avx(PyObject* self, PyObject* args)
{
    return dsp_goertzel_template(self, args, 4);
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
