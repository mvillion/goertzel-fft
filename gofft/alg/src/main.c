#include "include.h"
#include <math.h>

static PyObject* dsp_goertzel_template(
    PyObject* self, PyObject* args, goertzel_fun_t *fun, goertzel_fun_t *fun_cx,
    goertzelf_fun_t *funf, goertzelf_fun_t *funf_cx)
{
    PyArrayObject *in_data;
    PyObject *output;
    double k;

    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &in_data, &k))
        return NULL;
    if (in_data == NULL) return NULL;

    // Ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    in_data = PyArray_GETCONTIGUOUS(in_data);

    // create output dimensions
    // last axis is removed, replaced by complex data i&q
    npy_intp out_dim[NPY_MAXDIMS];
    int n_dim = PyArray_NDIM(in_data);
    memcpy(out_dim, PyArray_DIMS(in_data), n_dim*sizeof(npy_intp));
    long int data_len = out_dim[n_dim-1];
    npy_intp n_data = 1;
    for (int k = 0; k < n_dim-1; k++)
        n_data *= out_dim[k];

    int typenum = PyArray_TYPE(in_data);
    if (typenum == NPY_FLOAT64 || typenum == NPY_COMPLEX128)
    {
        double *data = (double *)PyArray_DATA(in_data);
        output = PyArray_SimpleNew(n_dim-1, out_dim, NPY_COMPLEX128);
        double *out_res = (double *)PyArray_DATA((PyArrayObject *)output);
        if (typenum == NPY_FLOAT64 && fun != NULL)
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                fun(data, data_len, k, out_res);
                data += data_len;
                out_res += 2;
            }
        else if (typenum == NPY_COMPLEX128 && fun_cx != NULL)
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                fun_cx(data, data_len, k, out_res);
                data += data_len*2;
                out_res += 2;
            }
    }
    else //if (typenum == NPY_FLOAT32 || typenum == NPY_COMPLEX64)
    {
        float *data = (float *)PyArray_DATA(in_data);
        output = PyArray_SimpleNew(n_dim-1, out_dim, NPY_COMPLEX64);
        float *out_res = (float *)PyArray_DATA((PyArrayObject *)output);
        if (typenum == NPY_FLOAT32 && fun != NULL)
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                funf(data, data_len, k, out_res);
                data += data_len;
                out_res += 2;
            }
        else if (typenum == NPY_COMPLEX64 && fun_cx != NULL)
            for (npy_intp i_data = 0; i_data < n_data; i_data++)
            {
                funf_cx(data, data_len, k, out_res);
                data += data_len*2;
                out_res += 2;
            }
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

#define DEF_DSP(name, fun_cx, funf, funf_cx) \
static PyObject* dsp_ ## name (PyObject* self, PyObject* args) \
{ \
    return dsp_goertzel_template(self, args, &name, fun_cx, funf, funf_cx); \
}
DEF_DSP(goertzel, &goertzel_cx, &goertzelf, &goertzelf_cx)
DEF_DSP(goertzel_rad2, NULL, &goertzelf_rad2, NULL)
DEF_DSP(goertzel_rad2_sse, NULL, NULL, NULL)
DEF_DSP(goertzel_rad4, NULL, &goertzelf_rad4, NULL)
DEF_DSP(goertzel_rad4_avx, &goertzel_rad4_cx_avx, NULL, NULL)
DEF_DSP(goertzel_rad4u2_avx, NULL, NULL, NULL)
DEF_DSP(goertzel_rad4u4_avx, NULL, NULL, NULL)
DEF_DSP(goertzel_rad8_avx, &goertzel_rad8_cx_avx, &goertzelf_rad8_avx, NULL)
DEF_DSP(goertzel_rad12_avx, &goertzel_rad12_cx_avx, NULL, NULL)
DEF_DSP(goertzel_rad16_avx, NULL, &goertzelf_rad16_avx, NULL)
DEF_DSP(goertzel_rad20_avx, NULL, NULL, NULL)
DEF_DSP(goertzel_rad24_avx, NULL, &goertzelf_rad24_avx, NULL)
DEF_DSP(goertzel_rad40_avx, NULL, &goertzelf_rad40_avx, NULL)
DEF_DSP(goertzel_rad4x2_test, NULL, NULL, NULL)
DEF_DSP(goertzel_rad4_fma, NULL, NULL, NULL)
DEF_DSP(goertzel_rad8_fma, NULL, &goertzelf_rad8_fma, NULL)
DEF_DSP(goertzel_rad20_fma, NULL, NULL, NULL)
DEF_DSP(goertzel_rad24_fma, NULL, &goertzelf_rad24_fma, NULL)
#undef DEF_DSP

#define DEF_DSP_DFT(name, fun_cx, funf, funf_cx) \
static PyObject* dsp_ ## name (PyObject* self, PyObject* args) \
{ \
    return dsp_goertzel_dft_template(self, args, &name, fun_cx); \
}
DEF_DSP_DFT(goertzel_dft, &goertzel_cx_dft, NULL, NULL)
DEF_DSP_DFT(goertzel_rad2_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad2_sse_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad4_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad8_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad12_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad16_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad20_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad24_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad40_avx_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad4_fma_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad8_fma_dft, NULL, NULL, NULL)
DEF_DSP_DFT(goertzel_rad20_fma_dft, NULL, NULL, NULL)
#undef DEF_DSP_DFT

#define stringify(x) #x
#define DEF_DSP(radix, arch) \
    { \
        stringify(goertzel_rad ## radix ## _ ## arch), \
        dsp_goertzel_rad ## radix ## _ ## arch, METH_VARARGS, \
        "Goertzel radix-" stringify(radix) \
        " algorithm using " stringify(arch) " instructions." \
    },
#define DEF_DSP_DFT(radix, arch) \
    { \
        stringify(goertzel_rad ## radix ## _ ## arch ## _dft), \
        dsp_goertzel_rad ## radix ## _ ## arch ## _dft, METH_VARARGS, \
        "Goertzel radix-" stringify(radix) \
        " algorithm using " stringify(arch) " instructions to compute dft." \
    },
/* Set up the methods table */
static PyMethodDef methods[] = {
    {
        "goertzel", dsp_goertzel, // Python name, C name
        METH_VARARGS, // input parameters
        "Goertzel algorithm." // doc string
    },
    {
        "goertzel_rad2", dsp_goertzel_rad2, METH_VARARGS,
        "Goertzel radix-2 algorithm."
    },
    {
        "goertzel_rad4", dsp_goertzel_rad4, METH_VARARGS,
        "Goertzel radix-4 algorithm."
    },
DEF_DSP(2, sse)
DEF_DSP(4, avx)
DEF_DSP(8, avx)
DEF_DSP(12, avx)
DEF_DSP(16, avx)
DEF_DSP(20, avx)
DEF_DSP(24, avx)
DEF_DSP(40, avx)
    {
        "goertzel_rad4u2_avx", dsp_goertzel_rad4u2_avx, METH_VARARGS,
        "Goertzel radix-4 algorithm using AVX instructions (unrolled 2 times)."
    },
    {
        "goertzel_rad4u4_avx", dsp_goertzel_rad4u4_avx, METH_VARARGS,
        "Goertzel radix-4 algorithm using AVX instructions (unrolled 4 times)."
    },
    {
        "goertzel_rad4x2_test", dsp_goertzel_rad4x2_test, METH_VARARGS,
        "Goertzel radix-4 algorithm using AVX instructions on 2 frequencies."
    },
DEF_DSP(4, fma)
DEF_DSP(8, fma)
DEF_DSP(20, fma)
DEF_DSP(24, fma)
    {
        "goertzel_dft", dsp_goertzel_dft, METH_VARARGS,
        "Goertzel algorithm to compute dft."
    },
    {
        "goertzel_rad2_dft", dsp_goertzel_rad2_dft, METH_VARARGS,
        "Goertzel radix-2 algorithm to compute dft."
    },
DEF_DSP_DFT(2, sse)
DEF_DSP_DFT(4, avx)
DEF_DSP_DFT(8, avx)
DEF_DSP_DFT(12, avx)
DEF_DSP_DFT(16, avx)
DEF_DSP_DFT(20, avx)
DEF_DSP_DFT(24, avx)
DEF_DSP_DFT(40, avx)
DEF_DSP_DFT(4, fma)
DEF_DSP_DFT(8, fma)
DEF_DSP_DFT(20, fma)
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
