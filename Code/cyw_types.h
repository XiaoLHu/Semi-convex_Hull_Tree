#ifndef CYW_TYPES_H_INCLUDED
#define CYW_TYPES_H_INCLUDED


#define USE_DOUBLE 0
#if USE_DOUBLE > 0
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FLOAT_TYPE double
#define FLOAT_TYPE4 double4
#define MAX_FLOAT_TYPE      1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE      3.402823466e+38f
#define MIN_FLOAT_TYPE     -3.402823466e+38f
#endif

#define INFINITE            3.402823466e+38f
#define WORKGROUP_SIZE_COPY_INIT 32

#endif // CYW_TYPES_H_INCLUDED
