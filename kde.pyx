# cython:language_level=3
# cython:cdivision_warnings=False
# distutils: language=c
# cython: cdivision=True
# kde.pyx

# kde.pyx
# python3.8 setup.py build_ext --inplace
import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, M_PI, pow, cos

cdef double abs(double x):
    if x <= 0:
        return -x
    else:
        return x

# 分位数
@cython.boundscheck(False)
@cython.wraparound(False)
cdef calculate_quantile(np.ndarray[np.double_t,ndim=1] arr, double q):
    cdef int n = arr.shape[0]
    cdef double[::1] sorted_arr
    sorted_arr = np.sort(arr)
    # arr.sort()
    cdef int k = int(n * q)
    cdef double result
    # Assuming the array is already sorted
    result = sorted_arr[k]
    return result

# 定义高斯核函数
cdef double gaussian_kernel_pdf(double x):
    cdef double res = exp(-0.5 * pow(x,2)) / sqrt(2.0 * M_PI)
    return res
# 均匀核
cdef double uniform_kernel_pdf(double x):
    cdef double res = 1.0/2.0*(abs(x)<=1.0)
    return res
# epanchnikov核
cdef inline double epanchnikov_kernel_pdf(double x):
    cdef double res
    if abs(x) <= 1:
         res = 3.0/4.0 * (1.0-pow(x,2))
    else:
        res = 0
    return res
# Quartic kernel
cdef double quartic_kernel_pdf(double x):
    cdef double res = 15.0/16.0 * pow((1-pow(x,2)),2) * (abs(x)<=1.0)
    return res
# Triweight kernel
cdef double triweight_kernel_pdf(double x):
    cdef double res=  35.0/32.0 * pow((1-pow(x,2)),3 )* (abs(x)<=1)
    return res
# Tricube kernel
cdef double tricube_kernel_pdf(double x):
    cdef double res = 70.0/81.0 * pow( (1-pow(abs(x),3)),3) * (abs(x)<=1)
    return res
# Cosine kernel
cdef double cosine_kernel_pdf(double x):
    return M_PI/4 * cos(M_PI/2 * x) * (abs(x)<=1)
# Sigmoid kernel
cdef double sigmoid_kernel_pdf(double x):
    return 2/M_PI * 1/(exp(x)+exp(-x))
# 定义三角核函数
cdef double triangle_kernel_pdf(double x):
    cdef double res = (1.0 - abs(x)) * (abs(x) <= 1)
    return res

# 定义核密度估计函数
# ctypedef double (*FuncPtr)(double)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=1] kde(np.ndarray[np.double_t, ndim=1] input_data, np.ndarray[np.double_t, ndim=1] compute_data, bytes bandwidth_method=None, bytes kernel_name=None):
    cdef Py_ssize_t n_input = input_data.shape[0]
    cdef Py_ssize_t n_compute = compute_data.shape[0]
    cdef double bandwidth
    cdef double kernel_value
    cdef int i, j
    cdef np.ndarray[np.double_t,ndim=1] result
    result = np.empty(n_compute,dtype=np.double)
    # 计算带宽
    # %TODO 带宽方法选择
    if bandwidth_method == b'silverman':
        bandwidth = (4.0 / (3.0 * n_input))**0.2 * np.std(input_data)
    # elif bandwidth_method == b'other_method':
        # 使用其他带宽计算方法
    #    bandwidth = myband(input_data)
    else:
        bandwidth = (4.0 / (3.0 * n_input)) ** 0.2 * np.std(input_data)
        # raise ValueError(f"Unsupported bandwidth method: {bandwidth_method.decode('utf-8')}")
    cdef double (*kernel_func)(double)
    if kernel_name == b'gaussidan':
        kernel_func = <double (*)(double)>gaussian_kernel_pdf
    elif kernel_name == b'epanchnikov':
        kernel_func = <double (*)(double)>epanchnikov_kernel_pdf
    elif kernel_name == b'triangle':
        kernel_func = <double (*)(double)>triangle_kernel_pdf
    elif kernel_name == b'uniform':
        kernel_func = <double (*)(double)>uniform_kernel_pdf
    elif kernel_name==b'quartic':
        kernel_func = <double (*)(double)>quartic_kernel_pdf
    else:
        kernel_func = <double (*)(double)>epanchnikov_kernel_pdf
    # 循环计算每个数据点的核密度估计值
    for i in range(n_compute):
        result[i] = 0.0
        for j in range(n_input):
            kernel_value = kernel_func((compute_data[i] - input_data[j]) / bandwidth)
            # if kernel_name == b'gaussian':
            #     kernel_value = kernel_func((compute_data[i] - input_data[j]) / bandwidth)
            # elif kernel_name == b'triangle':
            #     kernel_value = triangle_kernel((compute_data[i] - input_data[j]) / bandwidth)
            # else:
            #     raise ValueError(f"Unsupported kernel: {kernel_name.decode('utf-8')}")
            result[i] += kernel_value

        result[i] /= (n_input * bandwidth)
    return result
# 核密度函数积分

# TODO 积分函数
# gaussian kernel integrate
cdef double gaussian_kernel_cdf(double x):
    cdef double res = exp(-0.5 * pow(x,2)) / sqrt(2.0 * M_PI)
    return res
# 均匀核 DONE
cdef double uniform_kernel_cdf(double x):
    cdef double res # = 1.0/2.0 * (x+1) *(abs(x)<=1.0)
    if x <= -1:
        res = 0
    elif x <=1:
        res = 1.0/2.0*(x+1)
    else:
        res = 1
    return res
# epanchnikov核 %DONE
cpdef inline double epanchnikov_kernel_cdf(double x):
    cdef double res
    if x <= -1:
        res = 0.0
    elif x<=1:
        res = 1.0/2.0+3.0/4.0 * x - pow(x,3)/4.0
    else:
        res = 1.0
    return res
# Quartic kernel TODO
cdef double quartic_kernel_cdf(double x):
    cdef double res
    if x <= -1:
        res = 0
    elif x<=1:
        res = 0.5 + 0.9375 * x - 0.625 *pow(x,3) + 0.1875 *pow(x,5)
    else:
        res = 1
    return res
# Triweight kernel TODO
cdef double triweight_kernel_cdf(double x):
    cdef double res=  35.0/32.0 * pow((1-pow(x,2)),3 )* (abs(x)<=1)
    return res
# Tricube kernel TODO
cdef double tricube_kernel_cdf(double x):
    cdef double res = 70.0/81.0 * pow( (1-pow(abs(x),3)),3) * (abs(x)<=1)
    return res
# Cosine kernel TODO
cdef double cosine_kernel_cdf(double x):
    return M_PI/4.0 * cos(M_PI/2.0 * x) * (abs(x)<=1)
# Sigmoid kernel TODO
cdef double sigmoid_kernel_cdf(double x):
    return 2.0/M_PI * 1.0/(exp(x)+exp(-x))

# 定义三角核函数 DONE
cdef inline double triangle_kernel_cdf(double x):
    cdef double res
    if x <= -1:
        res = 0
    elif x <= 0:
        res = 0.5 * pow((1 + x),2)
    elif x <= 1:
        res = 0.5 + x - pow(x,2)/2.0
    else:
        res = 1
    return res
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=1] kde_cdf(np.ndarray[np.double_t, ndim=1] input_data, np.ndarray[np.double_t, ndim=1] compute_data, bytes bandwidth_method=None, bytes kernel_name=None):
    cdef int n_input = input_data.shape[0]
    cdef int n_compute = compute_data.shape[0]
    cdef double bandwidth
    cdef double kernel_value
    cdef int i, j
    cdef np.ndarray[np.double_t,ndim=1] result
    cdef np.ndarray[np.double_t,ndim=2] temp
    temp = np.empty((n_input,n_compute),dtype=np.double)
    # cdef FuncPtr kernel_func
    result = np.empty(n_compute,dtype=np.double)
    # 计算带宽
    # %TODO 带宽方法选择
    if bandwidth_method == b'silverman':
        bandwidth = (4.0 / (3.0 * n_input))**0.2 * np.std(input_data)
    # elif bandwidth_method == b'other_method':
        # 使用其他带宽计算方法
    #    bandwidth = myband(input_data)
    else:
        bandwidth = (4.0 / (3.0 * n_input)) ** 0.2 * np.std(input_data)
        # raise ValueError(f"Unsupported bandwidth method: {bandwidth_method.decode('utf-8')}")
    cdef double (*kernel_func)(double)
    if kernel_name == b'gaussidan':
        kernel_func = <double (*)(double)>gaussian_kernel_cdf
    elif kernel_name == b'epanchnikov':
        kernel_func = <double (*)(double)>epanchnikov_kernel_cdf
    elif kernel_name == b'triangle':
        kernel_func = <double (*)(double)>triangle_kernel_cdf
    elif kernel_name == b'uniform':
        kernel_func = <double (*)(double)>uniform_kernel_cdf
    elif kernel_name==b'quartic':
        kernel_func = <double (*)(double)>quartic_kernel_cdf
    else:
        kernel_func = <double (*)(double)>epanchnikov_kernel_cdf
    # 循环计算每个数据点的核密度估计值
    for i in range(n_compute):
        result[i] = 0.0
        for j in range(n_input):
            kernel_value = kernel_func((compute_data[i] - input_data[j]) / bandwidth)
            # if kernel_name == b'gaussian':
            #     kernel_value = kernel_func((compute_data[i] - input_data[j]) / bandwidth)
            # elif kernel_name == b'triangle':
            #     kernel_value = triangle_kernel((compute_data[i] - input_data[j]) / bandwidth)
            # else:
            #     raise ValueError(f"Unsupported kernel: {kernel_name.decode('utf-8')}")
            result[i] += kernel_value/n_input

        # result[i] /= (n_input)
    return result
