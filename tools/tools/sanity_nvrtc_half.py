import cupy as cp, os

code = r'''
#include <cuda_fp16.h>
extern "C" __global__ void axpy_half(const __half* x, const __half* y, __half* z, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<n){
        float xi = __half2float(x[i]);
        float yi = __half2float(y[i]);
        z[i] = __float2half(xi + yi);
    }
}
'''

cuda_inc = os.path.join(os.environ.get("CUDA_PATH", "/usr/local/cuda"), "include")

# NE PAS passer -arch ; on ajoute juste l'include path
mod = cp.RawModule(code=code, options=('-std=c++14', f'-I{cuda_inc}'))
ker = mod.get_function('axpy_half')

n=1024
x=cp.ones(n, dtype=cp.float16)
y=cp.ones(n, dtype=cp.float16)
z=cp.empty_like(x)
ker(((n+255)//256,), (256,), (x, y, z, n))
cp.cuda.runtime.deviceSynchronize()
assert cp.allclose(z, 2, atol=1e-3)
print("OK: NVRTC + cuda_fp16.h opérationnels")
