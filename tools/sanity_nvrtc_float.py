import cupy as cp

code = r'''
extern "C" __global__ void axpy_f(const float* x, const float* y, float* z, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<n) z[i] = x[i] + y[i];
}
'''

# NE PAS passer -arch ici (CuPy le gère)
mod = cp.RawModule(code=code, options=('-std=c++14',))
ker = mod.get_function('axpy_f')

n=1024
x=cp.ones(n, dtype=cp.float32)
y=cp.ones(n, dtype=cp.float32)
z=cp.empty_like(x)

ker(((n+255)//256,), (256,), (x, y, z, n))
cp.cuda.runtime.deviceSynchronize()
assert cp.allclose(z, 2.0)
print("OK: NVRTC (float) opérationnel")
