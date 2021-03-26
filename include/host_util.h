#ifndef __HOST_UTIL_H_
#define __HOST_UTIL_H_

#ifndef __device__ 
#define __device__
#endif 
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include <ctime>

#ifndef __CUDACC__

template<typename T>
inline __host__
void __nanosleep(T ns) {
        struct timespec time1,time2;
        time1.tv_sec  = 0;
        time2.tv_nsec = ns;
        nanosleep(&time1, &time2);
}

template<typename T>
inline __host__
T __activemask() {
    T var;
    (void) var;
    return (T)1;
}

template<typename T>
inline __host__
int __popc(T v) {
    if (sizeof(T) == 4)
        return __builtin_popcount((unsigned)v);
    if (sizeof(T) == 8)
        return __builtin_popcountll((unsigned long long)v);
    return 0;

}

template<typename T>
inline __host__
int __ffs(T v) {
    if (sizeof(T) == 4)
        return __builtin_ffs((int)v);
    if (sizeof(T) == 8)
        return __builtin_ffsll((long long)v);
    return 0;

}

template<typename T>
inline __host__
void __syncwarp(T mask) {
    (void) mask;
    return;
}

template<typename T>
inline __host__
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32) {
    (void) mask;
    (void) srcLane;
    (void) width;
    return var;
}


template<typename T>
inline __host__
unsigned int __match_any_sync(unsigned mask, T var) {
    (void) mask;
    (void) var;
    return 1;
}

#endif 

//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __HOST_UTIL_H_
