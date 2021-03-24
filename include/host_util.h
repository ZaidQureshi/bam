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

static inline __host__
void __nanosleep(unsigned ns) {
        struct timespec time1,time2;
        time1.tv_sec  = 0;
        time2.tv_nsec = ns;
        nanosleep(&time1, &time2);
}

static inline __host__
unsigned __activemask() {
        return 1;
}

static inline __host__
int __popc(unsigned v) {
   return __builtin_popcount(v);
}

static inline __host__
int __ffs(int x) {
    return __builtin_ffs(x);
}


static inline __host__
void __syncwarp(unsigned mask) {
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



//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __HOST_UTIL_H_
