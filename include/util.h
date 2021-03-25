#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef __device__ 
#define __device__
#endif 
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif



#include "cuda.h"
#include "nvm_util.h"
#include "host_util.h"
//#include <ctype>
#include <cstdio>


#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef __CUDACC__
inline void gpuAssert(int code, const char *file, int line, bool abort=false)
{
    if (code != 0)
    {
	fprintf(stderr,"Assert: %i %s %d\n", code, file, line);
	if (abort) exit(1);
    }
}
#else

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(1);
    }
}
#endif

#define CEIL(X, Y, Z) ((X + Y - 1) >> Z)


#ifndef HEXDUMP_COLS
#define HEXDUMP_COLS 16
#endif
inline __device__ void hexdump(void *mem, unsigned int len)
{
        unsigned int i;

        for(i = 0; i < len + ((len % HEXDUMP_COLS) ? (HEXDUMP_COLS - len % HEXDUMP_COLS) : 0); i++)
        {
                /* print offset */
                if(i % HEXDUMP_COLS == 0)
                {
                        printf("\n0x%06x: ", i);
                }

                /* print hex data */
                if(i < len)
                {
                        printf("%02x ", 0xFF & ((char*)mem)[i]);
                }
                else /* end of block, just aligning for ASCII dump */
                {
                        printf("   ");
                }

                /* print ASCII dump */
//                if(i % HEXDUMP_COLS == (HEXDUMP_COLS - 1))
//                {
//                        for(j = i - (HEXDUMP_COLS - 1); j <= i; j++)
//                        {
//                                if(j >= len) /* end of block, not really printing */
//                                {
//                                        printf(' ');
//                                }
//                                else if(isprint(((char*)mem)[j])) /* printable char */
//                                {
//                                        printf(0xFF & ((char*)mem)[j]);
//                                }
//                                else /* other char */
//                                {
//                                        putchar('.');
//                                }
//                        }
//                        putchar('\n');
//                }
        }
        printf("\n");
}

template <typename T>
void __ignore(T &&)
{ }
/*warp memcpy, assumes alignment at type T and num is a count in type T*/
template <typename T>
inline __device__
void warp_memcpy(T* dest, const T* src, size_t num) {
#ifndef __CUDACC__
    uint32_t mask = 1;
#else
    uint32_t mask = __activemask();
#endif
        uint32_t active_cnt = __popc(mask);
        uint32_t lane = lane_id();
        uint32_t prior_mask = mask >> (32 - lane);
        uint32_t prior_count = __popc(prior_mask);

        for(size_t i = prior_count; i < num; i+=active_cnt)
                dest[i] = src[i];
}

//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __UTIL_H__
