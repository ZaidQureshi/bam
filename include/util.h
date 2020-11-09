#ifndef __UTIL_H__
#define __UTIL_H__

#include "cuda.h"
#include "nvm_util.h"
//#include <ctype>
#include <cstdio>

#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(1);
    }
}

#define CEIL(X, Y, Z) ((X + Y - 1) >> Z)


#ifndef HEXDUMP_COLS
#define HEXDUMP_COLS 16
#endif
void hexdump(void *mem, unsigned int len)
{
        unsigned int i, j;

        for(i = 0; i < len + ((len % HEXDUMP_COLS) ? (HEXDUMP_COLS - len % HEXDUMP_COLS) : 0); i++)
        {
                /* print offset */
                if(i % HEXDUMP_COLS == 0)
                {
                        printf("0x%06x: ", i);
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
                if(i % HEXDUMP_COLS == (HEXDUMP_COLS - 1))
                {
                        for(j = i - (HEXDUMP_COLS - 1); j <= i; j++)
                        {
                                if(j >= len) /* end of block, not really printing */
                                {
                                        putchar(' ');
                                }
                                else if(isprint(((char*)mem)[j])) /* printable char */
                                {
                                        putchar(0xFF & ((char*)mem)[j]);
                                }
                                else /* other char */
                                {
                                        putchar('.');
                                }
                        }
                        putchar('\n');
                }
        }
}


/*warp memcpy, assumes alignment at type T and num is a count in type T*/
template <typename T>
__device__
void warp_memcpy(T* dest, const T* src, size_t num) {
        uint32_t mask = __activemask();
        uint32_t active_cnt = __popc(mask);
        uint32_t lane = lane_id();
        uint32_t prior_mask = mask >> (32 - lane);
        uint32_t prior_count = __popc(prior_mask);

        for(size_t i = prior_count; i < num; i+=active_cnt)
                dest[i] = src[i];
}

#endif // __UTIL_H__
