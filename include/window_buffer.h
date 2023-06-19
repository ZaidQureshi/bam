#ifndef __WINDOW_BUFFER_H__
#define __WINDOW_BUFFER_H__

#ifndef __device__ 
#define __device__
#endif 
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__  
#define __forceinline__ inline
#endif

#include <cstdint>
#include <stdio.h>
#include <page_cache.h>

template <typename T = float>
 __forceinline__
__device__
void write_to_queue(void* src, void* dst, size_t size, uint32_t mask){
     T* src_ptr = (T*) src;
     T* dst_ptr = (T*) dst;
     
     uint32_t count = __popc(mask);
     uint32_t lane_id = threadIdx.x %32;
     
     uint32_t my_id = count - (__popc(mask>>(lane_id)));
     for(; my_id < size/sizeof(T); my_id += count){
          dst_ptr[my_id] =  src_ptr[my_id]; 
         if(my_id == 1023 || my_id == 1022){
         }
     }
 }

        






#endif //__BAFS_PTR_H__
