#ifndef __BAFS_PTR_H__
#define __BAFS_PTR_H__

#ifndef __device__ 
#define __device__
#endif 
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__  
#define __forceinline__ inline
#endif

#include "page_cache.h"
#include <cstdint>

template<typename T>
class bafs_ptr {
private:
    array_t<T>* h_pData;
    array_d_t<T>* pData;
    uint64_t start_idx;
public:
    __host__
    void print_stats() const {
        if (h_pData)
            h_pData->print_reset_stats();
    }
    __host__ __device__ bafs_ptr():
        h_pData(NULL), pData(NULL),start_idx(0){
    }
    // __host__ __device__ bafs_ptr(array_d_t<T>* const pValue):
    //     h_pData(NULL), pData(pValue),start_idx(0){
    // }

    __host__ __device__ bafs_ptr(array_d_t<T>* const pValue, const uint64_t start_off):
        h_pData(NULL), pData(pValue),start_idx(start_off){
    }

    __host__ __device__ bafs_ptr(array_t<T>* const pValue):
        h_pData(pValue), pData(pValue->d_array_ptr),start_idx(0){

    }

    __host__ __device__ bafs_ptr(array_t<T>* const pValue, const uint64_t start_off):
        h_pData(pValue), pData(pValue->d_array_ptr),start_idx(start_off){
    }

    __host__ __device__ ~bafs_ptr(){}

    __host__ __device__ bafs_ptr(const bafs_ptr &var){
        h_pData = var.h_pData;
        pData = var.pData;
        start_idx = var.start_idx;
    }

    __device__ T operator*(){
        return (*pData)[start_idx];
    }

    __host__ __device__ bafs_ptr<T>& operator=(const bafs_ptr<T>& obj) {
        if(*this == obj)
            return *this;
        else{
            this->h_pData = obj.h_pData;
            this->pData = obj.pData;
            this->start_idx = obj.start_idx;
        }
        return *this;
    }

    template<typename T_>
    friend __host__ __device__ bool operator==(const bafs_ptr<T_>& lhs, const bafs_ptr<T_>& rhs);

    // template<typename T_>
    // friend __host__ __device__ bool operator==(bafs_ptr<T>* lhs, const bafs_ptr<T_>& rhs);

    __host__ __device__ void operator()(const uint64_t i, const T val) {
        (*pData)(i, val);
    }
    __host__ __device__ T operator[](const uint64_t i) {
        return (*pData)[start_idx+i];
    }

    __host__ __device__ const T operator[](const uint64_t i) const {
        return (*pData)[start_idx+i];
    }

    __host__ __device__ bafs_ptr<T> operator+(const uint64_t i){
        uint64_t new_start_idx = this->start_idx+i;
        return bafs_ptr<T>(this->pData, new_start_idx);
    }
    __host__ __device__ bafs_ptr<T> operator-(const uint64_t i){
        uint64_t new_start_idx = this->start_idx-i;
        return bafs_ptr<T>(this->pData, new_start_idx);
    }
//posfix operator
    __host__ __device__ bafs_ptr<T> operator++(int){
        bafs_ptr<T> cpy = *this;
        this->start_idx += 1;
        return cpy;
    }
//prefix operator
    __host__ __device__ bafs_ptr<T>& operator++(){
        this->start_idx += 1;
        return *this;
    }

//posfix operator
    __host__ __device__ bafs_ptr<T> operator--(int){
        bafs_ptr<T> cpy = *this;
        this->start_idx -= 1;
        return cpy;
    }
//prefix operator
    __host__ __device__ bafs_ptr<T>& operator--(){
        this->start_idx -= 1;
        return *this;
    }

    __host__ __device__ void memcpy_to_array_aligned(const uint64_t src_idx, const uint64_t count, T* dest) const {
        pData->memcpy(src_idx, count, dest);
    }
};



template<typename T_>
__host__ __device__
bool operator==(const bafs_ptr<T_>& lhs, const bafs_ptr<T_>& rhs){
   return (lhs.pData == rhs.pData && lhs.start_idx == rhs.start_idx && lhs.h_pData == rhs.h_pData);
}

// template<typename T_>
// __host__ __device__
// bool operator==(bafs_ptr<T_>* lhs, const bafs_ptr<T_>& rhs){
//    return (lhs->pData == rhs.pData && lhs->start_idx == rhs.start_idx);
// }


//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif //__BAFS_PTR_H__
