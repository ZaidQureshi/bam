#ifndef __SPTR_H__
#define __SPTR_H__

#include <iostream> 
#include <cstdint> 

template<class T> class SPtr {
private:
    T* pData;
  uint64_t start_idx;    
public:
    __host__ __device__ SPtr(T* const pValue): 
        pData(pValue),start_idx(0){
    };
    
    __host__ __device__ SPtr(T* const pValue, const uint64_t start_off): 
        pData(pValue),start_idx(start_off){
    };
    
    __host__ __device__ ~SPtr(){}

    __host__ __device__ SPtr(const SPtr &var){
          pData = var.pData; 
         start_idx = var.start_idx;    
    }
       
      __device__ T& operator*(){
         return pData[start_idx];
    }
      
    __host__ __device__ SPtr<T>& operator=(const SPtr<T>& obj);
       
    __host__ __device__ T& operator[](const uint64_t i){
          return pData[start_idx+i];
    }
       
    __host__ __device__ const T& operator[](const uint64_t i) const{
          return pData[start_idx+i];
    }

    __host__ __device__ SPtr<T> operator+(const uint64_t i){
        uint64_t new_start_idx = this->start_idx+i;
        return SPtr<T>(this->pData, new_start_idx);
    }
    __host__ __device__ SPtr<T> operator-(const uint64_t i){
        uint64_t new_start_idx = this->start_idx-i;
        return SPtr<T>(this->pData, new_start_idx);
    }
//posfix operator    
    __host__ __device__ SPtr<T> operator++(int){
        SPtr<T> cpy = *this; 
        this->start_idx += 1;
        return cpy; 
    }
//prefix operator    
    __host__ __device__ SPtr<T>& operator++(){
        this->start_idx += 1;
        return *this; 
    }

//posfix operator    
    __host__ __device__ SPtr<T> operator--(int){
        SPtr<T> cpy = *this; 
        this->start_idx -= 1;
        return cpy; 
    }
//prefix operator    
    __host__ __device__ SPtr<T>& operator--(){
        this->start_idx -= 1;
        return *this; 
    }

//    T getValue(uint64_t i) {return pData[i];}
};

template<class T>
__host__ __device__ SPtr<T>& SPtr<T>:: operator=(const SPtr<T>& obj){
   if(this == obj)
     return *this; 
   else{
     this->pData = obj.pData; 
        this->start_idx = obj.start_idx; 
   }
   return *this;  
}
     
#endif
