
/**
 * Zipf (Zeta) random distribution.
 *
 * Implementation taken from drobilla's May 24, 2017 answer to
 * https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
 *
 * That code is referenced with this:
 * "Rejection-inversion to generate variates from monotone discrete
 * distributions", Wolfgang Hörmann and Gerhard Derflinger
 * ACM TOMACS 6.3 (1996): 169-184
 *
 * Note that the Hörmann & Derflinger paper, and the stackoverflow
 * code base incorrectly names the paramater as `q`, when they mean `s`.
 * Thier `q` has nothing to do with the q-series. The names in the code
 * below conform to conventions.
 *
 * Example usage:
 *
 *    std::random_device rd;
 *    std::mt19937 gen(rd());
 *    zipf_distribution<> zipf(300);
 *
 *    for (int i = 0; i < 100; i++)
 *        printf("draw %d %d\n", i, zipf(gen));
 */


#include <iostream> 
#include <zip.h>


int main(){

      std::random_device rd;
      std::mt19937 gen(rd());
      zipf_distribution<uint64_t> zipf(100, 0.5); //number of unique keys
  
      for (int i = 0; i < 10000; i++) //number of values to draw from the unique keys.
          printf("i: %d val: %llu\n", i, zipf(gen));
}
