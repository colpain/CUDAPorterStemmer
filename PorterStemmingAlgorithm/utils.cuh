#ifndef _PORTER_STEMMING_ALGORITHM_UTILS_
#define _PORTER_STEMMING_ALGORITHM_UTILS_
#include "all.cuh"

// remove extra spaces cpu
bool copy_word(char *&dest, char const *&src);
void trim_whitespace(char *dest, char const *src);

/*
    For host functions use
*/
extern "C" struct ParallelStrategy {
    ParallelStrategy(uint32_t b, uint32_t t, uint32_t m, uint32_t total_threads_, uint32_t patition_, uint32_t data_size_) {
        num_blocks = b;
        num_threads_per_block = t;
        shared_memory_per_block = m;
        num_total_threads = total_threads_;
        partition = patition_;
        data_size = data_size_;
    }
    ParallelStrategy() {};
    std::string ToString();
    uint32_t num_blocks;
    uint32_t num_threads_per_block;
    uint32_t shared_memory_per_block;
    uint32_t num_total_threads;
    uint32_t partition;
    uint32_t data_size;
};

cudaDeviceProp GetDeviceProperty(uint32_t device_number);


__forceinline__ __device__ void toLower(char *p) 
{
switch(*p)
{
  case 'A':*p='a'; return;
  case 'B':*p='b'; return;
  case 'C':*p='c'; return;
  case 'D':*p='d'; return;
  case 'E':*p='e'; return;
  case 'F':*p='f'; return;
  case 'G':*p='g'; return;
  case 'H':*p='h'; return;
  case 'I':*p='i'; return;
  case 'J':*p='j'; return;
  case 'K':*p='k'; return;
  case 'L':*p='l'; return;
  case 'M':*p='m'; return;
  case 'N':*p='n'; return;
  case 'O':*p='o'; return;
  case 'P':*p='p'; return;
  case 'Q':*p='q'; return;
  case 'R':*p='r'; return;
  case 'S':*p='s'; return;
  case 'T':*p='t'; return;
  case 'U':*p='u'; return;
  case 'V':*p='v'; return;
  case 'W':*p='w'; return;
  case 'X':*p='x'; return;
  case 'Y':*p='y'; return;
  case 'Z':*p='z'; return;
};
return ;
}
__forceinline__ __device__ void toUpper(char *p) 
{
switch(*p)
{
  case 'a':*p='A'; return;
  case 'b':*p='B'; return;
  case 'c':*p='C'; return;
  case 'd':*p='D'; return;
  case 'e':*p='E'; return;
  case 'f':*p='F'; return;
  case 'g':*p='G'; return;
  case 'h':*p='H'; return;
  case 'i':*p='I'; return;
  case 'j':*p='J'; return;
  case 'k':*p='K'; return;
  case 'l':*p='L'; return;
  case 'm':*p='M'; return;
  case 'n':*p='N'; return;
  case 'o':*p='O'; return;
  case 'p':*p='P'; return;
  case 'q':*p='Q'; return;
  case 'r':*p='R'; return;
  case 's':*p='S'; return;
  case 't':*p='T'; return;
  case 'u':*p='U'; return;
  case 'v':*p='V'; return;
  case 'w':*p='W'; return;
  case 'x':*p='X'; return;
  case 'y':*p='Y'; return;
  case 'z':*p='Z'; return;
};
return ;
}
__forceinline__ __device__ bool isDelimiters(const char& p) {
{
    switch(p)
    {    
      case ' ': return true;
      case '.': return true;
      case ',': return true;
      case '\t': return true;
      case '\r': return true;
      case '\n': return true;
      case ':': return true;
      case ';': return true;
      case '"': return true;
      case '/': return true;
      case '\\': return true;
      case 0x27: return true;
      case '>': return true;
      case '<': return true;
      case '=': return true;
      case '-': return true;
      case '+': return true;
      case '?': return true;
      case '!': return true;
      case '(': return true;
      case ')': return true;
      case '[': return true;
      case ']': return true;
      case '{': return true;
      case '}': return true;
    };
    return false;
}
}

inline bool isDelimitersCPU(const char& p) {
{
    switch(p)
    {    
      case ' ': return true;
      case '.': return true;
      case ',': return true;
      case '\t': return true;
      case '\r': return true;
      case '\n': return true;
      case ':': return true;
      case ';': return true;
      case '"': return true;
      case '/': return true;
      case '\\': return true;
      case 0x27: return true;
      case '>': return true;
      case '<': return true;
      case '=': return true;
      case '-': return true;
      case '+': return true;
      case '?': return true;
      case '!': return true;
      case '(': return true;
      case ')': return true;
      case '[': return true;
      case ']': return true;
      case '{': return true;
      case '}': return true;
    };
    return false;
}
}
/**
    This function calculate how many threads and blocks are needed
    in order to achieve max performance.
    The return strategy meets:
    1. Shared memory constraint
    shared memory size i
    2. Speed up constraint

    @note this is for char data type only!
    TODO This doesn't solve bank conflict problem, fix this in future
*/
ParallelStrategy OptimalLinearDataStrategy(cudaDeviceProp device, uint32_t linear_data_size, uint32_t factor_of_speedup);

/*
    @note: OptimalLinearDataStrategy is a solver to solve 
           optimal strategy linear programming problem
    max. f_0(x)
    s.t f_i(x) for all i in N
    C_i in N
    f(x)_0: num_threads * C_0 - global_mem_used * C_1 + 
            shared_mem_used * C_2 
    f(x)_1: num_threads <= blockDim.x * maxGridSize
    f(x)_2: num_threads >= speedup factors
    f(x)_3: global_mem_used <= GPU global memory
    f(x)_4: shared_mem_used <= shared memory per block
    f(x)_5: global_mem_used + shared_mem_used >= data size
*/

#endif // _PORTER_STEMMING_ALGORITHM_UTILS_