# CUDAPorterStemmer
A CUDA GPU implementation of Porter Stemmer algorithm to raw text preprocessing

This project is a combination of C++ and CUDA binaries, some workloads are operated and handled on CPU, and some workloads are handled and processed on GPU.
The basic idea is CPU will read the raw file, and then divide the raw file into multiple small chunks, then pass small chunks to GPU for parallel processing and wait for results from GPU threads, once CPU receives all GPU results, CPU will aggregate all results togehter (put them in right order, append broken strings etc) then return the final results. 


# Components
- CUDAPorterStemmer/PorterStemmingAlgorithm/all.cuh specifies dependencies
- CUDAPorterStemmer/PorterStemmingAlgorithm/data_handler.cuh specifies the CPU data handler utilities
- CUDAPorterStemmer/PorterStemmingAlgorithm/utils.cuh specifies mappings and the GPU prallelization stratigies (basially a linear programming optimization to decide the work load size)
- CUDAPorterStemmer/PorterStemmingAlgorithm/stemmer_function.cuh speicifies the CUDA processing components (all inlined in headewr file because of some bugs, i dont remember the details) 
- CUDAPorterStemmer/PorterStemmingAlgorithm/kernel.cu speicifies the main file and also some CPU utils to cleaning up after recive the results from GPU threads
- CUDAPorterStemmer/PorterStemmingAlgorithm/porter_stemming_cpu.cuh specifies the CPU version of porter stammer for benchmarking purpose
