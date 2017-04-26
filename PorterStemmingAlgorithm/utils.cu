#include "utils.cuh"
#include <fstream>


bool copy_word(char *&dest, char const *&src) { 
    while (isspace(*(unsigned char *)src))
        ++src;

    while (*src && *src != ' ') {
        *dest = *src;
        ++dest;
        ++src;
    }
    return *src != '\0';
}

void trim_whitespace(char *dest, char const *src) { 
    while (copy_word(dest, src))
        *dest++ = ' ';
    *dest = '\0';
}


cudaDeviceProp GetDeviceProperty(uint32_t device_number) {
    cudaDeviceProp* prop = new cudaDeviceProp;
    auto error_code = cudaGetDeviceProperties(prop, device_number);
    if(error_code) {
        std::cout << "Get property error" << ", error code: " << error_code << std::endl;
        exit(9);
    }
    std::cout.flush();
    return *prop;
}

ParallelStrategy OptimalLinearDataStrategy(cudaDeviceProp device, 
                                           uint32_t linear_data_size, 
                                           uint32_t factor_of_speedup) {
    
    uint64_t num_blocks = ceil((double)linear_data_size / (double)device.sharedMemPerBlock);
    uint64_t threads_needed = factor_of_speedup;
    uint64_t partition = 1;
    
    // only consider 1 dimension for now
    // this condition checks if data size exceed gpu memory
    if (num_blocks > device.maxGridSize[0]) {
        /* 
            blocks needed exceeds max num blocks per grid
            In this case, we have to divide original raw data into pieces
            and pass it to device in multiple times
        */
        num_blocks = device.maxGridSize[0];
        // this partition has extra part to include modulers 
        partition = ceil( (double)linear_data_size / (double)(device.maxGridSize[0] * device.sharedMemPerBlock)); 
    }
    else if (num_blocks <= device.maxGridSize[0]) {
        partition = 1;
    }
    
    
    uint32_t threads_per_block = ceil((double)threads_needed / (double)num_blocks);
    // round up
    threads_needed = threads_per_block * num_blocks; 
    if (threads_per_block > device.maxThreadsPerBlock) {
        // TODO split data to meet this constraint
        throw std::invalid_argument("Can not achieve speed up requirement, please reset speed up factors");
    }
    
    double mem_for_each_thread = (double)device.sharedMemPerBlock / (double)threads_per_block;
    double mem_required = (double)linear_data_size / (double)threads_needed;
    // validate strategy
    if (mem_for_each_thread < mem_required) {
        throw std::out_of_range("mem allocated for each thread is less than mem requirement.");
    }
    return ParallelStrategy(num_blocks, threads_per_block, 
        device.sharedMemPerBlock, threads_needed, partition, linear_data_size);
}

std::string ParallelStrategy::ToString() {
    std::stringstream buffer;
    auto actual_width = ceil((double)shared_memory_per_block / (double)num_threads_per_block);
    auto expect_width = ceil((double)data_size / (double)num_total_threads);
    buffer << "expect data size: " << data_size << "\n";
    buffer << "actual data size: " << num_total_threads * actual_width << "\n";
    buffer << "expect total threads: " << num_total_threads << "\n";
    buffer << "actual total threads: " << num_blocks * num_threads_per_block << "\n";
    buffer << "num block: " << num_blocks << "\n";
    buffer << "num threads per block: " << num_threads_per_block << "\n";
    buffer << "expect thread data size: " << expect_width <<"\n";
    buffer << "actual thread data size: " << actual_width << "\n";

    return buffer.str();
  
}