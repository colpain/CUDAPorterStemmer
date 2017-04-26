#include "all.cuh"
#include "data_handler.cuh"
#include "utils.cuh"
#include "porter_stemming_cpu.cuh"



#define _WRITE_TO_FILE_
#define PARALLEL_EFFORT 21000
static char* gpu_result;

#define INPUT_FILE "C:\\test\\last_block.10"
#define GPU_OUTPUT "C:\\test\\test.txt"
/*
#define INPUT_FILE "../last_block.10"
#define GPU_OUTPUT "/tmp/test.txt"
*/

// becuase input data size are known, so strategy can be computed in preprocess 
static ParallelStrategy strategy;
static ParallelStrategy* d_strategy;
static uint32_t* pos_holder;
static uint32_t* d_pos_holder;
static cudaDeviceProp prop;


double StemmingGPU(FileReader& data_handler);
double StemmingCPU(FileReader& data_handler);
void PrepareGPU(int32_t data_size);

int main(int argc, char* argv[])
{
    std::string file_name;
    if ( argc != 2 ) { // argc should be 2 for correct execution
        std::cout << "Please pass file name" << std::endl;
        //exit(-1);
    } else { 
    	file_name = argv[1];
    }
    FileReader reader;
    file_name = INPUT_FILE;
    reader.Read(file_name);
    reader.PadString();





    PrepareGPU(reader.GetDataSize());
    StemmingGPU(reader);




    return 0;
}



__global__ void Kernel(char* raw_data, char* output, uint32_t* start, uint32_t* length, ParallelStrategy* strategy, uint32_t* pos_holder);

double StemmingGPU(FileReader& reader){

    for (uint32_t i = 0; i < strategy.partition; ++i) {
        uint32_t start_offset = i * strategy.shared_memory_per_block * strategy.num_threads_per_block;
        uint32_t length = 0;
        uint32_t* d_length = nullptr;
        uint32_t* d_start_offset = nullptr;

        cudaMalloc((void**)&d_length, sizeof(uint32_t));
        cudaMalloc((void**)&d_start_offset, sizeof(uint32_t));
        // is size smaller than GPU memory
        if(strategy.partition == 1) {
            length = reader.GetDataSize();
        }
        // if the partition is last  
        else if(strategy.partition > 1 && i == strategy.partition - 1) {
            length = reader.GetDataSize() - strategy.shared_memory_per_block * i;
        } 
        else {
            length = strategy.shared_memory_per_block;
        }
        if (length == 0) {
            throw std::invalid_argument("0 length partition");
        }

        // length -2 for padding elements
        length -= 2;
        // count processing time and memory cpy time
        // strategy insures data satisfying all strategy constraints, for details
        // please check ParallelStrategy class
        cudaMemcpy(d_start_offset, &start_offset, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_length, &length, sizeof(uint32_t), cudaMemcpyHostToDevice);
        // invoke kernel function
        Kernel<<<strategy.num_blocks, strategy.num_threads_per_block>>>
           (reader.RawDataToKernel(), 
            reader.OutputDataToKernel(), 
            d_start_offset, d_length, d_strategy, d_pos_holder);
        length += 2;
    }
    auto status = cudaDeviceSynchronize();
    if(status) {
        std::cout << "Error happened synchronize " << std::endl;
        std::cout << "Error code: " << status << std::endl;
        std::cout << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
    
    gpu_result = reader.RawDataFromKernel();
    status = cudaMemcpy(pos_holder, d_pos_holder, 2 * sizeof(uint32_t) * strategy.num_total_threads, cudaMemcpyDeviceToHost);
    if(status) {
        std::cout << "Error happened copy memory from kernel to host " << std::endl;
        std::cout << "Error code: " << status << std::endl;
        std::cout << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
    

    // takes 5ms
    //stem the word in the intersection of threads
    PorterStemmingCPU cleaner(gpu_result);

    for(uint32_t i = 1; i < 2 * strategy.num_total_threads - 1; i+=2){
        if(pos_holder[i] != 0 && pos_holder[i+1] != 0) {
            pos_holder[i] += 1;
            pos_holder[i+1] += 1;
            for(int32_t j = pos_holder[i]; j < pos_holder[i+1] + 1; ++j) {
                toLowerCPU(gpu_result + j);
            }
            int32_t tail = cleaner.stem(pos_holder[i], pos_holder[i+1]);

            for (uint32_t k = tail + 1; k < pos_holder[i+1] + 1; ++k)
            {
                gpu_result[k] = ' ';
            }
            for(int32_t j = pos_holder[i]; j < tail + 1; ++j) {
                toUpperCPU(gpu_result + j);
            }

        }
    }


    // remove spaces
    std::vector<char> dest_string(reader.GetDataSize());
    trim_whitespace(&dest_string[0], &gpu_result[0]);
    //std::cout << dest_string.data();

    return 0;
}


__global__ void Kernel(char* raw_data, char* output, uint32_t* start, uint32_t* length, ParallelStrategy* strategy, uint32_t* pos_holder) {

    // advance pointer for ${start} steps
    char* data = raw_data + *start + 1;
    char* shifted_output = output + *start + 1;

    Scanner stemmer(data, shifted_output, *length, *strategy);
    stemmer.Scan(pos_holder); // use strategy.num_total_thread as length
        
}

void PrepareGPU(int32_t data_size) {
    // get device property for parallel strategy
    prop = GetDeviceProperty(0);
    // compute parallel strategy
    strategy = OptimalLinearDataStrategy(prop, data_size, PARALLEL_EFFORT);
    cudaMalloc((void**)&d_strategy, sizeof(ParallelStrategy));
    cudaMemcpy(d_strategy, &strategy, sizeof(ParallelStrategy), cudaMemcpyHostToDevice);
    // position holder for cleaning
    cudaMalloc((void**)&d_pos_holder,sizeof(uint32_t) * (2 * strategy.num_total_threads));
    pos_holder = new uint32_t[2 * strategy.num_total_threads];
}

