//
//  data_handler.h
//  PorterStemmingAlgorithm
//
//  Created by Charles on 2015-03-20.
//  Copyright (c) 2015 Charles. All rights reserved.
//

#ifndef __PorterStemmingAlgorithm__data_handler__
#define __PorterStemmingAlgorithm__data_handler__

#include "all.cuh"
#include "utils.cuh"

#define bzero(memArea, len)  memset((memArea), 0, (len))
enum POSFLAG
{
    UNSET = -5, SET = -6
};
/*
    This class read plain text from file
    and store it into char* or string
*/
class FileReader {
    // file object
    std::ifstream reader_;
    // store raw data
    std::string data_;
    // device raw data
    char* d_data_;
    // device output
    char* d_output_;
    // host output
    char* h_data_;
public:
    FileReader();
    ~FileReader() {

        cudaFree(d_data_);
        cudaFreeHost(h_data_);
    }
    
    FileReader(const FileReader& other) {
        data_ = other.data_;
    }
    FileReader(std::string s) { data_ = s; }
    
    std::string& Data() { return data_; }
    
    void Read(std::string path);
    /*
        This function insert a blank text before and after
        data
    */
    void PadString();

    char* GetRawData();

    char* RawDataToKernel();
    char* OutputDataToKernel();

    char* RawDataFromKernel();

    int GetDataSize() { return data_.size() * sizeof(char); }
    // debugging methods
    std::string ToString() { return data_; }
    // read jSonar file
    std::string ParseFile(std::string file_name);
};

/*
    This is a thread level device class,
    each class has a scanner
    This class uses stemmer, Scanner object
    scans raw data and pass word offsets to stemmer
    Each thread has one Scanner, read and write data
*/
#include "stemmer_function.cuh"

struct Word {
    __device__ Word() {
        start = POSFLAG::UNSET;
        end = POSFLAG::UNSET;
    }
    int32_t start;
    int32_t end;
};

class Scanner {

    const char* raw_data_;
    char* output_;
    // point to output array
    char* shifter_;

    uint32_t thread_data_size_;

    uint32_t data_size_;
    uint32_t width_;

    uint32_t global_index_;
    /*
        Scanning rules
    */



    __device__ void ParseLetters(Word* word, int32_t current_pos, uint32_t* pos_holder, char* output);
    __device__ bool IsAlpha(const char& c);

public:
    __device__ Scanner(char* raw_data, char* outut, uint32_t data_size, ParallelStrategy device);
    __device__ void Scan(uint32_t* pos_holder);
    __device__ ~Scanner() ;
};






#endif /* defined(__PorterStemmingAlgorithm__data_handler__) */
