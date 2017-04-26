//
//  data_handler.cc
//  PorterStemmingAlgorithm
//
//  Created by Charles on 2015-03-20.
//  Copyright (c) 2015 Charles. All rights reserved.
//

#include "data_handler.cuh"
#include "all.cuh"


FileReader::FileReader() {
    data_.clear();
    d_data_ = nullptr;
    d_output_ = nullptr;
    
    h_data_ = nullptr;
}

void FileReader::Read(std::string path) {
    ParseFile(path);
    std::string line;
    std::stringstream string_buffer;
    reader_.open("./temp.txt");
    if (reader_.is_open())
    {
        while ( std::getline (reader_,line) )
        {
            string_buffer << line << ' ';
        }
        reader_.close();
    }
    else {
        std::cout << "Unable to open file: " << path << std::endl;
        exit(0);
    }
    reader_.close();
    if( remove( "temp.txt" ) != 0 )
      perror( "Error deleting file" );

    data_ = string_buffer.str();
    
}

void FileReader::PadString() {
    data_.push_back('.');
    data_.insert(data_.begin(), '.');
}

char* FileReader::GetRawData() {
    if(h_data_ == nullptr)
        std::strcpy(h_data_, data_.data());

    return h_data_;
}


char* FileReader::RawDataToKernel() {
    if(d_data_ != nullptr) {
        cudaFree(d_data_);
    }
    cudaMalloc((void**)&d_data_, data_.size() * sizeof(char));
    cudaMemcpy(d_data_, data_.data(), data_.size() * sizeof(char), cudaMemcpyHostToDevice);
    // feed padded data
    return d_data_;
}

char* FileReader::OutputDataToKernel() {
    if(d_output_ != nullptr) {
        cudaFree(d_output_);
    }
    cudaMalloc((void**)&d_output_, data_.size() * sizeof(char));
    // initialize data to write
    cudaMemcpy(d_output_, data_.data(), data_.size() * sizeof(char), cudaMemcpyHostToDevice);
    return d_output_;
}

char* FileReader::RawDataFromKernel() {
    if(h_data_ != nullptr) {
        delete h_data_;
    }
    cudaError_t status = cudaMallocHost((void**)&h_data_, data_.size() * sizeof(char));
    if (status != cudaSuccess) {
          printf("Error allocating pinned host memoryn");
          std::cout << "Error code: " << status << std::endl;
          exit(-1);
    }
    cudaMemcpy(h_data_, d_output_, data_.size() * sizeof(char), cudaMemcpyDeviceToHost);
    h_data_[data_.size() - 1] = 0;

    return h_data_;
}


/*
 Scanner class
*/
__device__ Scanner::Scanner(char* raw_data, char* output, uint32_t data_size, ParallelStrategy strategy) {
    
    raw_data_ = raw_data;
    output_ = output;
    data_size_ = data_size;
    thread_data_size_ = ceil((double) data_size_ / (double)strategy.num_total_threads );
    //thread_data_size_ = ceil((double)strategy.shared_memory_per_block / (double)strategy.num_threads_per_block);
    width_ = thread_data_size_;
    global_index_ = blockIdx.x * blockDim.x + threadIdx.x;

}

__device__ Scanner::~Scanner() {
    
}

__device__ void Scanner::Scan(uint32_t* pos_holder) {
    // initialize pos holder
    pos_holder[2 * global_index_] = 0;
    pos_holder[2 * global_index_ + 1] = 0;
    // get thread start and end position
    uint32_t start = global_index_ * width_;
    uint32_t end = (global_index_ + 1) * width_;


    Word pos;

    for(int32_t i = start; i < end; ++i) {
        if(i < data_size_) {

            ParseLetters(&pos, i, pos_holder, output_);

            if(pos.start != POSFLAG::UNSET && pos.end != POSFLAG::UNSET) {

		        shifter_ = output_ + pos.start;
                uint32_t length = pos.end - pos.start;
                for(int32_t i = 0; i < length + 1; ++i) {
                    toLower(shifter_ + i);
                }
		        auto tail = STEMMERGPU::stem_func(shifter_, length, shifter_);
                for(int32_t k = tail + 1; k < length + 1; ++k) {
                    shifter_[k] = ' ';
                }
                // to upper
                for(int32_t k = 0; k < tail + 1; ++k) {
                    toUpper(shifter_ + k);
                }
                pos.start = POSFLAG::UNSET, pos.end = POSFLAG::UNSET;


           }

        }
    }

     // mark start
    if(pos.end == POSFLAG::UNSET && pos.start != POSFLAG::UNSET) {
        pos_holder[2 * global_index_ + 1] = pos.start;
        pos.start = POSFLAG::UNSET;
    }

    

    
}




__device__ bool Scanner::IsAlpha(const char& c) {
    return ((c >='a' && c <='z') || (c >='A' && c <='Z'));
}

 __device__ void Scanner::ParseLetters(Word* word, int32_t current_pos, uint32_t* pos_holder, char* output) {
     if(!isDelimiters(raw_data_[current_pos])) {
         if(!isDelimiters(raw_data_[current_pos - 1])) {

         }
         else if(isDelimiters(raw_data_[current_pos - 1])) {
             output[current_pos - 1] = ' ';
             // if end hasn't set and position has't been set
             if(word->end == POSFLAG::UNSET) {
                 word->start = current_pos;
             }
             else if (word->end != POSFLAG::UNSET) {
                 // mismatch part start
             }
         }
         else if(!isDelimiters(raw_data_[current_pos + 1])) {
         }
         else if(isDelimiters(raw_data_[current_pos + 1])) {
             output[current_pos + 1] = ' ';
             // detect end position, if start has been set, then it's a pair
             // add end 
             if(word->start != POSFLAG::UNSET) {
                 word->end = current_pos;
             }
             // if start is unset, mark end
             else if (word->start == POSFLAG::UNSET) {
                 pos_holder[2 * global_index_] = current_pos;

             }
         }
     }

     else if (isDelimiters(raw_data_[current_pos])) {
         output[current_pos] = ' ';
         if(!isDelimiters(raw_data_[current_pos - 1])) {
             // detect end position, if start has been set, then it's a pair
             // add end 
             if(word->start != POSFLAG::UNSET) {
                 word->end = current_pos - 1;
             }
             // if start is unset, mark end
             else if (word->start == POSFLAG::UNSET) {
                 pos_holder[2 * global_index_] = current_pos - 1;
             }
         }
         else if(isDelimiters(raw_data_[current_pos - 1])) {
             output[current_pos - 1] = ' ';
         }
         else if(!isDelimiters(raw_data_[current_pos + 1])) {
             if(word->end == POSFLAG::UNSET) {
                 word->start = current_pos + 1;
             }
             else if(word->end == POSFLAG::UNSET) {
                 // mismatch start
             }
         }
         else if(isDelimiters(raw_data_[current_pos + 1])) {
             output[current_pos + 1] = ' ';
         }
     }
 }






 


 // SONAR FILE PARSER====================

using namespace std;
#define SONAR_COMPRESSION_NONE 0
#define SONAR_COMPRESSION_LZ4 1
// Values for the block_leader_disk_t.block_type
#define FULL_BLOCK 1 // No more room in this block
#define EMPTY_BLOCK 2 // All fields in this blocks are undefined
#define NUM_BOUNDS 4
#define COL_HEADER_MAGIC 0xdeadbeef


typedef unsigned int bdword;
typedef int sonar_item_offset_t;

#pragma pack(push,1)

struct block_header_disk_t {

    int32_t item_size; // 0 if strings block

    int64_t data_part_size; // How much of it is data..

    int32_t is_compressed; // 0 if the block is not compressed , else one of the SONAR_COMPRESSION values above.

    int64_t compressed_data_size;

    block_header_disk_t() {
        item_size = 0; 
        data_part_size = 0; 
        is_compressed = 0; 
        compressed_data_size = 0;
    }

};



#define overall_block_size(x) ( (x).data_part_size  )

#define overall_compressed_size(x) ( (x).compressed_data_size )



struct block_leader_disk_t {

    uint32_t magic_number;

    int32_t block_type; // full, empty, or still open

    uint64_t first_document;

    uint64_t document_count;

    int32_t prefered_type; // Type which most of data was inserted as

    int32_t markers_size; // Size in bytes of the null, undefined, and soft-deleted  documents bit fields

    time_t ctime;

    time_t mtime;
    block_leader_disk_t() {
        magic_number = 0;
        block_type = EMPTY_BLOCK;
        first_document = 0;
        document_count = 0;
        prefered_type = 0;
        markers_size = 0;
        ctime = 0;
    }
    union bounds_t {
        template<class T> struct template_bounds_t {
            T lower_values[NUM_BOUNDS];
            T upper_values[NUM_BOUNDS];
        };
        template_bounds_t<double> number_bounds;
        template_bounds_t<int64_t> long_bounds;
    } bounds;
};



#define MAXFILESIZE (24*1024*1024)



#define IS_DELIM(c) ((unsigned int) (*c) < 256 and is_word_delimiter[(unsigned int) (*c)])



static char *is_word_delimiter = nullptr;



static void init_word_delimiters()

{

    static unsigned char delimiters[] = { ' ', '.', ',', '\t', '\r', '\n', ':', ';', '"', '\'', '/', '\\', 0x27, '>', '<', '=', '-', '+', '?', '!', '(', ')', '[', ']', '{', '}' };

    is_word_delimiter = (char *) malloc(256);

    bzero(is_word_delimiter, 256);

    for (unsigned int i = 0; i < sizeof(delimiters); i++) {

        is_word_delimiter[(int) delimiters[i]] = 1;

    }

}







std::string FileReader::ParseFile(std::string file_name)

{

    int fd;

    char *nulls; // pointer to which elements are null bitmap

    char *undefs; // pointer to which elements are undefined bitmap

    block_leader_disk_t leader;

    struct block_header_disk_t header;

    const uint32_t *string_table; // Cummulative length of the strings

    const char *string_body; // The actual strings

    char *buf;

    



    

    init_word_delimiters();

    

    fd = open(file_name.c_str(), O_RDONLY);

    

    buf = new char[MAXFILESIZE];

    int dsize = read( fd, buf, MAXFILESIZE );

    if ( fd <= 0 ) {

        exit(1);

    }

    close(fd);

    

    // Read block leader

    

    auto fptr = buf;

    memcpy(&leader, fptr, sizeof( block_leader_disk_t));

    fptr += sizeof( block_leader_disk_t);

    

    // Read block nulls and undefs bitmap

    

    nulls = new char[leader.markers_size];

    undefs = new char[leader.markers_size];

    

    memcpy(nulls, fptr, leader.markers_size);

    fptr += leader.markers_size;

    memcpy(undefs, fptr, leader.markers_size);

    fptr += leader.markers_size;

    

    // Read block header

    

    memcpy(&header, fptr, sizeof( header));

    fptr += sizeof( header);

    

    // get string table pointer

    string_table = ( uint32_t *) fptr;

    fptr += sizeof(uint32_t) * leader.document_count ;

    

    // get strings body

    string_body = fptr;

    

    // print the strings

    int string_id = 0 ;

    auto string_ptr = string_body;

    auto string_len = string_table[0];

    std::ofstream writer;
    writer.open("./temp.txt");


    std::string str_to_return;

    do {

        string string_to_print( string_ptr, string_len );

        writer << string_to_print << std::endl;


        string_id++;

        if ( string_id < leader.document_count ) {

            string_ptr += string_len;

            string_len = string_table[string_id] - string_table[string_id-1];

        } else {

            break;

        }

    } while ( true ) ;

    writer.close();

    delete [] buf;

    delete [] nulls;

    delete [] undefs;


    return str_to_return;

}

