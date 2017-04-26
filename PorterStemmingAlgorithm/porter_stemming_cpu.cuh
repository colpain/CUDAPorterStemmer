//

//  porter_stemming_shared_memory.h

//  PorterStemmingAlgorithm

//

//  Created by Charles on 2015-03-20.

//  Copyright (c) 2015 Charles. All rights reserved.

//



#ifndef __PorterStemmingAlgorithm__porter_stemming_cpu__

#define __PorterStemmingAlgorithm__porter_stemming_cpu__



#include "all.cuh"



class PorterStemmingCPU {

public:

    char* data_;



    PorterStemmingCPU(char* data) {

        data_ = data;

    }

    

    int stem(int start, int end);

    

private:

    

    /*

     create shared memory for each block to process

     use a efficient way to copy data from global memeory

     to shared memory



     also pay attension to avoid bank conflicts

     see:

     https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

     as examples

    */

    bool cons(int i);

    int m();

    int vowelinstem();

    int doublec(int j);

    int cvc(int i);

    

    // memmove and memcmp

    int ends(const char* s);

    void setto(const char*s);

    

    

    void r(const char* s);

    void step1ab();

    void step1c();

    void step2();

    void step3();

    void step4();

    void step5();

    

    int j;

    int k0;

    int k;

};
void toLowerCPU(char *p);
void toUpperCPU(char *p);
#endif /* defined(__PorterStemmingAlgorithm__porter_stemming_shared_memory__) */