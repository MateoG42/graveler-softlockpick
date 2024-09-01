#include<curand.h>
#include<curand_kernel.h>
#include<stdio.h>
#include<math.h>
#include<random>
#include<chrono>

__global__ void init_curand(unsigned int seed, curandState* state)
{
    //get thead index and set rng
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void roll_kernel(curandState* state, int* result)
{
    //get index for current thread
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    //get the rng
    curandState local_state = state[idx];
    //declare counters
    int par_turns = 0;
    int safe_turns = 0;

    //for each turn
    for(int i = 0; i < 231; i++)
    {
        float randf = curand_uniform(&local_state);//get random float between 0 and 1 with uniform distribution
        state[idx] = local_state;//update rng
        randf *= 10;//multiply by 10
        int roll = (int)randf % 4;//mod 4 to get a roll between 0 and 4

        //update relevant counter
        switch(roll)
        {
            case 0:
                par_turns += 1;
                break;
            default:
                safe_turns += 1;
        }

        //if fail state is reached, stop executing loop
        if(safe_turns > 54)
        {
            break;
        }
    }    

    result[idx] = par_turns; //save result
}

int main(int argc, char* argv[])
{
    int max_tries = atoi(argv[1]); //number of attempts before max out
    int n = 256; //number of parallel threads on gpu

    int total_steps = (max_tries + n - 1)/n; //get number of loops to run splitting the max tries by the thread count rounded up
    int max_par = 0; //our highest count

    //allocate the rng in gpu
    curandState* d_state;
    cudaMalloc((void**)&d_state, n*sizeof(curandState));

    init_curand<<<1, n>>>(54321, d_state);

    //start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < total_steps; i++) //run total_steps batches of n threads
    {
        //gpu memory allocation
        int size = n * sizeof(int); //size in bytes
        int* d_result;
        cudaMalloc((void**)&d_result, size);
        cudaMemset(d_result, 0, size); //initialize as 0

        //host memory to copy to
        int* result;
        result = (int*)malloc(size);
        
        //perform rolls
        roll_kernel<<<1, n>>>(d_state, d_result);

        //copy over
        cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

        //get max number of paralysis turns in batch
        int max = 0;
        for(int i = 0; i < n; i++)
        {
            if(result[i] > max)
            {
                max = result[i];
            }
        }
        if(max > max_par) //replace overall max if batch max is greater
        {
            max_par = max;
        }
    
        //gree allocated memory
        cudaFree(d_result);
        free(result);
    }
    //end timer
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    cudaFree(d_state); //free rng

    //outputs results
    printf("Max paralysis turns: %d\n", max_par);
    printf("Took %d ms\n", (int)duration.count());

    return 0;
}