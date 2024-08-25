#include<curand.h>
#include<curand_kernel.h>
#include<stdio.h>
#include<math.h>
#include<random>
#include<chrono>

__global__ void init_curand(unsigned int seed, curandState* state)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void roll_kernel(curandState* state, int* result)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    curandState local_state = state[idx];
    int parTurns = 0;
    int safe_turns = 0;

    //for each turn
    for(int i = 0; i < 231; i++)
    {
        float randf = curand_uniform(&local_state);
        state[idx] = local_state;
        randf *= 10;
        int roll = (int)randf % 4;

        if(roll == 0)
        {
            parTurns += 1;
        }
        else
        {
            safe_turns += 1;
        }

        if(safe_turns > 54)
        {
            break;
        }
    }    

    result[idx] = parTurns;
}

int main(int argc, char* argv[])
{
    int max_tries = atoi(argv[1]);
    int n = 256;

    int total_steps = (max_tries + n - 1)/n;
    int max_par = 0;

    //allocate the state in device
    curandState* d_state;
    cudaMalloc((void**)&d_state, n*sizeof(curandState));

    init_curand<<<1, n>>>(54321, d_state);

    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < total_steps; i++)
    {
        int size = n * sizeof(int);
        int* d_result;
        cudaMalloc((void**)&d_result, size);
        cudaMemset(d_result, 0, size);

        int* result;
        result = (int*)malloc(size);
        
        roll_kernel<<<1, n>>>(d_state, d_result);

        cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

        int max = 0;
        for(int i = 0; i < n; i++)
        {
            if(result[i] > max)
            {
                max = result[i];
            }
        }
        if(max > max_par)
        {
            max_par = max;
        }
    
        cudaFree(d_result);
        free(result);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    cudaFree(d_state);

    printf("Max paralysis turns: %d\n", max_par);
    printf("Took %d ms\n", (int)duration.count());

    return 0;
}