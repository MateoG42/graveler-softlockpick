#include<cstdlib>
#include<iostream>
#include<chrono>

using namespace std;

int main(int argc, char* argv[])
{
    int rollCounts[] = {0, 0, 0, 0};
    int epochs = 0;
    int maxPar = 0;

    srand(time(NULL));

    //continue trying until we finish or reach max attempts'
    auto startTime = chrono::high_resolution_clock::now();
    while(rollCounts[0] < 177 && epochs < atoi(argv[1]))
    {
        int rollCounts[] = {0, 0, 0, 0};

        //for each turn
        bool safe = true;
        for(int i = 0; i < 231 && safe == true; i++)
        {
            int roll = rand() % 4; //roll d4
            rollCounts[roll] += 1; //increment whatever was rolled

            //check if we runn out of safe turns to skip remaining iterations
            if((rollCounts[1] + rollCounts[2] + rollCounts[3]) > 54)
            {
                safe = false;
            }
        }

        //save max paralysis turns
        if(rollCounts[0] > maxPar)
        {
            maxPar = rollCounts[0];
        }   

        epochs++; 
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Max paralysis turns: " << maxPar << endl;
    std::cout << "Ran over " << epochs << " epochs" << endl;
    std::cout << "Took " << duration.count() << " ms" << endl;
    
    return 0;
}