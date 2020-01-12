#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

using namespace std;

const int FILTER_WIDTH = 7;
const int BLOCK_SIZE = 256;

int FILTER[FILTER_WIDTH*FILTER_WIDTH] = {
	1,4,7,10,7,4,1,
	4,12,26,33,26,12,4,
	7,26,55,71,55,26,7,
	10,33,71,91,71,33,10,
	7,26,55,71,55,26,7,
	4,12,26,33,26,12,4,
	1,4,7,10,7,4,1
};

// Display the first and last 10 items
void displayResult(const int original[], const int result[], int size) {
	cout << "Display result: ";
	cout << "(original -> result)\n";

	for (int i = 0; i < 10; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
	cout << ".\n.\n.\n";

	for (int i = size - 10; i < size; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
}

void initData(string file, int **data, int *sizeX, int *sizeY) {
	int x;
	int y;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> x;
		myfile >> y;

		int *temp = new int[x * y];
		for( i=0; i < x * y; i++){
			myfile >> temp[(int)i];
		}
		myfile.close();
		*data = temp;
		*sizeX = x;
		*sizeY = y;
	}
	else {
		cout << "ERROR: File " << file << " not found!\n";
		exit(0);
	}
	cout << i << " entries imported\n";
}

void saveResult(string file, int data[], int sizeX, int sizeY) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << sizeX << "\n";
		myfile << sizeY << "\n";
		for (i = 0; i < sizeX * sizeY; i++){
			myfile << data[i] << "\n";
		}
		myfile.close();
	}
	else {
		cout << "ERROR: Cannot save to " << file << "!\n";
		exit(0);
	}
	cout << i << " entries saved\n";
}

// Kernel function for 2D smoothing in GPU
__global__
void calculateResult(int sizeX, int sizeY, int *data, int *result, int *filter){
    
    int halfFilterWidth = FILTER_WIDTH/2;
    //int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    //int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int strideY = blockDim.y * gridDim.y;
    
    // start from last column in image
    for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < sizeX ; x += stride){
    // start from last row in image
        for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < sizeY; y += strideY){
            // store numberator and denominator for convolution calculation
            int numerator = 0;
            int denominator = 0;
            
            // traverse the filter in the x-direction
            for(int filterX = FILTER_WIDTH - 1; filterX >= 0; filterX--){
                // traverse the filter in the y-direction
                for(int filterY = FILTER_WIDTH -1; filterY >= 0; filterY--){
                    
                    int xPos = x + filterX -halfFilterWidth;
                    int yPos = y + filterY - halfFilterWidth;
                    
                    // adjust xPos to accomodate edges in grid
                    if(xPos < 0){
                        xPos = 0;
                    }
                    else if(xPos < sizeX){
                    }
                    else{
                       xPos = sizeX - 1;
                    }
                    
                    // adjust yPos to accomodate edges in grid
                    if(yPos < 0){
                        yPos = 0;
                    }
                    else if(yPos < sizeY){
                    }
                    else{
                       yPos = sizeY - 1;
                    }
                    
                    // adjust numerator and denominator
                    numerator += data[yPos * sizeX + xPos] * filter[filterY * FILTER_WIDTH + filterX];
                    denominator += filter[filterY * FILTER_WIDTH + filterX];
                }
            }
            // store result
            result[y * sizeX + x] = numerator/denominator;
        }
    }
}

// GPU implementation
void GPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the smoothed image

    int size = sizeX * sizeY;
	// Allocate device memory for result[], data[] and FILTER[] and copy data onto the device
    int *r, *d, *f;

    cudaMalloc((void**)&r, size*sizeof(int));
    cudaMalloc((void**)&d, size*sizeof(int));
    cudaMalloc((void**)&f, size*sizeof(int));

    cudaMemcpy(r, result, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d, data, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(f, FILTER, FILTER_WIDTH*FILTER_WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    
	// Start timer for kernel
	auto startKernel = chrono::steady_clock::now();

    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Call the kernel function
    calculateResult<<<numBlocks, BLOCK_SIZE>>>(sizeX, sizeY, d, r,f);

	// End timer for kernel and display kernel time
	cudaDeviceSynchronize(); // <- DO NOT REMOVE

	auto endKernel = chrono::steady_clock::now();
	cout << "Kernel Elapsed time: " << chrono::duration <double, milli>(endKernel - startKernel).count() << "ms\n";

	// Copy reuslt from device to host
    cudaMemcpy(result, r, size*sizeof(float), cudaMemcpyDeviceToHost);
	// Free device memory
    cudaFree(&d);
    cudaFree(&r);
    cudaFree(&f);

}



// CPU implementation
void CPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the smoothed image

	// Smooth the image with filter size = FILTER_WIDTH
	// apply partial filter for the border
 
        
    int halfFilterWidth = FILTER_WIDTH/2;
    
    // start from last column in image
    for(int x = sizeX -1; x >= 0; x--){
    // start from last row in image
        for(int y = sizeY - 1; y >= 0; y--){
            // store numberator and denominator for convolution calculation
            int numerator = 0;
            int denominator = 0;
            
            // traverse the filter in the x-direction
            for(int filterX = FILTER_WIDTH - 1; filterX >= 0; filterX--){
                // traverse the filter in the y-direction
                for(int filterY = FILTER_WIDTH -1; filterY >= 0; filterY--){
                    
                    int xPos = x + filterX -halfFilterWidth;
                    int yPos = y + filterY - halfFilterWidth;
                    
                    // adjust xPos to accomodate edges in grid
                    if(xPos < 0){
                        xPos = 0;
                    }
                    else if(xPos < sizeX){
                    }
                    else{
                       xPos = sizeX - 1;
                    }
                    
                    // adjust yPos to accomodate edges in grid
                    if(yPos < 0){
                        yPos = 0;
                    }
                    else if(yPos < sizeY){
                    }
                    else{
                       yPos = sizeY - 1;
                    }
                    
                    // adjust numerator and denominator
                    numerator += data[yPos * sizeX + xPos] * FILTER[filterY * FILTER_WIDTH + filterX];
                    denominator += FILTER[filterY * FILTER_WIDTH + filterX];
                }
            }
            // store result
            result[y * sizeX + x] = numerator/denominator;
        }
    }
}

// The input is a 2D grayscale image
// The image is flattened into a text file of pixel values.
int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image2D.txt" : argv[1];

	int sizeX;
	int sizeY;
	int *dataForCPUTest;
	int *dataForGPUTest;	

	initData(inputFile, &dataForCPUTest, &sizeX, &sizeY);
	initData(inputFile, &dataForGPUTest, &sizeX, &sizeY);

	int size = sizeX * sizeY;
	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, sizeX, sizeY);
	auto endCPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";

	displayResult(dataForCPUTest, resultForCPUTest, size);

	saveResult("2D_result_CPU.txt",resultForCPUTest, sizeX, sizeY);

	cout << "\n";
	cout << "GPU Implementation\n";

	auto startGPU = chrono::steady_clock::now();
	GPU_Test(dataForGPUTest, resultForGPUTest, sizeX, sizeY);
	auto endGPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	displayResult(dataForGPUTest, resultForGPUTest, size);
	saveResult("2D_result_GPU.txt",resultForGPUTest, sizeX, sizeY);

	return 0;
}
