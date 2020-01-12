#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

const int FILTER_WIDTH = 9;
const int BLOCK_SIZE = 256;

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

// Read 1D image text file to int array, also read the size of the image
void initData(string file, int **data, int *size) {
	int n;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> n;
		int *temp = new int[n];
		for( i = 0; i < n; i++){
			myfile >> temp[(int)i];
		}
		myfile.close();
		*data = temp;
		*size = n;
	}
	else {
		cout << "ERROR: File " << file << " not found!\n";
		exit(0);
	}
	cout << i << " entries imported\n";
}

// Save to 1D image text file
void saveResult(string file, int data[], int size) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << size << "\n";
		for (i = 0; i < size; i++){
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

// Implement the kneral function for 1D smoothing in GPU
__global__
void calculateResult(int size, int *data, int *result){
    // Iterate through each data pixel, incorporating parallelism with thread blocks and grid
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int stride = blockDim.x * gridDim.x;
     
     int len = FILTER_WIDTH/2;
     
     for (int i = index; i < size; i += stride){
        // Store sum and count since this is a mean filter (calculatingaverage)
        int sum = 0;
        int count = 0;
        // Convolute the 1D mean filter to the image
        for(int j = i-len; j < i+len+1; j++){
            // Dont use any pixel in the filter that extends past the input image
            if(j >= 0 && j < size){
                sum += data[j];
                count++;
            }
        }
        // Calculate result (average) from stored sum and count variables
        result[i] = sum/count;
    }
}

// GPU implementation
void GPU_Test(int data[], int result[], int size) {
	// input:
	//	int data[] - int array holding the original image
	//	int size - the size (length) of the image
	//  output:
	//	int result[] - int array holding the smoothed image

	// Allocate device memory for data[] and result[] and copy data onto the device
    int *r, *d;
    
    cudaMalloc((void**)&r, size*sizeof(int));
    cudaMalloc((void**)&d, size*sizeof(int));
    
    cudaMemcpy(r, result, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d, data, size*sizeof(int), cudaMemcpyHostToDevice);
    
    // calculate the number of thread blocks to use
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Call the kernel function
    calculateResult<<<numBlocks,BLOCK_SIZE>>>(size, d, r);

    // Copy reuslt from device to host
    cudaMemcpy(result, r, size*sizeof(float), cudaMemcpyDeviceToHost);
    
	// Free device memory
    cudaFree(&d);
    cudaFree(&r);
}




// CPU implementation
void CPU_Test(int data[], int result[], int size) {
	// input:
	//	int data[] - int array holding the original image
	//	int size - the size (length) of the image
	// output:
	//	int result[] - int array holding the smoothed image

	// Smooth the image with filter size = FILTER_WIDTH
	// apply partial filter for the border
 
    int len = FILTER_WIDTH/2;
    
    // Iterate through each data pixel
    for(int i = 0; i < size; i++){
        // Store sum and count since this is a mean filter (calculating average)
        int sum = 0;
        int count = 0;
        // Convolute the 1D mean filter to the image
        for(int j = i-len; j < i+len+1; j++){
            // Dont use any pixel in the filter that extends pastthe input image
            if(j >= 0 && j < size){
                sum += data[j];
                count++;
            }
        }

        // Calculate result (average) from stored sum and count variables
        result[i] = sum/count;
        
    }
    

}

int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image1D.txt" : argv[1];

	int size;
	int *dataForCPUTest;
	int *dataForGPUTest;

	initData(inputFile, &dataForCPUTest, &size);
	initData(inputFile, &dataForGPUTest, &size);

	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, size);
	auto endCPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";

	displayResult(dataForCPUTest, resultForCPUTest, size);

	saveResult("1D_result_CPU.txt", resultForCPUTest, size);

	cout << "\n";

	cout << "GPU Implementation\n";

	auto startGPU = chrono::steady_clock::now();
	GPU_Test(dataForGPUTest, resultForGPUTest, size);
	auto endGPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	displayResult(dataForGPUTest, resultForGPUTest, size);
	saveResult("1D_result_GPU.txt",resultForGPUTest, size);

	return 0;
}
