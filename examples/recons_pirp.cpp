#include <iostream>
#include <chrono>

#include "IRPSolver.h"

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        throw std::runtime_error("No CUDA capable GPU device found!");
    }

    int deviceId = rank % deviceCount;
    cudaSetDevice(deviceId);

    int rows = 130;
    int cols = 130;
    IntArray imSize {rows, cols};

    // Each process handles the same number of angles
    int totalAngles = 90;
    int numAngles = totalAngles / size;
    int numImages = 1;
    F2DArray fresnelNumbers(1, FArray(1, 1.0f));

    // load images
    FArray holograms(numAngles * numImages * rows * cols);
    float buffer[rows * cols];
    float **dstack = new float*[numAngles];
    char *filename = new char[100];
    for (int i = 0; i < numAngles; i++) {
		sprintf(filename, "/home/hujiarui/workspace/irp_code/data/detector_intensity%02d_130x130px.raw", i + 1 + rank * numAngles);
        FILE *in = fopen(filename, "rb");
        fread(buffer, sizeof(float), rows * cols, in);
        dstack[i] = new float[rows * cols];
        for (int x = 0; x < cols; x++) {
            for (int y = 0; y < rows; y++) {
                dstack[i][x * rows + y] = buffer[y * cols + x];
            }
        }
        fclose(in);
    }
    for (int i = 0; i < numAngles; i++) {
        memcpy(holograms.data() + i * rows * cols, dstack[i], rows * cols * sizeof(float));
    }    

    for (int i = 0; i < numAngles; i++) {
        delete[] dstack[i];
    }
    delete[] dstack; delete[] filename;

    auto start = std::chrono::high_resolution_clock::now();
    FArray result = reconstruct_irp(holograms, totalAngles, numImages, imSize, fresnelNumbers, MPI_COMM_WORLD, 2, 1e-5f, 1);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "Finished reconstruction for " << totalAngles << " angles on " << size << " GPUs!" << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

        FILE *out = fopen("../result.raw", "w");
        fwrite(result.data(), sizeof(float), rows * cols * rows, out);
        fclose(out);
    }

    MPI_Finalize();
    return 0;
}