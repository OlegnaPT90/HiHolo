#include "IRPSolver.h"

Grid::Grid(int ix, int iy, int iz) : nx(ix), ny(iy), nz(iz), size(ix * iy * iz)
{
    cudaMalloc(&values, size * sizeof(float));
    blockSize = 1024;
    gridSize = (size + blockSize - 1) / blockSize;
}

Grid::Grid(int ix, int iz) : nx(ix), ny(ix), nz(iz), size(ix * ix * iz)
{
    cudaMalloc(&values, size * sizeof(float));
    blockSize = 1024;
    gridSize = (size + blockSize - 1) / blockSize;
}

Grid::~Grid()
{
    cudaFree(values);
}

void Grid::setValues(float value)
{
    initializeData<<<gridSize, blockSize>>>(values, value, size);
}

void Grid::setZeros()
{
    cudaMemset(values, 0, size * sizeof(float));
}

void Grid::scaleValues(float scale)
{
    scaleFloatData<<<gridSize, blockSize>>>(values, size, scale);
}

CUDAStreamART::CUDAStreamART(int ix, int iy, int N): nx(ix), ny(iy), numStreams(N)
{
    streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaMalloc((void**)&tp, (nx + 4) * ny * numStreams * sizeof(float));
    cudaMalloc((void**)&tp_back, (nx + 4) * ny * sizeof(float));
}

void CUDAStreamART::syncStreams()
{
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}

void CUDAStreamART::merge(float *project)
{
    cudaMemcpy(project, tp + 2 * ny, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    int blockSize = 1024;
    int numBlocks = (nx * ny + blockSize - 1) / blockSize;
    for (int i = 1; i < numStreams; i++) {
        addFloatData<<<numBlocks, blockSize>>>(project, tp + i * (nx + 4) * ny + 2 * ny, nx * ny);
    }
}

void CUDAStreamART::projectGrid(const Grid &grid, float *project, float phi)
{    
    for (int i = 0; i < numStreams; i++) {
        if (i == numStreams - 1) {
            grid_project(grid.getValues(), tp + i * (nx + 4) * ny, grid.getNx(), grid.getNz(),
                         phi, i * grid.getSize() / numStreams, grid.getSize(), streams[i]);
        } else {
            grid_project(grid.getValues(), tp + i * (nx + 4) * ny, grid.getNx(), grid.getNz(),
                         phi, i * grid.getSize() / numStreams, (i + 1) * grid.getSize() / numStreams, streams[i]);
        }
    }

    syncStreams();
    merge(project);
}

void CUDAStreamART::backProjectGrid(Grid &grid, const float *project, float phi, int size, int rank)
{
    cudaMemset(tp_back, 0, (nx + 4) * ny * sizeof(float));
    cudaMemcpy(tp_back + 2 * ny, project, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    int procSize = grid.getSize() / size;

    for (int i = 0; i < numStreams; i++) {
        if (i == numStreams - 1) {
            grid_back_project(grid.getValues(), tp_back, grid.getNx(), grid.getNz(), phi,
                              rank * procSize + i * procSize / numStreams, (rank + 1) * procSize, streams[i]);
        } else {
            grid_back_project(grid.getValues(), tp_back, grid.getNx(), grid.getNz(), phi,
                              rank * procSize + i * procSize / numStreams, rank * procSize + (i + 1) * procSize / numStreams, streams[i]);
        }
    }
    
    syncStreams();
}

void CUDAStreamART::maxMapGrid(Grid &grid, const float *project, float phi, int size, int rank)
{
    grid.setValues(1e11f);
    cudaMemset(tp_back, 0, (nx + 4) * ny * sizeof(float));
    cudaMemcpy(tp_back + 2 * ny, project, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
    int procSize = grid.getSize() / size;

    for (int i = 0; i < numStreams; i++) {
        if (i == numStreams - 1) {
            grid_max_map(grid.getValues(), tp_back, grid.getNx(), grid.getNz(), phi, 
                         rank * procSize + i * procSize / numStreams, (rank + 1) * procSize, streams[i]);
        } else {
            grid_max_map(grid.getValues(), tp_back, grid.getNx(), grid.getNz(), phi,
                         rank * procSize + i * procSize / numStreams, rank * procSize + (i + 1) * procSize / numStreams, streams[i]);
        }
    }

    syncStreams();
    int blockSize = 1024;
    int numBlocks = (procSize + blockSize - 1) / blockSize;
    limitGrid<<<numBlocks, blockSize>>>(grid.getValues() + rank * procSize, 0.99f * 1e11f, procSize);
}

CUDAStreamART::~CUDAStreamART()
{   
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;

    cudaFree(tp);
    cudaFree(tp_back);
}

void grid_project(const float *grid, float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream)
{
    float sinp = sin(phi);
    float cosp = cos(phi);
    float phi_adjusted = phi;
    while (phi_adjusted < 0) { phi_adjusted += 2 * M_PIf32; }
    while (phi_adjusted > M_PIf32/ 2) { phi_adjusted -= M_PIf32 / 2; }
    if (phi_adjusted > M_PIf32 / 4) { phi_adjusted = M_PIf32 / 2 - phi_adjusted; }
    
    float a = std::sin(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float b = std::cos(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float h = 1.0f / (a + b);
    float border = 0.5f * (nx - 1);

    // 将p_value初始化为0
    cudaMemset(project, 0, (nx + 4) * nz * sizeof(float));

    // 设置CUDA内核参数
    int threadsPerBlock = 1024;
    int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    // 启动CUDA内核
    grid_project_k<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(grid, project, nx, nz, a, b, h,
                                                                  cosp, sinp, start, end, border);
}

void grid_back_project(float *grid, const float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream)
{
    float sinp = sin(phi);
    float cosp = cos(phi);
    float phi_adjusted = phi;
    while (phi_adjusted < 0) { phi_adjusted += 2 * M_PIf32; }
    while (phi_adjusted > M_PIf32/ 2) { phi_adjusted -= M_PIf32 / 2; }
    if (phi_adjusted > M_PIf32 / 4) { phi_adjusted = M_PIf32 / 2 - phi_adjusted; }
    
    float a = std::sin(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float b = std::cos(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float h = 1.0f / (a + b);
    float border = 0.5f * (nx - 1);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;

    grid_back_project_k<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(grid, project, nx, nz, a, b, h,
                                                                       cosp, sinp, start, end, border);
}

void grid_max_map(float *grid, const float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream)
{
    float sinp = sin(phi);
    float cosp = cos(phi);
    float phi_adjusted = phi;
    while (phi_adjusted < 0) { phi_adjusted += 2 * M_PIf32; }
    while (phi_adjusted > M_PIf32/ 2) { phi_adjusted -= M_PIf32 / 2; }
    if (phi_adjusted > M_PIf32 / 4) { phi_adjusted = M_PIf32 / 2 - phi_adjusted; }
    
    float a = std::sin(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float b = std::cos(phi_adjusted + M_PIf32 / 4) / std::sqrt(2.0f);
    float h = 1.0f / (a + b);
    float border = 0.5f * (nx - 1);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (end - start + threadsPerBlock - 1) / threadsPerBlock;
    grid_max_map_k<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(grid, project, nx, nz, a, b, h,
                                                                  cosp, sinp, start, end, border);
}

void reprojectART(Grid &grid, float *projects, int totalAngles, int numStreams, float *angles, int iters, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int numAngles = totalAngles / size;
    int procSize = grid.getSize() / size;
    CUDAStreamART streamART(grid.getNx(), grid.getNy(), numStreams);

    // Synchronize all projections from all GPUs
    int imSize = grid.getNx() * grid.getNy();
    int localSize = numAngles * imSize;
    MPI_Allgather(MPI_IN_PLACE, localSize, MPI_FLOAT, projects, localSize, MPI_FLOAT, comm);

    for (int i = 0; i < totalAngles; i++) {
        streamART.maxMapGrid(grid, projects + i * imSize, *(angles + i), size, rank);
    }
    // Swap values for different parts of the grid
    MPI_Allgather(MPI_IN_PLACE, procSize, MPI_FLOAT, grid.getValues(), procSize, MPI_FLOAT, comm);

    Grid *map = new Grid(grid.getNx(), grid.getNy(), grid.getNz());
    float *tmpProjs;
    cudaMalloc(&tmpProjs, totalAngles * imSize * sizeof(float));

    for (int k = 0; k < iters; k++) {
        map->setZeros();
        
        for (int i = 0; i < numAngles; i++) {
            streamART.projectGrid(grid, tmpProjs + rank * localSize + i * imSize, *(angles + rank * numAngles + i));
        }

        int blockSize = 1024;
        int numBlocks = (localSize + blockSize - 1) / blockSize;
        updateProject<<<numBlocks, blockSize>>>(tmpProjs + rank * localSize, projects + rank * localSize, localSize);
        MPI_Allgather(MPI_IN_PLACE, localSize, MPI_FLOAT, tmpProjs, localSize, MPI_FLOAT, comm);
        
        for (int i = 0; i < totalAngles; i++) {
            streamART.backProjectGrid(*map, tmpProjs + i * imSize, *(angles + i), size, rank);
        }
        // Swap values for different parts of map
        MPI_Allgather(MPI_IN_PLACE, procSize, MPI_FLOAT, map->getValues(), procSize, MPI_FLOAT, comm);

        numBlocks = (grid.getSize() + blockSize - 1) / blockSize;
        updateGrid<<<numBlocks, blockSize>>>(grid.getValues(), map->getValues(), totalAngles, grid.getSize());
    }

    for (int i = 0; i < numAngles; i++) {
        streamART.projectGrid(grid, projects + rank * localSize + i * imSize, *(angles + rank * numAngles + i));
    }

    delete map;
    cudaFree(tmpProjs);
}

    FArray reconstruct_irp(const FArray &holograms, int totalAngles, int numImages, const IntArray &imSize,
                           const F2DArray &fresnelNumbers, MPI_Comm comm, int kmax, float epsilon, int numStreams)
    {
        if (fresnelNumbers.size() != numImages)
            throw std::invalid_argument("The number of images and fresnel numbers does not match!");

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        int numAngles = totalAngles / size;
        int projSize = imSize[0] * imSize[1];

        float *d_holograms, *projection, *squared_error;
        cudaMalloc((void**)&d_holograms, holograms.size() * sizeof(float));
        cudaMemcpy(d_holograms, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);

        cuFloatComplex *complexWave, *propedComplexWave;
        cudaMalloc((void**)&complexWave, numAngles * projSize * sizeof(cuFloatComplex));
        cudaMalloc((void**)&propedComplexWave, numAngles * numImages * projSize * sizeof(cuFloatComplex));

        int blockSize = 1024;
        int decGridSize = (holograms.size() + blockSize - 1) / blockSize;
        sqrtIntensity<<<decGridSize, blockSize>>>(d_holograms, holograms.size());
        setAmpPhase0<<<decGridSize, blockSize>>>(propedComplexWave, d_holograms, numAngles * numImages * projSize);
        
        // Create propagators
        PropagatorPtr propPtr = std::make_shared<Propagator>(imSize, fresnelNumbers, CUDAPropKernel::Fourier);

        Grid *grid = new Grid(imSize[0], imSize[1]);
        grid->setValues(1.0f);
        cudaMalloc((void**)&projection, totalAngles * projSize * sizeof(float));
        float *angles = new float[totalAngles];
        for (int i = 0; i < totalAngles; i++) {
            angles[i] = i * M_PIf32 / totalAngles;
        }

        // start the iterative reconstruction
        float error, total_error;
        float last_error = 0.0f;
        cudaMalloc((void**)&squared_error, numAngles * numImages * projSize * sizeof(float));
        std::cout << "Starting the iterative reconstruction..." << std::endl;

        int iteration = 1;
        int objGridSize = (numAngles * projSize + blockSize - 1) / blockSize;
        for (int n = 0; n < 1; n++) {
            // propagate the wave fields to object plane
            for (int i = 0; i < numAngles; i++) {
                propPtr->backPropagate(propedComplexWave + i * numImages * projSize, complexWave + i * projSize);
            }
            scaleComplexData<<<objGridSize, blockSize>>>(complexWave, numAngles * projSize, 1.0f / numImages);

            // delta part of the objects' index of refraction
            computeLogAbs<<<objGridSize, blockSize>>>(projection + rank * numAngles * projSize, complexWave, numAngles * projSize);
            std::cout << "\t" << n << ": ART (1000 iterations)" << std::endl;
            reprojectART(*grid, projection, totalAngles, numStreams, angles, 1000, comm);
            // set amplitude to the wave fields
            scaleExpData<<<objGridSize, blockSize>>>(projection + rank * numAngles * projSize, numAngles * projSize, -1.0f);
            setAmplitude<<<objGridSize, blockSize>>>(complexWave, projection + rank * numAngles * projSize, numAngles * projSize);

            // beta part of the objects' index of refraction
            computePhase<<<objGridSize, blockSize>>>(complexWave, projection + rank * numAngles * projSize, numAngles * projSize);
            absData<<<objGridSize, blockSize>>>(projection + rank * numAngles * projSize, numAngles * projSize);
            std::cout << "\t" << n << ": ART (1000 iterations)" << std::endl;
            reprojectART(*grid, projection, totalAngles, numStreams, angles, 1000, comm);
            // set phase to the wave fields
            scaleFloatData<<<objGridSize, blockSize>>>(projection + rank * numAngles * projSize, numAngles * projSize, -1.0f);
            setPhaseAmp1<<<objGridSize, blockSize>>>(complexWave, projection + rank * numAngles * projSize, numAngles * projSize);

            // propagate the wave fields to detector plane
            for (int i = 0; i < numAngles; i++) {
                propPtr->propagate(complexWave + i * projSize, propedComplexWave + i * numImages * projSize);
            }

            // calculate the error between the measured and propagated holograms
            computeSquError<<<decGridSize, blockSize>>>(squared_error, propedComplexWave, 
                                                        d_holograms, numAngles * numImages * projSize);
            thrust::device_ptr<float> thrustPtr(squared_error);
            error = thrust::reduce(thrustPtr, thrustPtr + numAngles * numImages * projSize, 0.0f, thrust::plus<float>());

            total_error = 0.0f;
            // 将所有进程的error值累加到rank 0
            MPI_Reduce(&error, &total_error, 1, MPI_FLOAT, MPI_SUM, 0, comm);
            if (rank == 0) {
                error = std::sqrt(total_error / (totalAngles * numImages * projSize));
                std::cout << "\t" << n << ": Error: " << error << " (Delta Error: " << last_error - error << ")" << std::endl;
                if (n != 0 && last_error - error < epsilon) {
                    iteration *= 2;
                }
                last_error = error;
            }
            // 将rank 0的iteration广播到所有进程
            MPI_Bcast(&iteration, 1, MPI_INT, 0, comm);

            // magnitude constraint
            setAmplitude<<<decGridSize, blockSize>>>(propedComplexWave, d_holograms, numAngles * numImages * projSize);
        }

        FArray result(grid->getSize());
        cudaMemcpy(result.data(), grid->getValues(), grid->getSize() * sizeof(float), cudaMemcpyDeviceToHost);

        delete[] angles; delete grid;
        cudaFree(complexWave); cudaFree(projection);
        cudaFree(squared_error); cudaFree(propedComplexWave); cudaFree(d_holograms);

        return result;
    }