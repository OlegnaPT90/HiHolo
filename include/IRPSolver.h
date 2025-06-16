#ifndef IRPSOLVER_H
#define IRPSOLVER_H

#include "Propagator.h"
#include <mpi.h>

class Grid
{
    private:
        int nx, ny, nz;
        float *values;
        int size;
        int blockSize;
        int gridSize;
    public:
        Grid(int ix, int iy, int iz);
        Grid(int ix, int iz);
        ~Grid();
        void setValues(float value);
        void scaleValues(float scale);
        void setZeros();
        int getSize() const {return size;}
        float *getValues() const {return values;}
        int getNx() const {return nx;}
        int getNy() const {return ny;}
        int getNz() const {return nz;}
};

class CUDAStreamART
{
    private:
        cudaStream_t *streams;
        float *tp;
        float *tp_back;
        int nx, ny;
        int numStreams;

    public:
        CUDAStreamART(int ix, int iy, int N);
        void syncStreams();
        void merge(float *project);
        void projectGrid(const Grid &grid, float *project, float phi);
        void backProjectGrid(Grid &grid, const float *project, float phi, int size, int rank);
        void maxMapGrid(Grid &grid, const float *project, float phi, int size, int rank);
        ~CUDAStreamART();
};

void grid_project(const float *grid, float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream = 0);
void grid_back_project(float *grid, const float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream = 0);
void grid_max_map(float *grid, const float *project, int nx, int nz, float phi, int start, int end, cudaStream_t stream = 0);
void reprojectART(Grid &grid, float *projects, int totalAngles, int numStreams, float *angles, int iters, MPI_Comm comm);

FArray reconstruct_irp(const FArray &holograms, int totalAngles, int numImages, const IntArray &imSize,
        const F2DArray &fresnelNumbers, MPI_Comm comm, int kmax = 2, float epsilon = 1e-5f, int numStreams = 1);

#endif