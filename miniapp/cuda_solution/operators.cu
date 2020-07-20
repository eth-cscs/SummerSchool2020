//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "cuda_helpers.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

// POD type holding information for device
struct DiffusionParams {
    int nx;
    int ny;
    double alpha;
    double dxs;
    double *x_old;
    double *bndN;
    double *bndE;
    double *bndS;
    double *bndW;
};

// params will be a global variable visible to cuda kernels.
// The setup_params_on_device function initialises a copy on
// the host, then copies it to the device variable.
__device__
DiffusionParams params;

void setup_params_on_device(int nx, int ny, double alpha, double dxs)
{
    auto p = DiffusionParams {
        nx,
        ny,
        alpha,
        dxs,
        data::x_old.device_data(),
        data::bndN.device_data(),
        data::bndE.device_data(),
        data::bndS.device_data(),
        data::bndW.device_data()
    };

    cuda_check_status(
        cudaMemcpyToSymbol(params, &p, sizeof(DiffusionParams))
    );
}

namespace kernels {
    __global__
    void stencil_shared(double* S, const double *U) {
        double extern __shared__ buffer[];

        auto nx = params.nx;
        auto ny = params.ny;
        auto bx = blockDim.x+2;
        auto by = blockDim.y+2;

        auto gi = threadIdx.x + blockDim.x*blockIdx.x;
        auto gj = threadIdx.y + blockDim.y*blockIdx.y;
        auto li = threadIdx.x + 1;
        auto lj = threadIdx.y + 1;
        auto gpos = gi + gj * nx;
        auto lpos = li + lj * bx;

        if(gi<nx && gj<ny) {
            // load the shared memory
            if(li==1) {     //  west boundary
                if(gi==0)
                    buffer[lpos-1] = params.bndW[gj];
                else
                    buffer[lpos-1] = U[gpos-1];
            }
            if(li==bx-2) {  //  east boundary
                if(gi==nx-1)
                    buffer[lpos+1] = params.bndE[gj];
                else
                    buffer[lpos+1] = U[gpos+1];
            }
            if(lj==1) {     //  south boundary
                if(gj==0)
                    buffer[lpos-bx] = params.bndS[gi];
                else
                    buffer[lpos-bx] = U[gpos-nx];
            }
            if(lj==by-2) {  //  south boundary
                if(gj==ny-1)
                    buffer[lpos+bx] = params.bndN[gi];
                else
                    buffer[lpos+bx] = U[gpos+nx];
            }
            buffer[lpos] = U[gpos];

            __syncthreads();

            S[gpos] = -(4. + params.alpha) * buffer[lpos]               // central point
                                   + buffer[lpos-1]  + buffer[lpos+1]   // east and west
                                   + buffer[lpos-bx] + buffer[lpos+bx]  // north and south
                                   + params.alpha * params.x_old[gpos]
                                   + params.dxs * buffer[lpos] * (1.0 - buffer[lpos]);
        }
    }
    __global__
    void stencil_interior(double* S, const double *U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x + 1;
        auto j = threadIdx.y + blockDim.y*blockIdx.y + 1;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;

        auto pos = i + nx*j;

        if (i<nx-1 && j<ny-1) {
            S[pos] = -(4. + alpha) * U[pos]                 // central point
                                    + U[pos-1]  + U[pos+1]  // east and west
                                    + U[pos-nx] + U[pos+nx] // north and south
                                    + alpha * params.x_old[pos]
                                    + params.dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_east_west(double* S, const double *U) {
        auto j = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        if(j>0 && j<ny-1) {
            // EAST : i = nx-1
            auto pos = find_pos(nx-1, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos-nx] + U[pos+nx]
                        + alpha*params.x_old[pos] + params.bndE[j]
                        + dxs * U[pos] * (1.0 - U[pos]);

            // WEST : i = 0
            pos = find_pos(0, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos+1] + U[pos-nx] + U[pos+nx]
                        + alpha*params.x_old[pos] + params.bndW[j]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_north_south(double* S, const double *U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        if(i>0 && i<nx-1) {
            // NORTH : j = ny -1
            auto pos = i + nx*(ny-1);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos-nx]
                        + alpha*params.x_old[pos] + params.bndN[i]
                        + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH : j = 0
            pos = i;
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos+nx]
                        + alpha*params.x_old[pos] + params.bndS[i]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_corners(double* S, const double* U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        // only 1 thread executes this kernel
        if(i==0) {
            // NORTH-EAST
            auto pos = find_pos(nx-1, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[ny-1] // east and west
                                   + U[pos-nx] + params.bndN[nx-1] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-EAST
            pos = find_pos(nx-1, 0);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[0]      // east and west
                                   + params.bndS[nx-1]+ U[pos+nx]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-WEST
            pos = find_pos(0, 0);
            S[pos] = -(4. + alpha) * U[pos]                // central point
                                   + params.bndW[0] + U[pos+1]    // east and west
                                   + params.bndS[0] + U[pos+nx] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // NORTH-WEST
            pos = find_pos(0, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                 // central point
                                   + params.bndW[nx-1]+ U[pos+1] // east and west
                                   + U[pos-nx] + params.bndN[0]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);
        }
    }
} // namespace kernels


#define USE_SHARED_KERNEL

void diffusion(data::Field const& U, data::Field &S)
{
    using data::options;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;

    // calculates the linear index into an array of width nx
    // from an (i,j) coordinate pair
    auto idx = [&nx] (size_t i, size_t j) {
        return i + j * nx;
    };

    static bool is_initialized = false;
    if(!is_initialized) {
        setup_params_on_device(nx, ny, alpha, dxs);
        is_initialized = true;
    }

    // apply stencil to the interior grid points
    // TODO: what is the purpose of the following?
    auto calculate_grid_dim = [] (size_t n, size_t block_dim) {
        return (n+block_dim-1)/block_dim;
    };

    // TODO: apply stencil to the interior grid points
    dim3 block_dim(16, 16);
    dim3 grid_dim(
            calculate_grid_dim(nx, block_dim.x),
            calculate_grid_dim(ny, block_dim.y));
    kernels::stencil_interior<<<grid_dim, block_dim>>>(S.device_data(), U.device_data());

#ifdef USE_SHARED_KERNEL
    kernels::stencil_shared<<<grid_dim, block_dim, (block_dim.x+2)*(block_dim.y+2)*sizeof(double)>>>(S.device_data(), U.device_data());
#else
    // apply stencil at east-west boundary
    auto bnd_grid_dim_y = calculate_grid_dim(ny, 64);
    kernels::stencil_east_west<<<bnd_grid_dim_y, 64>>>(S.device_data(), U.device_data());

    // apply stencil at north-south boundary
    auto bnd_grid_dim_x = calculate_grid_dim(nx, 64);
    kernels::stencil_north_south<<<bnd_grid_dim_x, 64>>>(S.device_data(), U.device_data());

    // apply stencil at corners
    kernels::stencil_corners<<<1, 1>>>(S.device_data(), U.device_data());
#endif
}
} // namespace operators
