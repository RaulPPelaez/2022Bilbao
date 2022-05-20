/*Raul P. Pelaez 2022. Particle-grid coupling example
 */
#include<uammd.cuh>
#include <misc/IBM.cuh>
#include"utils.cuh"

using namespace uammd;

// A simple Gaussian kernel compatible with the IBM module.
//A lot of them are already defined, i.e IBM_kernels::Peskin::threePoint
class Gaussian{
  const real prefactor;
  const real tau;
public:
  const int support;
  Gaussian(real width, int support):
    prefactor(pow(2.0*M_PI*width*width, -0.5)),
    tau(-0.5/(width*width)),
    support(support){}

  __device__ int3 getSupport(real3 pos, int3 cell){
    return {support, support, support};
  }

  __device__ real phi(real r, real3 pos) const{
    return prefactor*exp(tau*r*r);
  }
};


int main(){
  real L = 32;
  Box box({L,L,L});
  int3 cellDim {32,32,32};
  Grid grid(box, cellDim);

  int numberParticles = 1e6;
  int ncells = grid.getNumberCells();
  
  thrust::device_vector<real3> positions = generateRandomPositions(L, numberParticles);
  thrust::device_vector<real> particleQuantity(ncells), gridQuantity(ncells);
  thrust::fill(particleQuantity.begin(), particleQuantity.end(), 1);
  thrust::fill(gridQuantity.begin(), gridQuantity.end(), 0);

  const real width = 1; //An arbitrary width
  const int support = 8;//An arbitrary support
  auto kernel = std::make_shared<Gaussian>(width, support);

  IBM<Gaussian> ibm(kernel, grid);
  ibm.spread(thrust::raw_pointer_cast(positions.data()),
	     thrust::raw_pointer_cast(particleQuantity.data()),
	     thrust::raw_pointer_cast(gridQuantity.data()),
	     numberParticles);

  thrust::fill(particleQuantity.begin(), particleQuantity.end(), 0);

  ibm.gather(thrust::raw_pointer_cast(positions.data()),
	     thrust::raw_pointer_cast(particleQuantity.data()),
	     thrust::raw_pointer_cast(gridQuantity.data()),
	     numberParticles);
  //Now particleQuantity is filled with ones again
  return 0;
}
