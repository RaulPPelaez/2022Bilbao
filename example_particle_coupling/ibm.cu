/*Raul P. Pelaez 2022. Particle-grid coupling example
 */
#include<uammd.cuh>
#include <misc/IBM.cuh>
#include<thrust/random.h>
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

template<class Iter1, class Iter2>
void spreadWithIBM(Grid grid,
		   Iter1 positions,
                   Iter2 dataAtCellPositions,
                   Iter2 dataAtParticlePositions,
                   int numberParticles){
  const real width = 1; //An arbitrary width
  const int support = 8;//An arbitrary support
  auto kernel = std::make_shared<Gaussian>(width, support);
  IBM<Gaussian> ibm(kernel, grid);
  //Spreads dataAtParticlePositions into dataAtCellPositions
  ibm.spread(positions, dataAtParticlePositions, dataAtCellPositions, numberParticles);
}

template<class Iter1, class Iter2>
void interpolateWithIBM(Grid grid, Iter1 positions,
                        Iter2 dataAtCellPositions,
                        Iter2 dataAtParticlePositions,
                        int numberParticles){
  const real width = 1; //An arbitrary width
  const int support = 8;//An arbitrary support
  auto kernel = std::make_shared<Gaussian>(width, support);
  IBM<Gaussian> ibm(kernel, grid);
  //Interpolates dataAtCellPositions into dataAtParticlePositions
  ibm.gather(positions, dataAtParticlePositions, dataAtCellPositions, numberParticles);
}

//Creates and returns a vector with random positions inside a cubic box of side L (always the same random positions) 
thrust::device_vector<real3> generateRandomPositions(real L, int numberParticles){
  thrust::device_vector<real3> positions(numberParticles);
  auto it = thrust::make_counting_iterator<int>(0);
  thrust::transform(it, it+numberParticles,
		    positions.begin(),
		    [=]__device__(int i){
		      thrust::default_random_engine rng;
		      thrust::uniform_real_distribution<real> dist(-L*0.5, L*0.5);
		      rng.discard(i);
		      return make_real3(dist(rng), dist(rng), dist(rng));
		    }
		    );
  return positions;
}

int main(){
  real L = 32;
  Box box({L,L,L});
  int3 cellDim {32,32,32};
  Grid grid(box, cellDim);

  int numberParticles = 1e6;
  thrust::device_vector<real3> positions;
  positions = generateRandomPositions(L, numberParticles);
  
  
  int ncells = grid.getNumberCells();
  thrust::device_vector<real> particleQuantity(ncells), gridQuantity(ncells);
  thrust::fill(particleQuantity.begin(), particleQuantity.end(), 1);
  thrust::fill(gridQuantity.begin(), gridQuantity.end(), 0);
  
  spreadWithIBM(grid,
		thrust::raw_pointer_cast(positions.data()),
		thrust::raw_pointer_cast(gridQuantity.data()),
		thrust::raw_pointer_cast(particleQuantity.data()),
		numberParticles);
  
  thrust::fill(particleQuantity.begin(), particleQuantity.end(), 0);  
  interpolateWithIBM(grid,
		thrust::raw_pointer_cast(positions.data()),
		thrust::raw_pointer_cast(gridQuantity.data()),
		thrust::raw_pointer_cast(particleQuantity.data()),
		numberParticles);
  //Now particleQuantity is filled with ones again
  return 0;
}
