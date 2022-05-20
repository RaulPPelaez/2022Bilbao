#include <uammd.cuh>
#include <thrust/device_vector.h>
#include <thrust/random.h>

//Creates and returns a vector with random positions inside a cubic box of side L (always the same random positions) 
thrust::device_vector<uammd::real3> generateRandomPositions(uammd::real L, int numberParticles){
  thrust::device_vector<uammd::real3> positions(numberParticles);
  auto it = thrust::make_counting_iterator<int>(0);
  thrust::transform(it, it+numberParticles,
		    positions.begin(),
		    [=]__device__(int i){
		      thrust::default_random_engine rng;
		      thrust::uniform_real_distribution<uammd::real> dist(-L*0.5, L*0.5);
		      rng.discard(i);
		      return uammd::make_real3(dist(rng), dist(rng), dist(rng));
		    }
		    );
  return positions;
}
