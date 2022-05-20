/*Raul P. Pelaez 2022. The UAMMD letters made of particles suspended in a fluid.
Runs a Brownian Hydrodynamics simulation with particles starting in a periodic box at low temperature.

You can visualize the reuslts with superpunto
*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
#include<fstream>
#include <memory>

using namespace uammd;

// A simple Interactor that sums a gravitational force to each particle.
// It also adds a potential wall at the bottom of the domain.
struct GravityAndWall : public Interactor {
  real zwall;

  GravityAndWall(std::shared_ptr<ParticleData> pd, real zwall):
    Interactor(pd, "GravityAndWall"), zwall(zwall){}
  
  void sum(Interactor::Computables comp, cudaStream_t st) override{
    //This function can be called with different responsabilities.
    //Let us ignore that and compute only forces.
    // bool shouldComputeForces = comp.force;
    // bool shouldComputeEnergies = comp.energy;
    // bool shouldComputeVirials = comp.virial;
    auto pos = pd->getPos(access::gpu, access::read);
    auto force = pd->getForce(access::gpu, access::readwrite);
    real gravity = 0.1;
    real wallStrength = 1.0;
    real h = this->zwall;
    //This thrust call runs a for loop in the GPU. Iterates over all particles
    thrust::for_each_n(thrust::cuda::par.on(st),
		       thrust::make_counting_iterator<int>(0), pos.size(),
     		       [=]__device__(int i){
			real fz = 0;
			real pi_z = pos[i].z;
			if(pi_z<=h){
			   real distanceToWall = fabs(pi_z-h);
			   fz += wallStrength*distanceToWall*distanceToWall;
			}
			force[i].z += fz-gravity;
     		       });
  }

};

// Reads a file with positions into an UAMMD particle container and returns it
auto createParticles(){
  //Read contents of the file into a vector
  std::ifstream in ("pos.init");
  std::istream_iterator<real4> begin(in), end;
  std::vector<real4> h_pos(begin, end);
  int numberParticles = h_pos.size();
  //Create an UAMMD particle container
  auto pd = std::make_shared<ParticleData>(numberParticles);
  //Copy the file contents into the UAMMD positions (in the CPU)
  {
    auto pos = pd->getPos(access::cpu, access::write);
    std::copy(h_pos.begin(), h_pos.end(), pos.begin());
  }
  //Increase spatial coherence in memory
  pd->sortParticles();
  return pd;
}

// Initializes and returns a Force Coupling Methor Integrator module
auto createIntegrator(std::shared_ptr<ParticleData> pd, Box box){
  using Scheme = BDHI::FCMIntegrator;
  Scheme::Parameters par;
  par.temperature = 1;
  par.viscosity = 1;
  par.hydrodynamicRadius =  4;
  par.dt = 0.01;
  par.box = box;
  auto integrator = std::make_shared<Scheme>(pd, par);
  return integrator;
}

// Forwards the simulation and prints particles every once in a while
// This function works for any Integrator module
void runSimulation(std::shared_ptr<ParticleData> pd, std::shared_ptr<Integrator> bdhi){
  std::ofstream out("/dev/stdout");
  Timer tim; tim.tic();
  int numberSteps = 2000;
  int printSteps  = 200;
  forj(0, numberSteps){
    if(j%printSteps==0){
      auto pos = pd->getPos(access::cpu, access::read);
      out<<"#"<<std::endl;
      for(auto p: pos)out<<make_real3(p)<<" 0.7 0\n";
    }
    bdhi->forwardTime();
  }
  auto totalTime = tim.toc();
  System::log<System::MESSAGE>("mean FPS: %.2f", numberSteps/totalTime);
}

// Lets create a simulation and run it.
// We need a particle container and an Integrator. We will also add an
// Interactor to the Integrator.
//
// ParticleData
//      ^
//     / \
//    /   \
//   / 	Interactor: GravityAndWall
//  /     /
// /     v addInteractor()
//Integrator: FCM Hydrodynamics
int main(int argc, char *argv[]){
  {
    Box box({256, 128, 160});
    auto pd = createParticles(); //Particle container
    auto bdhi = createIntegrator(pd, box); //Integrator
    auto gravity = std::make_shared<GravityAndWall>(pd, -box.boxSize.z*0.5); //Interactor
    bdhi->addInteractor(gravity);
    runSimulation(pd, bdhi);
  }
  return 0;
}


//bdhi->addInteractor(createTPPoissonInteractor(pd));
//#include <Interactor/SpectralEwaldPoisson.cuh>
// auto createTPPoissonInteractor(std::shared_ptr<ParticleData> pd){
//   {
//     auto charges = pd->getCharge(access::cpu, access::write);
//     std::fill(charges.begin(), charges.end(), 1);
//   }
//   Poisson::Parameters par;
//   par.box = Box({256, 128, 160});
//   //Permittivity
//   par.epsilon = 1;
//   //Gaussian width of the sources
//   par.gw = 4.0;
//   //Overall tolerance of the algorithm
//   par.tolerance = 1e-2;
//   //If a splitting parameter is passed
//   // the code will run in Ewald split mode
//   //Otherwise, the non Ewald version will be used
//   //par.split = 1.0;
//   return std::make_shared<Poisson>(pd, par);
// }
