/*Raul P. Pelaez 2022. The UAMMD letters made of particles suspended in a fluid.
Runs a Brownian Hydrodynamics simulation with particles starting in a periodic box at low temperature.

You can visualize the reuslts with superpunto
*/

//This include contains the basic needs for an uammd project
#include"uammd.cuh"
#include"Integrator/BDHI/BDHI_FCM.cuh"
#include<fstream>

using namespace uammd;

struct GravityAndWall : public Interactor {
  real zwall;

  GravityAndWall(std::shared_ptr<ParticleData> pd, real zwall):
    Interactor(pd, "GravityAndWall"), zwall(zwall){}
  
  void sum(Interactor::Computables comp, cudaStream_t st) override{
    // bool shouldComputeForces = comp.force;
    // bool shouldComputeEnergies = comp.energy;
    // bool shouldComputeVirials = comp.virial;    
    auto pos = pd->getPos(access::gpu, access::read);
    auto force = pd->getForce(access::gpu, access::readwrite);
    real gravity = 0.1;
    real wallStrength = 1.0;
    real h = this->zwall;
    thrust::for_each_n(thrust::device,
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

auto readParticles(){
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

auto initializeSimulation(std::shared_ptr<ParticleData> pd, Box box){
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


void runSimulation(std::shared_ptr<ParticleData> pd, std::shared_ptr<Integrator> bdhi){
  std::ofstream out("/dev/stdout");
  Timer tim;
  tim.tic();
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

int main(int argc, char *argv[]){
  Box box({256, 128, 160});
  auto pd = readParticles();
  auto bdhi = initializeSimulation(pd, box);
  auto gravity = std::make_shared<GravityAndWall>(pd, -box.boxSize.z*0.5);
  bdhi->addInteractor(gravity);
  runSimulation(pd, bdhi);  
  return 0;
}
