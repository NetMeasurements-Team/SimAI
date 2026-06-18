#ifndef __ASTRASIM_UB_NETWORK_HH__
#define __ASTRASIM_UB_NETWORK_HH__

#include <iostream>
#include <queue>
#include "astra-sim/system/AstraNetworkAPI.hh"

using namespace std;

class ASTRASimNetwork final : public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;

public:
  explicit ASTRASimNetwork(const int rank, int npu_offset) : AstraNetworkAPI(rank), npu_offset(npu_offset) {}
  ~ASTRASimNetwork() override {}
  
  int sim_comm_size(AstraSim::sim_comm comm, int* size) override { return 0; }
  int sim_finish() override;
  double sim_time_resolution() override { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI* MEM) override { return 0; }
  AstraSim::timespec_t sim_get_time() override;
  void sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) override;
  
  int sim_send(void* buffer, 
              uint64_t count, 
              int type, 
              int dst, 
              int tag,
              AstraSim::sim_request* request, 
              void (*msg_handler)(void* fun_arg), 
              void* fun_arg) override;
               
  int sim_recv(void* buffer, 
              uint64_t count, 
              int type, 
              int src, 
              int tag,
              AstraSim::sim_request* request, 
              void (*msg_handler)(void* fun_arg), 
              void* fun_arg) override;
};
#endif