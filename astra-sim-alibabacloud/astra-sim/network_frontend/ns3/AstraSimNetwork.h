/*
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#ifndef __ASTRASIMNETWORK_HH__
#define __ASTRASIMNETWORK_HH__

#include "astra-sim/system/AstraNetworkAPI.hh"

using namespace std;
class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {

  int npu_offset;

public:
  ASTRASimNetwork(int rank, int npu_offset);
  ~ASTRASimNetwork() override;
  int sim_comm_size(AstraSim::sim_comm comm, int* size) override;
  int sim_finish() override;
  double sim_time_resolution() override;
  int sim_init(AstraSim::AstraMemoryAPI* MEM) override;
  AstraSim::timespec_t sim_get_time() override;
  void sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) override;
  int sim_send(
      void* buffer,
      uint64_t message_size,
      int type,
      int dst,
      int tag,
      AstraSim::sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg) override;
  int sim_recv(
      void* buffer,
      uint64_t message_size,
      int type,
      int src,
      int tag,
      AstraSim::sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg) override;
};
#endif
