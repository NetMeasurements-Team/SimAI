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

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "entry.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <atomic>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;

extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;
extern std::atomic<bool> waiting_sim_finish;

struct sim_event {
  void *buffer;
  uint64_t count;
  int type;
  int dst;
  int tag;
  string fnType;
};

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;

public:
  queue<sim_event> sim_event_queue;
  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;
  }
  ~ASTRASimNetwork() override {}
  int sim_comm_size(AstraSim::sim_comm comm, int *size) override { return 0; }
  int sim_finish() override {
    for (auto it = nodeHash.begin(); it != nodeHash.end(); ++it) {
      pair<int, int> p = it->first;
      if (p.second == 0) {
        std::cout << "sim_finish on sent, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data sent from node " << p.first << " is " << it->second
             << "\n";
      } else {
        std::cout << "sim_finish on received, " << " Thread id: " << pthread_self() << std::endl;
        cout << "All data received by node " << p.first << " is " << it->second
             << "\n";
      }
    }
    waiting_sim_finish = true;
    std::cout << "Waiting for any pending message..." << std::endl;
    Simulator::Schedule(Time(0), &check_sim_finish);
    return 0;
  }
  double sim_time_resolution() override { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI *MEM) override { return 0; }
  AstraSim::timespec_t sim_get_time() override {
    AstraSim::timespec_t timeSpec;
    timeSpec.time_val = Simulator::Now().GetNanoSeconds();
    return timeSpec;
  }
  void sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) override {
    Simulator::Schedule(NanoSeconds(delta.time_val), fun_ptr, fun_arg);
  }
  int sim_send(
      void *buffer,
      const uint64_t message_size,
      int type,
      int dst,
      /**
       * This field is used to uniquely identify a flow at the network frontend layer. Once a flow is received, the tag is
       * used to associate it with the correct event handler, and this is triggered with its arguments (which include the
       * flow id).
       * TODO: there is no reason for not using directly the flow_id as tag instead of creating a new value.
       */
      int tag,
      [[maybe_unused]] AstraSim::sim_request* request,
      void (*msg_handler)(void *fun_arg),
      void *fun_arg) override {
    dst += npu_offset;
    const auto ehd = static_cast<AstraSim::SendPacketEventHandlerData*>(fun_arg);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "[Send event registration] dst %d sim_send on rank %d tag %u channel id %d (flow_id %u)",
        dst, rank, tag, ehd->channel_id, ehd->flow_id);
    send_flow(rank, dst, message_size, msg_handler, fun_arg, tag, ehd->flow_id, ehd->nvls_on);
    return 0;
  }

  int sim_recv(
      void* buffer,
      const uint64_t message_size,
      int type,
      int src,
      int tag,
      [[maybe_unused]] AstraSim::sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg) override {
    // TODO move this code into entry.h in a new function recv_flow (for symmetry with sim_send -> send_flow)
    #ifdef NS3_MTP
    MtpInterface::ExplicitCriticalSection ecs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    src += npu_offset;
    auto recv_event = MsgEvent(src, rank, 1, message_size, msg_handler, fun_arg);
    const auto ehd = static_cast<AstraSim::RecvPacketEventHadndlerData*>(recv_event.fun_arg);
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "[Receive event registration] src %d sim_recv on rank %d tag %u channel id %d (flow_id %u)",
        src, rank, tag, ehd->channel_id, ehd->flow_id);

    if (recvHash.find(MsgEventKey{tag, {recv_event.src, recv_event.dst}}) != recvHash.end()) {
      // 1) ns3 has already received some message before sim_recv is called.
      const uint64_t already_received_size = recvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}];
      if (already_received_size == message_size) {
        // 1.1) The received message size is the same as what we expect. Exit.
        recvHash.erase(MsgEventKey{tag, {recv_event.src, recv_event.dst}});
        #ifdef NS3_MTP
        ecs.ExitSection();
        #endif
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message arrived early, skip registering] recvHash already had the expected bytes for src %d, dst %d,"
            " tag %u; directly invoke handler: t.count %llu, tag %u, flow_id %d",
            recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
        recv_event.callHandler();
        goto sim_recv_end_section;
      } else if (already_received_size > message_size) {
        // 1.2) The node received more than expected. Do trigger the callback handler for this message,
        //      for the Sys layer to call sim_recv for more messages.
        //      but also wait for the Sys layer to call sim_recv for more messages.
        recvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}] = already_received_size - message_size;
        #ifdef NS3_MTP
        ecs.ExitSection();
        #endif
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message arrived early (more left), skip registering] recvHash had more bytes (%u) than expected for "
            "src %d, dst %d, tag %u, directly invoke handler for them: t.count %llu, tag %u, flow_id %d",
            already_received_size, recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
        recv_event.callHandler();
        goto sim_recv_end_section;
      } else {
        // 1.3) The node received less than what we expected.
        //      Reduce the number of bytes we are waiting to receive and store the callback.
        recvHash.erase(MsgEventKey{tag, {recv_event.src, recv_event.dst}});
        recv_event.remaining_bytes -= already_received_size;
        expeRecvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}] = recv_event;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message arrived early (not enough), registering partial] recvHash had less bytes (%u) than expected"
            " for src %d, dest %d, tag %u; register the difference in expeRecvHash: t.count: %llu, flow_id %d",
            recv_event.src, recv_event.dst, tag, recv_event.remaining_bytes, ehd->flow_id);
      }
    } else {
      // 2) ns3 has not yet received anything.
      if (expeRecvHash.find(MsgEventKey{tag, {recv_event.src, recv_event.dst}}) == expeRecvHash.end()) {
        // 2.1) We have not been expecting anything so far.
        expeRecvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}] = recv_event;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message not arrived yet, registering] recvHash had no entry for src %d, dst %d, tag %u; register the"
            " message in expeRecvHash: t.count %llu, flow_id %d",
            recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
      } else {
        // 2.2) We have already been expecting something. Increment the number of bytes we are waiting to receive.
        const uint64_t c = expeRecvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}].remaining_bytes;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message not arrived yet, re-registering] recvHash had no entry for src %d, dst %d, tag %u, but we "
            "were already waiting %u bytes for it; updating the entry in expeRecvHash with %u additional bytes: "
            "flow_id %d", recv_event.src, recv_event.dst, tag, c, message_size, ehd->flow_id);
        expeRecvHash[MsgEventKey{tag, {recv_event.src, recv_event.dst}}].remaining_bytes += message_size;
      }
    }
    #ifdef NS3_MTP
    ecs.ExitSection();
    #endif

sim_recv_end_section:
    return 0;
  }
};

struct user_param {
  int thread;
  string workload;
  string network_topo;
  string network_conf;
  string run_name;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
    run_name = "";
  };
  ~user_param(){};
};

static int user_param_prase(const int argc, char* argv[], user_param* user_param) {
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:g:s:n:r:c:"))!=-1) {
    switch (opt) {
    case 'h':
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-c    network conf"<<std::endl;
      std::cout<<"-r    run name"<<std::endl;
      return 1;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'c':
      user_param->network_conf = optarg;
      break;
    case 'r':
      user_param->run_name = optarg;
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }
  return 0 ;
}

int main(int argc, char* argv[]) {
  user_param user_param;
  if(user_param_prase(argc, argv, &user_param)) {
    return 0;
  }
  MockNcclLog::set_log_name("SimAI" + (user_param.run_name.empty() ? "" : "." + user_param.run_name) +".log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");
  #ifdef NS3_MTP
  MtpInterface::Enable(user_param.thread);
  if (user_param.thread == 1) {
    GlobalValue::Bind("PartitionSchedulingMethod", StringValue ("ByPendingEventCount"));
  }
  #endif

  setup_ns3_simulation(user_param.network_topo, user_param.network_conf, user_param.run_name);
  int nodes_num = node_num - switch_num;
  int gpu_num = node_num - nvswitch_num - switch_num;

  std::map<int, int> node2nvswitch; 
  for (int i = 0; i < gpu_num; ++ i) {
    node2nvswitch[i] = gpu_num + i / gpus_per_server;
  }
  for (int i = gpu_num; i < gpu_num + nvswitch_num; ++ i) {
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  } 

  LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

  std::vector<ASTRASimNetwork *> networks(nodes_num, nullptr);
  std::vector<AstraSim::Sys *> systems(nodes_num, nullptr);

  for (int j = 0; j < nodes_num; j++) {
    networks[j] = new ASTRASimNetwork(j ,0);
    systems[j] = new AstraSim::Sys(
        networks[j],
        nullptr,
        j,
        0,
        1,
        {nodes_num},
        {1},
        "",
        user_param.workload,
        1,
        1,
        1,
        1,
        0,
        RESULT_PATH,
        user_param.run_name,
        true,
        false,
        gpu_type,
        {gpu_num},
        NVswitchs,
        gpus_per_server
    );
    systems[j]->nvswitch_id = node2nvswitch[j];
    systems[j]->num_gpus = nodes_num - nvswitch_num;
  }
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  std::cout << "simulator run " << std::endl;
  Simulator::Stop(Seconds(2000000000));
  Simulator::Run();
  Simulator::Destroy();
  
  #ifdef NS3_MPI
  MpiInterface::Disable();
  #endif
  return 0;
}
