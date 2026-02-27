#include "AstraSimNetwork.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "entry.h"
#include "ns3/core-module.h"
#include <iostream>

using namespace std;
using namespace ns3;

int ASTRASimUbNetwork::sim_finish() {
  std::cout << "Waiting for any pending UB message..." << std::endl;
  Simulator::Schedule(Time(0), &check_sim_finish); // Wait for UB events to drain
  return 0;
}

AstraSim::timespec_t ASTRASimUbNetwork::sim_get_time() {
  AstraSim::timespec_t timeSpec;
  timeSpec.time_val = Simulator::Now().GetNanoSeconds(); // Same as original implementation
  return timeSpec;
}

void ASTRASimUbNetwork::sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void* fun_arg), void* fun_arg) {
  // Schedule a generic event in ns-3
  Simulator::ScheduleWithContext(rank, NanoSeconds(delta.time_val), fun_ptr, fun_arg);
}

int ASTRASimUbNetwork::sim_send(void* buffer, uint64_t count, int type, int dst, int tag,
                                AstraSim::sim_request* request, void (*msg_handler)(void* fun_arg), void* fun_arg) {
  dst += npu_offset;

  task1 t;
  t.src = rank;
  t.dest = dst;
  t.count = count;
  t.type = 0;
  t.fun_arg = fun_arg;
  t.msg_handler = msg_handler;

  // Store task to track completion
  sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;

  // Trigger the flow in the Unified Bus backend
  // You need to implement ub_send_flow in entry.h to call your UbApp::Send
  ub_send_flow(rank, dst, count, tag, msg_handler, fun_arg);

  return 0;
}

int ASTRASimUbNetwork::sim_recv(void* buffer, uint64_t count, int type, int src, int tag,
                                AstraSim::sim_request* request, void (*msg_handler)(void* fun_arg), void* fun_arg) {
  src += npu_offset;

  task1 t;
  t.src = src;
  t.dest = rank;
  t.count = count;
  t.type = 1;
  t.fun_arg = fun_arg;
  t.msg_handler = msg_handler;

  // Check if the data has already arrived (early arrival)
  if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) != recvHash.end()) {
    uint64_t received_count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
    
    if (received_count == t.count) {
      // Exact match: Clean up and trigger callback
      recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
      t.msg_handler(t.fun_arg);
    } else if (received_count > t.count) {
      // Received more than expected: Decrement and trigger callback
      recvHash[make_pair(tag, make_pair(t.src, t.dest))] = received_count - t.count;
      t.msg_handler(t.fun_arg);
    } else {
      // Received less than expected: Consume what we have and wait for the rest
      recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
      t.count -= received_count;
      expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
    }
  } else {
    // Data hasn't arrived yet: Register expectation
    if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) == expeRecvHash.end()) {
      expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
    } else {
      // Should not happen in standard operation, but handle gracefully
      // expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))].count += t.count;
    }
  }

  return 0;
}

static int user_param_prase(const int argc, char* argv[], user_param* user_param) {
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:f:g:s:n:o:p:a:r:c:"))!=-1) {
    switch (opt) {
    case 'h':
      std::cout<<"-- SimAI with ns3 frontend using Unified Bus backend" <<std::endl;
      std::cout<<"-t    number of threads,default 1" <<std::endl;
      std::cout<<"-w    workloads default none " <<std::endl;
      std::cout<<"-n    network topo" <<std::endl;
      std::cout<<"-c    network conf" <<std::endl;
      std::cout<<"-o    node conf" <<std::endl;
      std::cout<<"-p    transport protocol conf" <<std::endl;
      std::cout<<"-a    routing conf" <<std::endl;
      std::cout<<"-f    Conf folder" <<std::endl; 
      std::cout<<"-r    run name" <<std::endl;

      return 1;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'f':
      user_param->folder = optarg;
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
    case 'o':
      user_param->node_conf = optarg;
      break;
    case 'p':
      user_param->transport_conf = optarg;
      break;
    case 'a':
      user_param->routing_conf = optarg;
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
  if(user_param_prase(argc,argv,&user_param)){
    return 0;
  }
  MockNcclLog::set_log_name("SimAI" + (user_param.run_name.empty() ? "" : "." + user_param.run_name) +".log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");

  #ifdef NS3_MTP
    MtpInterface::Enable(user_param.thread);
  #endif

  if (!user_param.folder.empty()){
    user_param.network_conf = user_param.folder + "/network_attribute.txt";
    user_param.node_conf = user_param.folder + "/node.csv";
    user_param.network_topo = user_param.folder + "/topology.csv";
    user_param.routing_conf = user_param.folder + "/routing_table.csv";
    user_param.transport_conf = user_param.folder + "/transport_channel.csv";
  }
  
  setup_ns3_simulation(user_param);
    

    // /Setup ASTRA-sim
    // std::vector<ASTRASimUbNetwork*> networks(nodes_num, nullptr);
    // std::vector<AstraSim::Sys*> systems(nodes_num, nullptr);

    // for (uint32_t j = 0; j < nodes_num; j++) {
    //     networks[j] = new ASTRASimUbNetwork(j, 0);
    //     // Initialize systems[j] with your specific topology parameters (gpus_per_server, etc.)
    //     // systems[j] = new AstraSim::Sys(...);
    // }

    // for (uint32_t i = 0; i < nodes_num; i++) {
    //     // Trigger ASTRA-sim workloads
    //     // systems[i]->workload->fire();
    // }

    // // 4. Run the simulator
    UbUtils::Get()->PrintTimestamp("Running Simulator.");
    Simulator::Run();
    
    UbUtils::Get()->Destroy();
    Simulator::Destroy();
    
    return 0;
}