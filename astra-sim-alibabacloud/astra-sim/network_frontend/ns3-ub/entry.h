/* astra-sim/network_frontend/ns3-ub/entry.h */
 #ifndef __UB_ENTRY_H__
 #define __UB_ENTRY_H__
 #include "common.h"
 #include "astra-sim/system/AstraNetworkAPI.hh"
 #include "ns3/ub-switch.h"
 #include "ns3/ub-port.h"
 #include "ns3/ub-link.h"
 #include "ns3/ub-controller.h"
 #include "ns3/ub-ldst-instance.h"
 #include "ns3/ub-routing-process.h"
 #include "ns3/ub-app.h"
 #include "ns3/ub-utils.h"
 #include <map>

 using namespace ns3;
 using namespace utils;

 struct task1 {
  int src;
  int dest;
  int type;
  uint64_t count;
  void* fun_arg;
  void (*msg_handler)(void* fun_arg);
  double schTime;
};

inline std::map<std::pair<int, std::pair<int, int>>, task1> expeRecvHash;
inline std::map<std::pair<int, std::pair<int, int>>, uint64_t> recvHash;
inline std::map<std::pair<int, std::pair<int, int>>, task1> sentHash;

 struct user_param {
  int thread;
  string workload;
  string network_topo;
  string network_conf;
  string node_conf;
  string transport_conf;
  string routing_conf;
  string folder;
  string run_name;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
    node_conf = "";
    transport_conf = "";
    routing_conf = "";
    run_name = "";
  };
  ~user_param(){};
};

static std::once_flag sim_finished;
static std::atomic<bool> waiting_sim_finish(false);

void ub_send_flow(int src, int dst, uint64_t count, int tag, void (*msg_handler)(void*), void* fun_arg) {
    // 1. Get the source node
    Ptr<Node> sourceNode = NodeList::GetNode(src);
    
    // 2. Get the UbApp installed on this node
    // Assuming UbApp is the first application, or iterate to find it
    Ptr<UbApp> app = DynamicCast<UbApp>(sourceNode->GetApplication(0));
    
    if (app) {
        // 3. Call the send function on UbApp
        // You need to ensure UbApp has a method to accept (dst, count, tag)
        // app->Send(dst, count, tag); 
    }
}

void ub_receive_callback(int src, int dst, uint64_t count, int tag) {
    // if (expeRecvHash.find(make_pair(tag, make_pair(src, dst))) != expeRecvHash.end()) {
    //     task1 t = expeRecvHash[make_pair(tag, make_pair(src, dst))];
    //     if (count == t.count) {
    //         expeRecvHash.erase(make_pair(tag, make_pair(src, dst)));
    //         t.msg_handler(t.fun_arg);
    //     } else if (count > t.count) {
    //         recvHash[make_pair(tag, make_pair(src, dst))] = count - t.count;
    //         t.msg_handler(t.fun_arg);
    //         expeRecvHash.erase(make_pair(tag, make_pair(src, dst)));
    //     } else {
    //         t.count -= count;
    //         expeRecvHash[make_pair(tag, make_pair(src, dst))] = t;
    //     }
    // } else {
    //     if (recvHash.find(make_pair(tag, make_pair(src, dst))) == recvHash.end()) {
    //         recvHash[make_pair(tag, make_pair(src, dst))] = count;
    //     } else {
    //         recvHash[make_pair(tag, make_pair(src, dst))] += count;
    //     }
    // }
}

void finish() {
  // for (auto it : appCon) {
  //   for (int i = 0; i < it.second.GetN(); i++) {
  //     Ptr<RdmaClient> app = DynamicCast<RdmaClient>(it.second.Get(i));
  //     app->FinishQp();
  //   }
  // }
}

void check_sim_finish() {
  if (waiting_sim_finish 
      // && waiting_to_notify_receiver.empty() && waiting_to_sent_callback.empty() &&
      // waiting_to_message_finish.empty() && sentHash.empty() && expeRecvHash.empty()
    ) {
    std::call_once(sim_finished, [] {
      std::cout << "All messages finished. Stopping simulation at time "
                << AstraSim::Sys::boostedTick() << "."
                << std::endl;
      finish();
    });
    Simulator::Stop();
  }
}

int setup_ns3_simulation(user_param param) {
  clock_t begint, endt;
  begint = clock();

  // Setup UB Topology via helpers (use cvs from UB for now)
  UbUtils::Get()->PrintTimestamp("Start configuring UB Topology.");
    
  RngSeedManager::SetSeed(10);

  UbUtils::Get()->SetComponentsAttribute(param.network_conf);
  UbUtils::Get()->CreateTraceDir();
  UbUtils::Get()->CreateNode(param.node_conf);
  UbUtils::Get()->CreateTopo(param.network_topo);
  UbUtils::Get()->AddRoutingTable(param.routing_conf);
  UbUtils::Get()->CreateTp(param.transport_conf);
  UbUtils::Get()->TopoTraceConnect();

  uint32_t nodes_num = NodeList::GetNNodes();
  std::cout << "Num nodes." << nodes_num<< std::endl;

  //Attach UbApp to all hosts 
  for (uint32_t i = 0; i < nodes_num; ++i) {
    Ptr<Node> node = NodeList::GetNode(i);
    Ptr<UbSwitch> ubSwitch = node->GetObject<UbSwitch>();
     if ( (ubSwitch != NULL) && 
          (ubSwitch->GetNodeType() == 1) && // End-host
          (node->GetNApplications() == 0))  {
      // std::cout << i << " " << ubSwitch->GetNodeType() << std::endl;
      Ptr<UbApp> client = CreateObject<UbApp>();
      // client->SetRecvCallback(MakeCallback(&ub_receive_callback));
      node->AddApplication(client);
      UbUtils::Get()->ClientTraceConnect(i);
    }
  }
  // In RDMA frontend
  // SetConfig();
  // SetupRDMANetwork(qp_finish, message_finish, send_finish, recv_finish);

  std::cout << "Running Simulation.\n";
  fflush(stdout);
  NS_LOG_INFO("Run Simulation.");

  endt = clock();
  return 0;
}

 #endif