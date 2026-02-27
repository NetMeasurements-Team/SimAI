/* astra-sim/network_frontend/ns3-ub/common.h */
// #ifndef __UB_COMMON_H__
// #define __UB_COMMON_H__

// #include "ns3/core-module.h"
// #include "ns3/network-module.h"
// #include "ns3/ub-utils.h"

// #include <vector>
// #include <map>
// #include <string>

// using namespace ns3;
// using namespace std;

// // --- Global Simulation State ---
// inline uint32_t node_num, switch_num, nvswitch_num, link_num;
// inline uint32_t gpus_per_server;
// inline std::string topology_file;
// inline NodeContainer n; // Global NodeContainer

// // --- Routing Helper Structures ---
// struct Interface {
//   uint32_t idx;
//   bool up;
//   uint64_t bw;
//   Interface() : idx(0), up(false), bw(0) {}
// };

// // Maps for AstraSim Routing Calculation
// inline map<Ptr<Node>, map<Ptr<Node>, Interface>> nbr2if;
// inline map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node>>>> nextHop;

// // --- Configuration ---
// inline double switch_fw_delay = 500.0; // ns
// inline bool g_is_ub_mode = true;

// // --- Helper Functions for Routing (BFS) ---
// inline void CalculateRoute(Ptr<Node> host) {
//   vector<Ptr<Node>> q;
//   map<Ptr<Node>, int> dis;
  
//   q.push_back(host);
//   dis[host] = 0;
  
//   for (size_t i = 0; i < q.size(); i++) {
//     Ptr<Node> now = q[i];
//     int d = dis[now];
    
//     if (nbr2if.find(now) == nbr2if.end()) continue;

//     for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++) {
//       Ptr<Node> next = it->first;
//       if (dis.find(next) == dis.end()) {
//         dis[next] = d + 1;
//         // UB Specific: Identify switches vs hosts based on UbSwitch type if needed
//         q.push_back(next);
//       }
      
//       if (d + 1 == dis[next]) {
//         // Simple Equal-Cost Multi-Path (ECMP) logic
//         nextHop[next][host].push_back(now);
//       }
//     }
//   }
// }

// inline void CalculateRoutes(NodeContainer& nodes) {
//   for (uint32_t i = 0; i < nodes.GetN(); i++) {
//     // Only run BFS from hosts (assuming first 'node_num' nodes are hosts if mixed in list, 
//     // or check type. For now, run on all to be safe).
//     CalculateRoute(nodes.Get(i));
//   }
// }

// inline Ipv4Address node_id_to_ip(uint32_t id) {
//   return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
// }

// #endif