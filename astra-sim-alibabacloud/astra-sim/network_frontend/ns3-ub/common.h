/* astra-sim/network_frontend/ns3-ub/common.h
 *
 * SimAI topology parser, UB network setup, and BFS routing installation
 * for the ns3-ub frontend.
 *
 * Reads the same topology file format as gen_Topo_Template.py (the SimAI
 * standard format used by the RDMA frontend) and builds the UB network using
 * UbSwitch / UbPort / UbLink primitives directly — following the exact same
 * node+port pre-allocation pattern that UbUtils::CreateNode / CreateTopo uses.
 *
 * Topology file format (single text file, same as RDMA):
 *   Line 1:  <total_nodes> <gpus_per_server> <nvswitch_num> <switch_num> <link_num> <gpu_type_str>
 *   Next:    <nvswitch_num> node IDs (one per token)
 *   Next:    <switch_num>   node IDs (one per token)
 *   Next:    <link_num> lines of: <src> <dst> <bw_str> <delay_str> <error_rate>
 *
 * CRITICAL DESIGN NOTES:
 *
 * 1) Port pre-allocation (fixes UbLink::Attach SIGSEGV)
 *   UbLink::Attach stores the Ptr<UbPort> via assignment:
 *       m_link[m_nDevices++].m_src = device;
 *   ns3::Ptr::operator= calls m_ptr->Unref() on the current (null-initialized)
 *   pointer — crashes if m_ptr==nullptr. Fix: pre-create all UbPort objects
 *   via CreateObject<UbPort>() before any link wiring (UbUtils pattern).
 *
 * 2) Routing (fixes simulation hanging)
 *   UbApp::SendTraffic relies on UbController::TpConnManager::GetTpns() to find
 *   outports toward the destination and create Transport Paths on-demand.
 *   Without a routing table, GetTpns returns empty, bindRst=false, WQE is
 *   silently dropped, and AstraSim callbacks never fire.
 *   Fix: ComputeAndInstallRouting() runs BFS from every node and populates
 *   UbRoutingProcess::AddShortestRoute for every (node, destIP) pair.
 */
#ifndef __UB_COMMON_H__
#define __UB_COMMON_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/node-list.h"
#include "ns3/ub-switch.h"
#include "ns3/ub-port.h"
#include "ns3/ub-link.h"
#include "ns3/ub-app.h"
#include "ns3/ub-controller.h"
#include "ns3/ub-ldst-api.h"
#include "ns3/ub-congestion-control.h"
#include "ns3/ub-network-address.h"
#include "astra-sim/system/Common.hh"

using namespace ns3;
using namespace utils;
using namespace std;

NS_LOG_COMPONENT_DEFINE("SIMAI_UB_SIMULATION");

// ---------------------------------------------------------------------------
// Global topology state (populated by SetupUBNetwork)
// ---------------------------------------------------------------------------
inline uint32_t ub_node_num        = 0;
inline uint32_t ub_gpus_per_server = 1;
inline uint32_t ub_nvswitch_num    = 0;
inline uint32_t ub_switch_num      = 0;
inline uint32_t ub_link_num        = 0;
inline GPUType  ub_gpu_type        = GPUType::H100;
struct UbAdjEntry { uint32_t neighbor; uint16_t outport; };
inline std::vector<std::vector<UbAdjEntry>> ub_adj;  // ub_adj[node] = [{neighbor, outport}, ...]
inline std::vector<uint32_t> ub_node_type_vec;

// ---------------------------------------------------------------------------
// Network Configuration globals TODO generalize with RDMA frontend and use a class
// ---------------------------------------------------------------------------
std::string flow_file, trace_file, trace_output_file;
std::string fct_output_file = "fct.txt";
double simulator_stop_time;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline GPUType ParseGpuType(const std::string& s) {
    if (s == "A100") return GPUType::A100;
    if (s == "A800") return GPUType::A800;
    if (s == "H100") return GPUType::H100;
    if (s == "H800") return GPUType::H800;
    return GPUType::NONE;
}

// ---------------------------------------------------------------------------
// SetupUBNetwork
//
//
//   Node creation:
//     - Parse header and node-type list
//     - Scan link list to count ports per node
//     - Create each Node, aggregate UbSwitch / UbController / UbLdstInstance
//     - Pre-allocate exactly the right number of UbPort objects per node
//     - Call ubSw->Init() and create UbCongestionControl
//
//   link wiring:
//     - For each link, retrieve the pre-existing UbPort objects by device index
//     - Create UbLink, set DataRate and Delay
//     - Call p1->Attach(channel) then p2->Attach(channel)
//       (UbPort::Attach calls channel->Attach(this) internally, which safely
//        assigns to the pre-constructed m_link[].m_src slot)
//
//   Returns: number of GPU (DEVICE) nodes, or -1 on error.
// ---------------------------------------------------------------------------
inline int SetupUBNetwork(
    const std::string& topo_file,
    void (*ub_task_completed_callback)(FILE*, uint32_t, uint32_t, uint32_t taskId)) {

    std::ifstream ifs(topo_file);
    if (!ifs.is_open()) {
        std::cerr << "[UB] ERROR: Cannot open topology file: " << topo_file << std::endl;
        return -1;
    }

    // Parse header
    std::string gpu_type_str;
    ifs >> ub_node_num >> ub_gpus_per_server >> ub_nvswitch_num
        >> ub_switch_num >> ub_link_num >> gpu_type_str;
    ub_gpu_type = ParseGpuType(gpu_type_str);

    if (ub_nvswitch_num > 0) {
        std::cerr << "[UB] WARNING: topology has " << ub_nvswitch_num
                  << " NVSwitch node(s). UB has no NVSwitch concept — "
                  << "treating them as plain UB switches." << std::endl;
    }

    std::cout << "[UB] Topology: nodes=" << ub_node_num
              << " gps=" << ub_gpus_per_server
              << " nvsw=" << ub_nvswitch_num
              << " sw=" << ub_switch_num
              << " links=" << ub_link_num
              << " gpu=" << gpu_type_str << std::endl;

    // Parse node-type list
    ub_node_type_vec.assign(ub_node_num, 0);  // default: DEVICE
    for (uint32_t i = 0; i < ub_nvswitch_num; i++) {
        uint32_t sid; ifs >> sid;
        ub_node_type_vec[sid] = 1;  // treat NVSwitch as SWITCH
    }
    for (uint32_t i = 0; i < ub_switch_num; i++) {
        uint32_t sid; ifs >> sid;
        ub_node_type_vec[sid] = 1;  // SWITCH
    }

    // Read all link records into memory 
    struct LinkRecord { uint32_t src, dst; std::string bw, delay; };
    std::vector<LinkRecord> links;
    links.reserve(ub_link_num);
    for (uint32_t i = 0; i < ub_link_num; i++) {
        LinkRecord lr;
        double err;
        ifs >> lr.src >> lr.dst >> lr.bw >> lr.delay >> err;
        links.push_back(lr);
    }
    ifs.close();

    // Count ports per node and GPUs (end-hosts)
    int gpu_count = 0;
    std::vector<uint32_t> port_count(ub_node_num, 0);
    for (auto& lr : links) {
        port_count[lr.src]++;
        port_count[lr.dst]++;
    }

    // Create nodes and pre-allocate ports
    FILE *fct_output = fopen(fct_output_file.c_str(), "w");
    for (uint32_t i = 0; i < ub_node_num; i++) {
        Ptr<Node> node = CreateObject<Node>();

        // Put UbSwitch and UbLdstInstance in all node types (to handle packet forwarding)
        Ptr<UbSwitch> sw = CreateObject<UbSwitch>();
        node->AggregateObject(sw);
        Ptr<UbLdstInstance> ldst = CreateObject<UbLdstInstance>();
        node->AggregateObject(ldst);
        ldst->Init(node->GetId());

        // Set node type
        if (ub_node_type_vec[i] == 0) {
            // Attach UbController to GPU (DEVICE) for transport channel management
            sw->SetNodeType(UB_DEVICE);
            Ptr<UbController> ctrl = CreateObject<UbController>();
            node->AggregateObject(ctrl);
            ctrl->CreateUbFunction();
            ctrl->CreateUbTransaction();
            
            // Attach UbApp to GPU (DEVICE) nodes for AstraSim traffic injection
            Ptr<UbApp> app = CreateObject<UbApp>();
            app->TraceConnectWithoutContext("WqeTaskCompletesNotify", MakeBoundCallback(ub_task_completed_callback, fct_output));

            node->AddApplication(app);

            gpu_count++;
        } else {
            sw->SetNodeType(UB_SWITCH);
        }

        // Pre-allocate one UbPort per link on this node.
        uint32_t nports = port_count[i];
        for (uint32_t p = 0; p < nports; p++) {
            Ptr<UbPort> port = CreateObject<UbPort>();
            port->SetAddress(Mac48Address::Allocate());
            node->AddDevice(port);
        }

        // Initialise switch internals
        sw->Init();

        // Congestion control 
        auto cc = UbCongestionControl::Create(ub_node_type_vec[i] == 0 ? UB_DEVICE : UB_SWITCH);
        cc->OnSwitchAttached(sw);

        NS_LOG_INFO("[UB] Created node " << i
                    << (ub_node_type_vec[i] == 0 ? " (DEVICE)" : " (SWITCH)")
                    << " with " << nports << " ports");
    }

    // Wire links using pre-existing ports and build adjacency list
    std::vector<uint32_t> next_port(ub_node_num, 0);
    ub_adj.assign(ub_node_num, {});

    for (auto& lr : links) {
        Ptr<Node> n1 = NodeList::GetNode(lr.src);
        Ptr<Node> n2 = NodeList::GetNode(lr.dst);

        // Retrieve the pre-allocated port at the next available slot
        uint32_t p1_idx = next_port[lr.src]++;
        uint32_t p2_idx = next_port[lr.dst]++;

        Ptr<UbPort> p1 = DynamicCast<UbPort>(n1->GetDevice(p1_idx));
        Ptr<UbPort> p2 = DynamicCast<UbPort>(n2->GetDevice(p2_idx));

        NS_ASSERT_MSG(p1 != nullptr,
            "[UB] Port " << p1_idx << " on node " << lr.src << " is not a UbPort");
        NS_ASSERT_MSG(p2 != nullptr,
            "[UB] Port " << p2_idx << " on node " << lr.dst << " is not a UbPort");

        // Set data rate on both ends
        p1->SetDataRate(DataRate(lr.bw));
        p2->SetDataRate(DataRate(lr.bw));

        // Create the link channel and connect both ports
        Ptr<UbLink> channel = CreateObject<UbLink>();
        channel->SetAttribute("Delay", StringValue(lr.delay));
        p1->Attach(channel);
        p2->Attach(channel);

        // Record bidirectional adjacency for routing BFS
        ub_adj[lr.src].push_back({lr.dst, static_cast<uint16_t>(p1_idx)});
        ub_adj[lr.dst].push_back({lr.src, static_cast<uint16_t>(p2_idx)});

        NS_LOG_INFO("[UB] Link " << lr.src << "[p" << p1_idx << "] <-> "
                    << lr.dst << "[p" << p2_idx
                    << "]  bw=" << lr.bw << "  delay=" << lr.delay);
    }

    std::cout << "[UB] Network ready: "
              << gpu_count    << " GPU (DEVICE) nodes, "
              << ub_switch_num << " switches, "
              << ub_nvswitch_num << " NVSwitches (as switches)." << std::endl;
    return gpu_count;
}


// ---------------------------------------------------------------------------
// ComputeAndInstallRouting
//
//   Runs BFS from every node to compute shortest paths, then installs
//   routing entries via UbRoutingProcess::AddShortestRoute.
//
//   The routing API uses destination IP (not node ID) as the key:
//     destIP = NodeIdToIp(destNodeId).Get()   (from ub-network-address.h)
//
//   For each node N and every other reachable node D:
//     - BFS finds all next-hop neighbors toward D at minimum distance
//     - For each switch next-hop, the outport on N toward that neighbor is added
//     - AddShortestRoute(destIP, {outPorts...}) is called on N's UbRoutingProcess
//
//   The adjacency is built from the same link records used during SetupUBNetwork
//   (passed in as a parameter to avoid re-reading the file).
// ---------------------------------------------------------------------------

inline void ComputeAndInstallRouting() {
    uint32_t N = ub_node_num;
    if (N == 0 || ub_adj.empty()) {
        std::cerr << "[UB] ERROR: ComputeAndInstallRouting called before SetupUBNetwork" << std::endl;
        return;
    }

    // Compute all-pairs shortest paths using BFS from every node
    std::vector<std::vector<uint32_t>> dist(N, std::vector<uint32_t>(N, UINT32_MAX));
    for (uint32_t src = 0; src < N; src++) {
        dist[src][src] = 0;
        std::queue<uint32_t> q;
        q.push(src);
        while (!q.empty()) {
            uint32_t u = q.front(); q.pop();

            // End-hosts (DEVICE nodes) should not act as transit routers,
            // do not expand neighbors of a DEVICE node unless it is the source node.
            if (ub_node_type_vec[u] == 0 /*DEVICE*/ && u != src) {
                continue;
            }

            for (auto& adj : ub_adj[u]) {
                if (dist[src][adj.neighbor] == UINT32_MAX) {
                    dist[src][adj.neighbor] = dist[src][u] + 1;
                    q.push(adj.neighbor);
                }
            }
        }
    }

    uint32_t route_count = 0;

    // For each source node, find shortest-path out-ports toward every destination
    for (uint32_t src = 0; src < N; src++) {
        Ptr<UbSwitch> srcSw = NodeList::GetNode(src)->GetObject<UbSwitch>();
        Ptr<UbRoutingProcess> rt = srcSw->GetRoutingProcess();

        // For each destination D != src, find all shortest-path next-hops from src
        for (uint32_t dst = 0; dst < N; dst++) {
            if (dst == src || dist[src][dst] == UINT32_MAX) continue;

            // Collect outports from src toward all neighbors that lie on a shortest path to dst
            std::vector<uint16_t> shortest_outports;
            for (auto& adj : ub_adj[src]) {
                uint32_t nbr = adj.neighbor;
                // Neighbor is on a shortest path if dist[nbr][dst] == dist[src][dst] - 1
                if (dist[nbr][dst] != UINT32_MAX && dist[nbr][dst] == dist[src][dst] - 1) {
                    shortest_outports.push_back(adj.outport);
                }
            }

            if (shortest_outports.empty()) continue;

            // Install per-node IP AND per-port IP (UbUtils installs both)
            uint32_t destNodeIp  = NodeIdToIp(dst).Get();
            rt->AddShortestRoute(destNodeIp, shortest_outports);

            // Also install per-port entries for each port on the destination node.
            // NOTE: A specific port p on the destination node dst is only reachable if the path
            // ends at the specific neighbor (last-hop switch) connected to that port.
            uint32_t n_dst_ports = NodeList::GetNode(dst)->GetNDevices();
            for (uint32_t p = 0; p < n_dst_ports; p++) {
                uint32_t destPortIp = NodeIdToIp(dst, p).Get();
                
                // Find the neighbor of dst connected to port p
                uint32_t last_hop = UINT32_MAX;
                for (auto& adj : ub_adj[dst]) {
                    if (adj.outport == p) {
                        last_hop = adj.neighbor;
                        break;
                    }
                }
                
                if (last_hop == UINT32_MAX) continue;
                // Collect next-hop outports from src toward last_hop
                std::vector<uint16_t> port_shortest_outports;
                if (src == last_hop) {
                    // Next-hop is dst, find the outport on src towards dst
                    for (auto& adj : ub_adj[src]) {
                        if (adj.neighbor == dst) {
                            port_shortest_outports.push_back(adj.outport);
                        }
                    }
                } else {
                    for (auto& adj : ub_adj[src]) {
                        uint32_t nbr = adj.neighbor;
                        if (dist[nbr][last_hop] != UINT32_MAX && dist[nbr][last_hop] == dist[src][last_hop] - 1) {
                            port_shortest_outports.push_back(adj.outport);
                        }
                    }
                }
                
                if (!port_shortest_outports.empty()) {
                    rt->AddShortestRoute(destPortIp, port_shortest_outports);
                }
            }
            route_count++;
        }
    }

    std::cout << "[UB] Routing installed: " << route_count
              << " destination entries across " << N << " nodes." << std::endl;
}


string extend_output_file_name(const string &instance_name, string output_file) {
  if (instance_name.empty()) {
    return output_file;
  }
  auto idx = output_file.find_last_of('/');
  idx = idx == string::npos ? 0 : idx + 1;
  return output_file.substr(0, idx) + instance_name + "." + output_file.substr(idx);
}

bool ReadConf(const string& network_conf, const string& run_name) {
  std::ifstream conf;
  conf.open(network_conf);
  if (!conf.is_open()) {
    return false;
  }
  //TODO check if file exists
  while (!conf.eof()) {
    std::string key;
    conf >> key;
    // FCT_OUTPUT_FILE is the only file actually written
    if (key.compare("FLOW_FILE") == 0) {
      conf >> flow_file;
    } else if (key.compare("TRACE_FILE") == 0) {
      conf >> trace_file;
    } else if (key.compare("TRACE_OUTPUT_FILE") == 0) {
      conf >> trace_output_file;
      trace_output_file = extend_output_file_name(run_name, trace_output_file);
    } else if (key.compare("SIMULATOR_STOP_TIME") == 0) {
      double v;
      conf >> v;
      simulator_stop_time = v;
    } else if (key.compare("FCT_OUTPUT_FILE") == 0) {
      conf >> fct_output_file;
      fct_output_file = extend_output_file_name(run_name, fct_output_file);
    } // else if (key.compare("ENABLE_TRACE") == 0) { TODO
    //   conf >> enable_trace;
    fflush(stdout);
  }
  conf.close();
  return true;

}

#endif /* __UB_COMMON_H__ */