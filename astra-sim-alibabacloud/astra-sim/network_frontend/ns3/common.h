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

#ifndef __COMMON_H__
#define __COMMON_H__

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include <ns3/core-module.h>
#include <ns3/error-model.h>
#include <ns3/internet-module.h>
#include <ns3/nvswitch-node.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/qbb-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>

#include <fstream>
#include <iostream>
#include <time.h>
#include <unordered_map>
#include <utility>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");

inline uint32_t cc_mode = 1;
inline bool enable_qcn = true, use_dynamic_pfc_threshold = true;
inline uint32_t packet_payload_size = 1000, l2_chunk_size = 0, l2_ack_interval = 0;
inline bool enable_pfc = false;
inline double switch_fw_delay = 0.0;
inline EventId monitor_qlen_event;
inline EventId monitor_bw_event;
inline EventId monitor_qp_rate_event;
inline EventId monitor_qp_cnp_num_event;
inline double pause_time = 5, simulator_stop_time = 3.01;
inline std::string data_rate, link_delay, topology_file, flow_file, trace_file, trace_output_file;
inline std::string fct_output_file = "fct.txt";
inline std::string pfc_output_file = "pfc.txt";
inline std::string send_output_file = "send.txt";
std::string recv_output_file = "recv.txt";

inline double alpha_resume_interval = 55, rp_timer, ewma_gain = 1 / 16;
inline double rate_decrease_interval = 4;
inline uint32_t fast_recovery_times = 5;
inline std::string rate_ai, rate_hai, min_rate = "100Mb/s";
inline std::string dctcp_rate_ai = "1000Mb/s";

inline bool clamp_target_rate = false, l2_back_to_zero = false;
inline double error_rate_per_link = 0.0;
inline uint32_t has_win = 1;
inline uint32_t global_t = 1;
inline uint32_t mi_thresh = 5;
inline bool flow_stripping = false;
inline bool packet_spraying = false;
inline bool reuse_qps = true;
inline bool var_win = false, fast_react = true;
inline bool multi_rate = true;
inline bool sample_feedback = false;
inline double pint_log_base = 1.05;
inline double pint_prob = 1.0;
inline double u_target = 0.95;
inline uint32_t int_multi = 1;
inline bool rate_bound = true;
inline int nic_total_pause_time = 0;
inline std::string nic_coalesce_method = "PER_QP";
inline double nack_gen_interval = 0.01;
inline size_t max_rdma_out_of_seq = 4;
inline bool r_dcqcn_ewma_gain, r_dcqcn_clamp = false;
inline uint32_t r_dcqcn_f = 1, r_dcqcn_bytes_threshold = 524240;
inline std::string r_dcqcn_rate_u_delay = "300us";
inline std::string r_dcqcn_alpha_u_delay = "2.56us";

inline uint32_t ack_high_prio = 0;
inline uint64_t link_down_time = 0;
inline uint32_t link_down_A = 0, link_down_B = 0;
inline uint32_t enable_trace = 1;
inline uint32_t buffer_size = 16;

inline uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
inline GPUType gpu_type;
inline std::vector<int> NVswitchs;

inline uint32_t qp_mon_interval = 100;
inline uint32_t bw_mon_interval = 10000;
inline uint32_t qlen_mon_interval = 10000;
inline uint64_t mon_start = 0, mon_end = 2100000000;

inline string qlen_mon_file;
inline string bw_mon_file;
inline string rate_mon_file;
inline string cnp_mon_file;
inline string total_flow_file = "/root/astra-sim/extern/network_backend/ns3-interface/simulation/monitor_output/";
inline FILE* total_flow_output = nullptr;

inline unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
inline unordered_map<uint64_t, double> rate2pmax;
inline std::ifstream topo_ifs, flowf, tracef;
inline NodeContainer n;
inline uint64_t nic_rate;
inline uint64_t maxRtt, maxBdp;
inline std::vector<Ipv4Address> serverAddress;
inline std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t>> portNumber;
inline std::map<std::pair<uint32_t, uint32_t>, uint32_t> ecmp_ctr;

struct Interface {
  uint32_t idx;
  bool up;
  uint64_t delay;
  uint64_t bw;

  Interface() : idx(0), up(false) {}
};
inline map<Ptr<Node>, map<Ptr<Node>, Interface>> nbr2if;
inline map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node>>>> nextHop;
inline map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairDelay;
inline map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairTxDelay;
inline map<uint32_t, map<uint32_t, uint64_t>> pairBw;
inline map<Ptr<Node>, map<Ptr<Node>, uint64_t>> pairBdp;
inline map<uint32_t, map<uint32_t, uint64_t>> pairRtt;

struct FlowInput {
  uint32_t src, dst, pg, maxPacketCount, port, dport;
  double start_time;
  uint32_t idx;
};

inline FlowInput flow_input = {};
inline uint32_t flow_num;
inline Ipv4Address node_id_to_ip(uint32_t id) {
  return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
}

uint32_t ip_to_node_id(Ipv4Address ip) { return (ip.Get() >> 8) & 0xffff; }

inline void get_pfc(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t type) {
  fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(),
          dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(),
          dev->GetIfIndex(), type);
}

struct QlenDistribution {
  vector<uint32_t> cnt;

  void add(uint32_t qlen) {
    uint32_t kb = qlen / 1000;
    if (cnt.size() < kb + 1)
      cnt.resize(kb + 1);
    cnt[kb]++;
  }
};

inline void monitor_qlen(FILE* qlen_output, NodeContainer* n) {
  for (uint32_t i = 0; i < n->GetN(); i++) {
    if (n->Get(i)->GetNodeType() == 1) {
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
      sw->PrintSwitchQlen(qlen_output);
    } else if (n->Get(i)->GetNodeType() == 2) {
      Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n->Get(i));
      sw->PrintSwitchQlen(qlen_output);
    }
  }
  Simulator::Schedule(MicroSeconds(qlen_mon_interval), &monitor_qlen, qlen_output, n);
}

void monitor_bw(FILE* bw_output, NodeContainer* n) {
  for (uint32_t i = 0; i < n->GetN(); i++) {
    if (n->Get(i)->GetNodeType() == 1) { // is switch
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
      sw->PrintSwitchBw(bw_output, bw_mon_interval);
    } else if (n->Get(i)->GetNodeType() == 2) { // is nvswitch
      Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n->Get(i));
      sw->PrintSwitchBw(bw_output, bw_mon_interval);
    } else { // is host
      Ptr<Node> host = n->Get(i);
      host->GetObject<RdmaDriver>()->m_rdma->PrintHostBW(bw_output, bw_mon_interval);
    }
  }
  if (bw_mon_interval + Simulator::Now().GetMicroSeconds() < mon_end) {
    monitor_bw_event = Simulator::Schedule(MicroSeconds(bw_mon_interval), &monitor_bw, bw_output, n);
  }
}

inline void monitor_qp_rate(FILE* rate_output, NodeContainer* n) {
  for (uint32_t i = 0; i < n->GetN(); i++) {
    if (n->Get(i)->GetNodeType() == 0) {
      Ptr<Node> host = n->Get(i);
      host->GetObject<RdmaDriver>()->m_rdma->PrintQPRate(rate_output);
    }
  }
  Simulator::Schedule(MicroSeconds(qp_mon_interval), &monitor_qp_rate, rate_output, n);
}

inline void monitor_qp_cnp_number(FILE* cnp_output, NodeContainer* n) {
  for (uint32_t i = 0; i < n->GetN(); i++) {
    if (n->Get(i)->GetNodeType() == 0) {
      Ptr<Node> host = n->Get(i);
      host->GetObject<RdmaDriver>()->m_rdma->PrintQPCnpNumber(cnp_output);
    }
  }
  Simulator::Schedule(MicroSeconds(qp_mon_interval), &monitor_qp_cnp_number, cnp_output, n);
}

inline void schedule_monitor() {
  FILE* qlen_output = fopen(qlen_mon_file.c_str(), "w");
  assert(qlen_output != nullptr);
  fprintf(qlen_output, "%s, %s, %s, %s, %s, %s\n", "time", "sw_id", "port_id", "q_id", "q_len", "port_len");
  fflush(qlen_output);
  Simulator::Schedule(MicroSeconds(mon_start), &monitor_qlen, qlen_output, &n);

  FILE* bw_output = fopen(bw_mon_file.c_str(), "w");
  assert(bw_output != nullptr);
  fprintf(bw_output, "%s, %s, %s, %s\n", "time", "node_id", "port_id", "bandwidth");
  fflush(bw_output);
  Simulator::Schedule(MicroSeconds(mon_start), &monitor_bw, bw_output, &n);

  FILE* rate_output = fopen(rate_mon_file.c_str(), "w");
  assert(rate_output != nullptr);
  fprintf(rate_output, "%s, %s, %s, %s, %s, %s, %s\n", "time", "src", "dst", "sport", "dport", "size", "curr_rate");
  fflush(rate_output);
  Simulator::Schedule(MicroSeconds(mon_start), &monitor_qp_rate, rate_output, &n);

  FILE* cnp_output = fopen(cnp_mon_file.c_str(), "w");
  assert(cnp_output != nullptr);
  fprintf(cnp_output, "%s, %s, %s, %s, %s, %s, %s\n", "time", "src", "dst", "sport", "dport", "size", "cnp_number");
  fflush(cnp_output);
  Simulator::Schedule(MicroSeconds(mon_start), &monitor_qp_cnp_number, cnp_output, &n);
}

inline void CalculateRoute(Ptr<Node> host) {
  vector<Ptr<Node>> q;
  map<Ptr<Node>, int> dis;
  map<Ptr<Node>, uint64_t> delay;
  map<Ptr<Node>, uint64_t> txDelay;
  map<Ptr<Node>, uint64_t> bw;
  q.push_back(host);
  dis[host] = 0;
  delay[host] = 0;
  txDelay[host] = 0;
  bw[host] = 0xfffffffffffffffflu;
  for (int i = 0; i < static_cast<int>(q.size()); i++) {
    Ptr<Node> now = q[i];
    int d = dis[now];
    for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++) {
      if (!it->second.up)
        continue;
      Ptr<Node> next = it->first;
      if (dis.find(next) == dis.end()) {
        dis[next] = d + 1;
        delay[next] = delay[now] + it->second.delay;
        txDelay[next] = txDelay[now] + packet_payload_size * 1000000000lu * 8 / it->second.bw;
        bw[next] = std::min(bw[now], it->second.bw);
        if (next->GetNodeType() == 1 || next->GetNodeType() == 2) {
          q.push_back(next);
        }
      }
      if (d + 1 == dis[next]) {
        bool via_nvswitch = false;
        for (auto x : nextHop[next][host]) {
          if (x->GetNodeType() == 2)
            via_nvswitch = true;
        }
        if (via_nvswitch == false) {
          if (now->GetNodeType() == 2) {
            while (nextHop[next][host].size() != 0)
              nextHop[next][host].pop_back();
          }
          nextHop[next][host].push_back(now);
        } else if (via_nvswitch == true && now->GetNodeType() == 2) {
          nextHop[next][host].push_back(now);
        }
        if (next->GetNodeType() == 0 && nextHop[next][now].size() == 0) {
          nextHop[next][now].push_back(now);
          pairBw[next->GetId()][now->GetId()] = pairBw[now->GetId()][next->GetId()] = it->second.bw;
        }
      }
    }
  }
  for (auto it : delay) {
    pairDelay[it.first][host] = it.second;
  }
  for (auto it : txDelay)
    pairTxDelay[it.first][host] = it.second;
  for (auto it : bw) {
    pairBw[it.first->GetId()][host->GetId()] = it.second;
  }
}

inline void CalculateRoutes(NodeContainer& n) {
  for (int i = 0; i < static_cast<int>(n.GetN()); i++) {
    Ptr<Node> node = n.Get(i);
    if (node->GetNodeType() == 0)
      CalculateRoute(node);
  }
}

inline void SetRoutingEntries() {
  for (auto i = nextHop.begin(); i != nextHop.end(); ++i) {
    Ptr<Node> node = i->first;
    auto& table = i->second;
    for (auto j = table.begin(); j != table.end(); ++j) {
      Ptr<Node> dst = j->first;
      Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      vector<Ptr<Node>> nexts = j->second;
      for (int k = 0; k < static_cast<int>(nexts.size()); k++) {
        Ptr<Node> next = nexts[k];
        const uint32_t interface = nbr2if[node][next].idx;
        if (node->GetNodeType() == 1) {
          DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
        } else if (node->GetNodeType() == 2) {
          DynamicCast<NVSwitchNode>(node)->AddTableEntry(dstAddr, interface);
          node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface, true);
        } else {
          bool is_nvswitch = false;
          if (next->GetNodeType() == 2) {
            is_nvswitch = true;
          }
          node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface, is_nvswitch);
          if (next->GetId() == dst->GetId()) {
            node->GetObject<RdmaDriver>()->m_rdma->add_nvswitch(dst->GetId());
          }
        }
      }
    }
  }
}

inline void printRoutingEntries() {
  map<uint32_t, string> types;
  types[0] = "HOST";
  types[1] = "SWITCH";
  types[2] = "NVSWITCH";
  map<Ptr<Node>, map<Ptr<Node>, vector<pair<Ptr<Node>, uint32_t> >>> NVSwitch, NetSwitch, Host; 
  for (auto i = nextHop.begin(); i != nextHop.end(); ++i) {
    Ptr<Node> src = i -> first;
    auto &table = i->second;
    for (auto j = table.begin(); j != table.end(); ++j) {
      Ptr<Node> dst = j->first;
      Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
      vector<Ptr<Node>> nexts = j->second;
      for (int k = 0; k < (int)nexts.size(); k++) {
        Ptr<Node> firstHop = nexts[k];
        uint32_t interface = nbr2if[src][firstHop].idx;
        if (src->GetNodeType() == 0) {
          Host[src][dst].emplace_back(firstHop, interface);
        } else if(src->GetNodeType() == 1) {
          NetSwitch[src][dst].emplace_back(firstHop, interface);
        } else if(src->GetNodeType() == 2) {
          NVSwitch[src][dst].emplace_back(firstHop, interface);
        }
      }
    }
  }

  cout << "*********************    PRINT SWITCH ROUTING TABLE    *********************" << endl << endl << endl;
  for(auto it = NetSwitch.begin(); it != NetSwitch.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "SWITCH: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId()
             << "[" << types[dst->GetNodeType()]
             << "] via " << nextHop->GetId()
             << "[" << types[nextHop->GetNodeType()]
             << "] from port: " << interface
             << endl;
      }
    }
  } 

  cout << "*********************    PRINT NVSWITCH ROUTING TABLE    *********************" << endl  << endl << endl;
  for(auto it = NVSwitch.begin(); it != NVSwitch.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "NVSWITCH: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId() << "[" << types[dst->GetNodeType()] << "] via " << nextHop->GetId() << "[" << types[nextHop->GetNodeType()] << "]" << " from port: " << interface << endl;
      }
    }
  } 

  cout << "*********************    HOST ROUTING TABLE    *********************" << endl << endl << endl;
  for(auto it = Host.begin(); it != Host.end(); ++ it) {
    Ptr<Node> src = it -> first;
    auto table = it -> second;
    cout << "HOST: " << src->GetId() << "'s routing entries are as follows:" << endl;
    for(auto j = table.begin(); j != table.end(); ++ j) {
      Ptr<Node> dst = j -> first;
      auto entries = j -> second;
      for(auto k = entries.begin(); k != entries.end(); ++ k) {
        Ptr<Node> nextHop = k->first;
        uint32_t interface = k->second;
        cout << "To " << dst->GetId() << "[" << types[dst->GetNodeType()] << "] via " << nextHop->GetId() << "[" << types[nextHop->GetNodeType()] << "]" << " from port: " << interface << endl;
      }
    }
  }
}

inline bool validateRoutingEntries() {
  return false;
}

inline void TakeDownLink(NodeContainer n, Ptr<Node> a, Ptr<Node> b) {
  if (!nbr2if[a][b].up) {
    return;
  }
  nbr2if[a][b].up = nbr2if[b][a].up = false;
  nextHop.clear();
  CalculateRoutes(n);
	for (uint32_t i = 0; i < n.GetN(); i++) {
		if (n.Get(i)->GetNodeType() == 1) {
		  DynamicCast<SwitchNode>(n.Get(i))->ClearTable();
		} else if(n.Get(i)->GetNodeType() == 2) {
		  DynamicCast<NVSwitchNode>(n.Get(i))->ClearTable();
		} else {
		  n.Get(i)->GetObject<RdmaDriver>()->m_rdma->ClearTable();
		}
	}
  DynamicCast<QbbNetDevice>(a->GetDevice(nbr2if[a][b].idx))->TakeDown();
  DynamicCast<QbbNetDevice>(b->GetDevice(nbr2if[b][a].idx))->TakeDown();
  SetRoutingEntries();

  for (uint32_t i = 0; i < n.GetN(); i++) {
    if (n.Get(i)->GetNodeType() == 0) {
      n.Get(i)->GetObject<RdmaDriver>()->m_rdma->RedistributeQp();
    }
  }
}

inline string get_output_file_name(const string& config_file, const string& output_file) {
  const auto idx = config_file.find_last_of('/');
	string ans = output_file.substr(0, output_file.length() - 4) + config_file.substr(idx + 7);
	return ans;
}

inline string extend_output_file_name(const string &instance_name, string output_file) {
  if (instance_name.empty()) {
    return output_file;
  }
  auto idx = output_file.find_last_of('/');
  idx = idx == string::npos ? 0 : idx + 1;
  return output_file.substr(0, idx) + instance_name + "." + output_file.substr(idx);
}

inline uint64_t get_nic_rate(const NodeContainer &n) {
  for (uint32_t i = 0; i < n.GetN(); i++)
    if (n.Get(i)->GetNodeType() == 0)
      return DynamicCast<QbbNetDevice>(n.Get(i)->GetDevice(1))
          ->GetDataRate()
          .GetBitRate();
  return 0;
}

inline bool ReadConf(const string& network_topo, const string& network_conf, const string& run_name) {
  std::ifstream conf;
  conf.open(network_conf);
  if (!conf.is_open()) {
    return false;
  }
  topology_file = network_topo;
  while (!conf.eof()) {
    std::string key;
    conf >> key;

    if (key.compare("ENABLE_QCN") == 0) {
      uint32_t v;
      conf >> v;
      enable_qcn = v;
    } else if (key.compare("USE_DYNAMIC_PFC_THRESHOLD") == 0) {
      uint32_t v;
      conf >> v;
      use_dynamic_pfc_threshold = v;
    } else if (key.compare("CLAMP_TARGET_RATE") == 0) {
      uint32_t v;
      conf >> v;
      clamp_target_rate = v;
    } else if (key.compare("PAUSE_TIME") == 0) {
      double v;
      conf >> v;
      pause_time = v;
    } else if (key.compare("DATA_RATE") == 0) {
      std::string v;
      conf >> v;
      data_rate = v;
    } else if (key.compare("LINK_DELAY") == 0) {
      std::string v;
      conf >> v;
      link_delay = v;
    } else if (key.compare("PACKET_PAYLOAD_SIZE") == 0) {
      uint32_t v;
      conf >> v;
      packet_payload_size = v;
    } else if (key.compare("L2_CHUNK_SIZE") == 0) {
      uint32_t v;
      conf >> v;
      l2_chunk_size = v;
    } else if (key.compare("L2_ACK_INTERVAL") == 0) {
      uint32_t v;
      conf >> v;
      l2_ack_interval = v;
    } else if (key.compare("L2_BACK_TO_ZERO") == 0) {
      uint32_t v;
      conf >> v;
      l2_back_to_zero = v;
    } else if (key.compare("FLOW_FILE") == 0) {
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
    } else if (key.compare("ALPHA_RESUME_INTERVAL") == 0) {
      double v;
      conf >> v;
      alpha_resume_interval = v;
    } else if (key.compare("RP_TIMER") == 0) {
      double v;
      conf >> v;
      rp_timer = v;
    } else if (key.compare("EWMA_GAIN") == 0) {
      double v;
      conf >> v;
      ewma_gain = v;
    } else if (key.compare("FAST_RECOVERY_TIMES") == 0) {
      uint32_t v;
      conf >> v;
      fast_recovery_times = v;
    } else if (key.compare("RATE_AI") == 0) {
      std::string v;
      conf >> v;
      rate_ai = v;
    } else if (key.compare("RATE_HAI") == 0) {
      std::string v;
      conf >> v;
      rate_hai = v;
    } else if (key.compare("ERROR_RATE_PER_LINK") == 0) {
      double v;
      conf >> v;
      error_rate_per_link = v;
    } else if (key.compare("CC_MODE") == 0) {
      conf >> cc_mode;
    } else if (key.compare("RATE_DECREASE_INTERVAL") == 0) {
      double v;
      conf >> v;
      rate_decrease_interval = v;
    } else if (key.compare("MIN_RATE") == 0) {
      conf >> min_rate;
    } else if (key.compare("FCT_OUTPUT_FILE") == 0) {
      conf >> fct_output_file;
      fct_output_file = extend_output_file_name(run_name, fct_output_file);
    } else if (key.compare("HAS_WIN") == 0) {
      conf >> has_win;
    } if (key.compare("REUSE_QPS") == 0) {
      uint32_t v;
      conf >> v;
      reuse_qps = v;
    } if (key.compare("ENABLE_FLOW_STRIPPING") == 0) {
      uint32_t v;
      conf >> v;
      flow_stripping = v;
    } else if (key.compare("ENABLE_PACKET_SPRAYING") == 0) {
      uint32_t v;
      conf >> v;
      packet_spraying = v;
    } else if (key.compare("GLOBAL_T") == 0) {
      conf >> global_t;
      global_t = 1;
    } else if (key.compare("MI_THRESH") == 0) {
      conf >> mi_thresh;
    } else if (key.compare("VAR_WIN") == 0) {
      uint32_t v;
      conf >> v;
      var_win = v;
    } else if (key.compare("FAST_REACT") == 0) {
      uint32_t v;
      conf >> v;
      fast_react = v;
    } else if (key.compare("U_TARGET") == 0) {
      conf >> u_target;
    } else if (key.compare("INT_MULTI") == 0) {
      conf >> int_multi;
    } else if (key.compare("RATE_BOUND") == 0) {
      uint32_t v;
      conf >> v;
      rate_bound = v;
    } else if (key.compare("ACK_HIGH_PRIO") == 0) {
      conf >> ack_high_prio;
    } else if (key.compare("DCTCP_RATE_AI") == 0) {
      conf >> dctcp_rate_ai;
    } else if (key.compare("NIC_TOTAL_PAUSE_TIME") == 0) {
      conf >> nic_total_pause_time;
    } else if (key.compare("NIC_COALESCE_METHOD") == 0) {
      conf >> nic_coalesce_method;
    } else if (key.compare("NACK_GEN_INTERVAL") == 0) {
      conf >> nack_gen_interval;
    } else if (key.compare("ENABLE_PFC") == 0) {
      uint32_t v;
      conf >> v;
      enable_pfc = v;
    } else if (key.compare("PFC_OUTPUT_FILE") == 0) {
      conf >> pfc_output_file;
      pfc_output_file = extend_output_file_name(run_name, pfc_output_file);
    } else if (key.compare("LINK_DOWN") == 0) {
      conf >> link_down_time >> link_down_A >> link_down_B;
    } else if (key.compare("ENABLE_TRACE") == 0) {
      conf >> enable_trace;
    } else if (key.compare("KMAX_MAP") == 0) {
      int n_k;
      conf >> n_k;
      for (int i = 0; i < n_k; i++) {
        uint64_t rate;
        uint32_t k;
        conf >> rate >> k;
        rate2kmax[rate] = k;
      }
    } else if (key.compare("KMIN_MAP") == 0) {
      int n_k;
      conf >> n_k;
      for (int i = 0; i < n_k; i++) {
        uint64_t rate;
        uint32_t k;
        conf >> rate >> k;
        rate2kmin[rate] = k;
      }
    } else if (key.compare("PMAX_MAP") == 0) {
      int n_k;
      conf >> n_k;
      for (int i = 0; i < n_k; i++) {
        uint64_t rate;
        double p;
        conf >> rate >> p;
        rate2pmax[rate] = p;
      }
    } else if (key.compare("BUFFER_SIZE") == 0) {
      conf >> buffer_size;
    } else if (key.compare("SWITCH_FORWARD_DELAY") == 0) {
      conf >> switch_fw_delay;
    } else if (key.compare("QLEN_MON_FILE") == 0) {
      conf >> qlen_mon_file;
      qlen_mon_file = extend_output_file_name(run_name, qlen_mon_file);
      qlen_mon_file = get_output_file_name(network_conf, qlen_mon_file);
    } else if (key.compare("BW_MON_FILE") == 0) {
      conf >> bw_mon_file;
      bw_mon_file = extend_output_file_name(run_name, bw_mon_file);
      bw_mon_file = get_output_file_name(network_conf, bw_mon_file);
    } else if (key.compare("RATE_MON_FILE") == 0) {
      conf >> rate_mon_file;
      rate_mon_file = extend_output_file_name(run_name, rate_mon_file);
      rate_mon_file = get_output_file_name(network_conf, rate_mon_file);
    } else if (key.compare("CNP_MON_FILE") == 0) {
      conf >> cnp_mon_file;
      cnp_mon_file = extend_output_file_name(run_name, cnp_mon_file);
      cnp_mon_file = get_output_file_name(network_conf, cnp_mon_file);
    } else if (key.compare("MON_START") == 0) {
      conf >> mon_start;
    } else if (key.compare("MON_END") == 0) {
      conf >> mon_end;
    } else if (key.compare("QP_MON_INTERVAL") == 0) {
      conf >> qp_mon_interval;
    } else if (key.compare("BW_MON_INTERVAL") == 0) {
      conf >> bw_mon_interval;
    } else if (key.compare("QLEN_MON_INTERVAL") == 0) {
      conf >> qlen_mon_interval;
    } else if (key.compare("MULTI_RATE") == 0) {
      int v;
      conf >> v;
      multi_rate = v;
    } else if (key.compare("SAMPLE_FEEDBACK") == 0) {
      int v;
      conf >> v;
      sample_feedback = v;
    } else if (key.compare("PINT_LOG_BASE") == 0) {
      conf >> pint_log_base;
    } else if (key.compare("PINT_PROB") == 0) {
      conf >> pint_prob;
    } else if (key.compare("RDCQCN_EWMAGAIN") == 0) {
      conf >> r_dcqcn_ewma_gain;
    } else if (key.compare("RDCQCN_F") == 0) {
      uint32_t v;
      conf >> v;
      r_dcqcn_f = v;
    } else if (key.compare("RDCQCN_RATE_UPDATE_DELAY") == 0) {
      conf >> r_dcqcn_rate_u_delay;
    } else if (key.compare("RDCQCN_ALPHA_UPDATE_DELAY") == 0) {
      conf >> r_dcqcn_alpha_u_delay;
    } else if (key.compare("RDCQCN_BYTES_THRESHOLD") == 0) {
      conf >> r_dcqcn_bytes_threshold;
    } else if (key.compare("RDCQCN_CLAMP") == 0) {
      uint32_t v;
      conf >> v;
      r_dcqcn_clamp = v;
    }
    fflush(stdout);
  }
  conf.close();
  return true;
}

inline void SetConfig() {
  Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(pause_time));
  Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(enable_qcn));
  Config::SetDefault("ns3::QbbNetDevice::DynamicThreshold", BooleanValue(use_dynamic_pfc_threshold));

  IntHop::multi = int_multi;
  if (cc_mode == 7) 
    IntHeader::mode = IntHeader::TS;
  else if (cc_mode == 3) 
    IntHeader::mode = IntHeader::NORMAL;
  else if (cc_mode == 10) 
    IntHeader::mode = IntHeader::PINT;
  else
    IntHeader::mode = IntHeader::NONE;

  if (cc_mode == 10) {
    Pint::set_log_base(pint_log_base);
    IntHeader::pint_bytes = Pint::get_n_bytes();
    printf("PINT bits: %d bytes: %d\n", Pint::get_n_bits(),
           Pint::get_n_bytes());
  }
}

inline void SetupNetwork(
    void (*qp_finish)(FILE*, Ptr<RdmaQueuePair>),
    void (*message_finish)(FILE*, Ptr<RdmaQueuePair>, const RdmaQueuePair::RdmaMessage&),
    void (*send_finish)(FILE *, Ptr<RdmaQueuePair>, const RdmaQueuePair::RdmaMessage&),
    void (*recv_finish)(FILE *, Ptr<RdmaRxQueuePair>, const RdmaRxQueuePair::RdmaMessage&)) {
  string gpu_type_str;

  topo_ifs.open(topology_file.c_str());
  flowf.open(flow_file.c_str());
  tracef.open(trace_file.c_str());
  if (!topo_ifs.is_open()) {
    std::cerr << "Unable to open topology file: " << topology_file << std::endl;
    std::cerr << "This error is fatal." << std::endl;
    exit(1);
  }

  topo_ifs >> node_num >> gpus_per_server >> nvswitch_num >> switch_num >> link_num >> gpu_type_str;
  flowf >> flow_num;
  tracef >> trace_num;
  if (gpu_type_str == "A100") {
    gpu_type = GPUType::A100;
  } else if (gpu_type_str == "A800") {
    gpu_type = GPUType::A800;
  } else if (gpu_type_str == "H100") {
    gpu_type = GPUType::H100;
  } else if (gpu_type_str == "H800") {
    gpu_type = GPUType::H800;
  } else {
    gpu_type = GPUType::NONE;
  }

  std::vector<uint32_t> node_type(node_num, 0);
  for (uint32_t i = 0; i < nvswitch_num; i++) {
    uint32_t sid;
    topo_ifs >> sid;
    node_type[sid] = 2;
	}
	for (uint32_t i = 0; i < switch_num; i++) {
		uint32_t sid;
		topo_ifs >> sid;
		node_type[sid] = 1;
	}
	for (uint32_t i = 0; i < node_num; i++) {
		if (node_type[i] == 0) {
		  n.Add(CreateObject<Node>());
		} else if (node_type[i] == 1) {
			Ptr<SwitchNode> sw = CreateObject<SwitchNode>();
			n.Add(sw);
		  sw->SetAttribute("EcnEnabled", BooleanValue(enable_qcn));
		  sw->SetAttribute("PfcEnabled", BooleanValue(enable_pfc));
		  sw->SetAttribute("ForwardDelay", DoubleValue(switch_fw_delay));
		} else if (node_type[i] == 2) {
			Ptr<NVSwitchNode> sw = CreateObject<NVSwitchNode>();
			n.Add(sw);
		}
	}

  NS_LOG_INFO("Create nodes.");
  InternetStackHelper internet;
  internet.Install(n);

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0) {
      serverAddress.resize(i + 1);
      serverAddress[i] = node_id_to_ip(i);
    } else if(n.Get(i)->GetNodeType() == 2) {
      serverAddress.resize(i + 1);
      serverAddress[i] = node_id_to_ip(i);
    }
  }

  NS_LOG_INFO("Create channels.");

  Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
  rem->SetRandomVariable(uv);
  uv->SetStream(50);
  rem->SetAttribute("ErrorRate", DoubleValue(error_rate_per_link));
  rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

  FILE *pfc_file = fopen(pfc_output_file.c_str(), "w");

  QbbHelper qbb;
  Ipv4AddressHelper ipv4;
  for (uint32_t i = 0; i < link_num; i++) {
    uint32_t src, dst;
    std::string data_rate, link_delay;
    double error_rate;
    topo_ifs >> src >> dst >> data_rate >> link_delay >> error_rate;
    Ptr<Node> snode = n.Get(src), dnode = n.Get(dst);
    
    qbb.SetDeviceAttribute("DataRate", StringValue(data_rate));
    qbb.SetChannelAttribute("Delay", StringValue(link_delay));

    if (error_rate > 0) {
      Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
      Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
      rem->SetRandomVariable(uv);
      uv->SetStream(50);
      rem->SetAttribute("ErrorRate", DoubleValue(error_rate));
      rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
      qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
    } else {
      qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
    }

    fflush(stdout);

    NetDeviceContainer d = qbb.Install(snode, dnode);
    if (snode->GetNodeType() == 0 || snode->GetNodeType() == 2) {
      Ptr<Ipv4> ipv4 = snode->GetObject<Ipv4>();
      ipv4->AddInterface(d.Get(0));
      ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[src], Ipv4Mask(0xff000000)));
    }
    if (dnode->GetNodeType() == 0 || dnode->GetNodeType() == 2) {
      Ptr<Ipv4> ipv4 = dnode->GetObject<Ipv4>();
      ipv4->AddInterface(d.Get(1));
      ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[dst], Ipv4Mask(0xff000000)));
    }

    nbr2if[snode][dnode].idx =
        DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
    nbr2if[snode][dnode].up = true;
    nbr2if[snode][dnode].delay =
        DynamicCast<QbbChannel>(
            DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())
            ->GetDelay()
            .GetTimeStep();
    nbr2if[snode][dnode].bw =
        DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
    nbr2if[dnode][snode].idx =
        DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
    nbr2if[dnode][snode].up = true;
    nbr2if[dnode][snode].delay =
        DynamicCast<QbbChannel>(
            DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())
            ->GetDelay()
            .GetTimeStep();
    nbr2if[dnode][snode].bw =
        DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();

    char ipstring[16];
    sprintf(ipstring, "10.%d.%d.0", i / 254 + 1, i % 254 + 1);
    ipv4.SetBase(ipstring, "255.255.255.0");
    ipv4.Assign(d);

    DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext(
        "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(0))));
    DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext(
        "QbbPfc", MakeBoundCallback(&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(1))));
  }

  nic_rate = get_nic_rate(n);
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 1) { 
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
      uint32_t shift = 3; 

      for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
        Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
        uint64_t rate = dev->GetDataRate().GetBitRate();
        NS_ASSERT_MSG(rate2kmin.find(rate) != rate2kmin.end(),
                      "must set kmin for each link speed");
        NS_ASSERT_MSG(rate2kmax.find(rate) != rate2kmax.end(),
                      "must set kmax for each link speed");
        NS_ASSERT_MSG(rate2pmax.find(rate) != rate2pmax.end(),
                      "must set pmax for each link speed");
        sw->m_mmu->ConfigEcn(j, rate2kmin[rate], rate2kmax[rate],
                             rate2pmax[rate]);
        uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())
                             ->GetDelay()
                             .GetTimeStep();
        uint32_t headroom = rate * delay / 8 / 1000000000 * 3 + packet_payload_size * 2;  // BDP + 2 packet;
        sw->m_mmu->ConfigHdrm(j, headroom);
        sw->m_mmu->pfc_a_shift[j] = shift;
        while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0) {
          sw->m_mmu->pfc_a_shift[j]--;
          rate /= 2;
        }
      }
      sw->m_mmu->ConfigNPort(sw->GetNDevices() - 1);
      sw->m_mmu->ConfigBufferSize(buffer_size * 1024 * 1024);
      sw->m_mmu->node_id = sw->GetId();
    } else if (n.Get(i)->GetNodeType() == 2) {
			Ptr<NVSwitchNode> sw = DynamicCast<NVSwitchNode>(n.Get(i));
      uint32_t shift = 3;  //by default 1/
      for (uint32_t j = 1; j < sw->GetNDevices(); j++) {
        Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
        uint64_t rate = dev->GetDataRate().GetBitRate();
        uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())
                             ->GetDelay()
                             .GetTimeStep();
        uint32_t headroom = rate * delay / 8 / 1000000000 * 3;
        sw->m_mmu->ConfigHdrm(j, headroom);
        sw->m_mmu->pfc_a_shift[j] = shift;
        while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0) {
          sw->m_mmu->pfc_a_shift[j]--;
          rate /= 2;
        }
      }
			sw->m_mmu->ConfigNPort(sw->GetNDevices()-1);
			sw->m_mmu->ConfigBufferSize(buffer_size * 1024 * 1024);
			sw->m_mmu->node_id = sw->GetId();
		}
  }

#if ENABLE_QP
  FILE *fct_output = fopen(fct_output_file.c_str(), "w");
  FILE *send_output = fopen(send_output_file.c_str(), "w");
  FILE *recv_output = fopen(recv_output_file.c_str(), "w");

  // Only tolerate out-of-order packets in case of packet spraying
  if (!packet_spraying) {
    max_rdma_out_of_seq = 0;
  }

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0 || n.Get(i)->GetNodeType() == 2) { 
      // create RdmaHw
      Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
      rdmaHw->SetAttribute("Mtu", UintegerValue(packet_payload_size));
      rdmaHw->SetAttribute("CcMode", UintegerValue(cc_mode));
      rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate(rate_ai)));
      rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate(rate_hai)));
      rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(min_rate)));
      rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(l2_chunk_size));
      rdmaHw->SetAttribute("L2AckInterval", UintegerValue(l2_ack_interval));
      rdmaHw->SetAttribute("L2BackToZero", BooleanValue(l2_back_to_zero));
      rdmaHw->SetAttribute("VarWin", BooleanValue(var_win));
      rdmaHw->SetAttribute("RateBound", BooleanValue(rate_bound));
      rdmaHw->SetAttribute("NicCoalesceMethod", StringValue(nic_coalesce_method));
      rdmaHw->SetAttribute("NACKGenerationInterval", DoubleValue(nack_gen_interval));
      rdmaHw->SetAttribute("MaxOutOfSeq", UintegerValue(max_rdma_out_of_seq));

      switch (cc_mode) {
      case 1:
        // MellanoxDcqcn
        rdmaHw->m_cc_configs.emplace_back("AlphaResumInterval", DoubleValue(alpha_resume_interval).Copy());
        rdmaHw->m_cc_configs.emplace_back("RateDecreaseInterval", DoubleValue(rate_decrease_interval).Copy());
        rdmaHw->m_cc_configs.emplace_back("ClampTargetRate", BooleanValue(clamp_target_rate).Copy());
        rdmaHw->m_cc_configs.emplace_back("RPTimer", DoubleValue(rp_timer).Copy());
        rdmaHw->m_cc_configs.emplace_back("EwmaGain", DoubleValue(ewma_gain).Copy());
        rdmaHw->m_cc_configs.emplace_back("FastRecoveryTimes", UintegerValue(fast_recovery_times).Copy());
        break;
      case 3:
        // Hpcc
        rdmaHw->m_cc_configs.emplace_back("FastReact", BooleanValue(fast_react).Copy());
        rdmaHw->m_cc_configs.emplace_back("TargetUtil", DoubleValue(u_target).Copy());
        rdmaHw->m_cc_configs.emplace_back("MiThresh", DoubleValue(mi_thresh).Copy());
        rdmaHw->m_cc_configs.emplace_back("MultiRate", BooleanValue(multi_rate).Copy());
        rdmaHw->m_cc_configs.emplace_back("SampleFeedback ", BooleanValue(sample_feedback).Copy());
        break;
      case 8:
        // Dctcp
        rdmaHw->m_cc_configs.emplace_back("DctcpRateAI", DataRateValue(DataRate(dctcp_rate_ai)).Copy());
        break;
      case 10:
        // HpccPint
        rdmaHw->m_cc_configs.emplace_back("PintProb", DoubleValue(pint_prob).Copy());
        break;
      case 12:
        // RealDcqcn
        rdmaHw->m_cc_configs.emplace_back("EwmaGain", DoubleValue(r_dcqcn_ewma_gain).Copy());
        rdmaHw->m_cc_configs.emplace_back("F", UintegerValue(r_dcqcn_f).Copy());
        rdmaHw->m_cc_configs.emplace_back("RateUpdateDelay", StringValue(r_dcqcn_rate_u_delay).Copy());
        rdmaHw->m_cc_configs.emplace_back("AlphaUpdateDelay", StringValue(r_dcqcn_alpha_u_delay).Copy());
        rdmaHw->m_cc_configs.emplace_back("BytesThreshold", UintegerValue(r_dcqcn_bytes_threshold).Copy());
        rdmaHw->m_cc_configs.emplace_back("Clamp", BooleanValue(r_dcqcn_clamp).Copy());
        break;
      default:;
      }

      Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
      Ptr<Node> node = n.Get(i);
      rdma->SetNode(node);
      rdma->SetRdmaHw(rdmaHw);

      node->AggregateObject(rdma);
      rdma->Init();
      rdma->TraceConnectWithoutContext("QpComplete", MakeBoundCallback(qp_finish, fct_output));
      rdma->TraceConnectWithoutContext("MessageComplete", MakeBoundCallback(message_finish, fct_output));
      rdma->TraceConnectWithoutContext("SendComplete", MakeBoundCallback(send_finish, send_output));
      rdma->TraceConnectWithoutContext("RecvComplete", MakeBoundCallback(recv_finish, recv_output));
    }
  }
#endif

  if (ack_high_prio)
    RdmaEgressQueue::ack_q_idx = 0;
  else
    RdmaEgressQueue::ack_q_idx = 3;

  CalculateRoutes(n);
  SetRoutingEntries();
  //printRoutingEntries();

  maxRtt = maxBdp = 0;
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() != 0)
      continue;
    for (uint32_t j = 0; j < node_num; j++) {
      if (n.Get(j)->GetNodeType() != 0)
        continue;
      uint64_t delay = pairDelay[n.Get(i)][n.Get(j)];
      uint64_t txDelay = pairTxDelay[n.Get(i)][n.Get(j)];
      uint64_t rtt = delay * 2 + txDelay;
      uint64_t bw = pairBw[i][j];
      uint64_t bdp = rtt * bw / 1000000000 / 8;
      pairBdp[n.Get(i)][n.Get(j)] = bdp;
      pairRtt[i][j] = rtt;
      if (bdp > maxBdp)
        maxBdp = bdp;
      if (rtt > maxRtt)
        maxRtt = rtt;
    }
  }
  printf("maxRtt=%lu maxBdp=%lu\n", maxRtt, maxBdp);

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 1) { 
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
      sw->SetAttribute("CcMode", UintegerValue(cc_mode));
      sw->SetAttribute("MaxRtt", UintegerValue(maxRtt));
      sw->SetAttribute("PacketSpraying", BooleanValue(packet_spraying));
    }
  }

  FILE* trace_output = fopen(trace_output_file.c_str(), "w");
  if (trace_output == nullptr) {
    cout << "open trace_output_file failed: " << trace_output_file << endl;
    exit(1);
  }

  if (enable_trace) {
    NodeContainer trace_nodes;
    for (uint32_t i = 0; i < trace_num; i++) {
      uint32_t nid;
      tracef >> nid;
      if (nid >= n.GetN()) {
        continue;
      }
      cout << "Add trace node " << nid << endl;
      trace_nodes = NodeContainer(trace_nodes, n.Get(nid));
    }

    cout << "Enabled tracing for " << trace_nodes.GetN() << " nodes (output: " << trace_output_file << ")" << endl;
    qbb.EnableTracing(trace_output, trace_nodes);
  }

  {
    SimSetting sim_setting;
    for (auto i : nbr2if) {
      for (auto j : i.second) {
        uint16_t node = i.first->GetId();
        uint8_t intf = j.second.idx;
        uint64_t bps = DynamicCast<QbbNetDevice>(i.first->GetDevice(j.second.idx))
            ->GetDataRate()
            .GetBitRate();
        sim_setting.port_speed[node][intf] = bps;
      }
    }
    sim_setting.win = maxBdp;
    sim_setting.Serialize(trace_output);
  }

  NS_LOG_INFO("Create Applications.");

  Time interPacketInterval = Seconds(0.0000005 / 2);
  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 0 || n.Get(i)->GetNodeType() == 2) {
      for (uint32_t j = 0; j < node_num; j++) {
        if (n.Get(j)->GetNodeType() == 0 || n.Get(j)->GetNodeType() == 2) {
          portNumber[i][j] = 10000;
        }
      }
    }
  }
  flow_input.idx = -1;

  topo_ifs.close();
  tracef.close();

  if (link_down_time > 0) {
    Simulator::Schedule(Seconds(2) + MicroSeconds(link_down_time),
                        &TakeDownLink, n, n.Get(link_down_A),
                        n.Get(link_down_B));
  }
}
#endif
