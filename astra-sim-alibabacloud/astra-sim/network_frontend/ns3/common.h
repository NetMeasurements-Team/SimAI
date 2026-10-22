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
#include <ns3/node-list.h>
#include <ns3/nvswitch-node.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/qbb-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <ns3/telemetry-header.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");

// TODO replace with a NetworkConfig class
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
inline bool source_routing = false;
inline bool expected_flow_hit_tracking = false;
inline std::string expected_flow_hit_output_file = "expected_flow_hits.txt";
inline uint32_t expected_flow_hit_group_size = 1;

// Telemetry flow index (claude_reports/TELEMETRY_FLOW_INDEX_PLAN.md).
inline bool telemetry_flow_index = false;
inline std::string telemetry_flow_file = "telemetry_flows.txt";
// TELEMETRY_FLOW_KEY: 0 = 5TUPLE (default, per-channel resolution),
// 1 = PAIR (all channels of a host pair share one index).
inline bool telemetry_flow_key_pair_mode = false;
inline std::string telemetry_index_dump_file = "";
inline bool telemetry_on_acks = true;
inline std::string telemetry_miss_output_file = "";
inline std::string telemetry_switch_counter_file = "";
// The presimulation's flow list, canonically sorted and deduplicated. Only the
// ordering matters: it fixes which flow gets a given switch's index 0 vs 1.
inline std::vector<FlowKey5Tuple> telemetry_flows;
// flow -> [(switchId, index at that switch)], sorted by switchId. Built once at
// setup, never mutated afterwards, and shared read-only by every host -- which
// is what lets step 3 hang a borrowed pointer off each QP with no locking and
// no per-host copy.
inline std::map<FlowKey5Tuple, std::vector<std::pair<uint16_t, uint16_t>>> telemetry_index;
// switchId -> N_s, the number of counters that switch must allocate.
inline std::map<uint16_t, uint32_t> telemetry_switch_cardinality;
inline bool reuse_qps = true;
inline bool var_win = false, fast_react = true;
inline bool rto = false;
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
        if (next->GetNodeType() == 1) {
          delay[next] += switch_fw_delay*1000;
        }
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

inline Ptr<UniformRandomVariable> source_routing_rand;

// Per-packet path sampler for SOURCE_ROUTING. Walks the same shortest-path
// DAG CalculateRoute already built (nextHop/nbr2if -- the exact set today's
// per-switch ECMP hashing draws from, see SRH_PLAN.md), drawing an
// independent uniform choice at every equal-cost branch, and returns the
// ordered list of switch/host *node ids* the packet will actually traverse,
// starting with the first switch out of srcId's NIC and ending with dstId
// itself (each switch resolves the *next* segment's node id to one of its
// own local ports via its m_srNextHop table -- see SwitchNode::GetOutDev).
// No table is cached: calling this fresh for every packet is what lets
// packets of the same flow take different equal-cost paths, and it also
// means a route recomputation (e.g. after a link goes down) is picked up by
// the very next packet with no separate invalidation step.
//
// NOTE: the first element (the first switch out of srcId) is the complete
// path, but is *not* part of the wire-format SRH segment list -- that hop
// is already decided by RdmaHw picking which NIC to send from, not by a
// switch reading an SRH. Callers building the actual SrHeader must drop
// element 0 (`std::vector<uint16_t>(path.begin() + 1, path.end())`); callers
// that want the true, complete path a packet takes -- e.g. the expected-hit
// oracle, RdmaHw::RecordExpectedFlowHits -- use the full return value as-is.
inline std::vector<uint16_t> BuildSourceRoute(uint32_t srcId, uint32_t dstId,
                                              uint32_t firstHopId) {
  std::vector<uint16_t> path;
  Ptr<Node> src = NodeList::GetNode(srcId);
  Ptr<Node> dst = NodeList::GetNode(dstId);
  if (src == dst)
    return path;

  auto pickNext = [](const vector<Ptr<Node>>& candidates) -> Ptr<Node> {
    if (candidates.size() == 1)
      return candidates[0];
    if (!source_routing_rand)
      source_routing_rand = CreateObject<UniformRandomVariable>();
    return candidates[source_routing_rand->GetInteger(0, candidates.size() - 1)];
  };

  auto srcIt = nextHop.find(src);
  if (srcIt == nextHop.end())
    return path;
  auto srcDstIt = srcIt->second.find(dst);
  if (srcDstIt == srcIt->second.end() || srcDstIt->second.empty())
    return path;
  // The first hop is NOT a free choice: the packet leaves through whichever NIC
  // RdmaHw::GetNicIdxOfQp already picked, which fixes the switch it lands on.
  // Sampling it here instead made path[0] a guess -- harmless on a single-ToR
  // topology where there is only one candidate, but wrong about half the time
  // on a dual-ToR one, which mis-credited the first hop in both the telemetry
  // header and the expected-hit oracle. The caller passes the real peer node;
  // we only fall back to sampling if it could not be resolved.
  Ptr<Node> cur;
  if (firstHopId != UINT32_MAX) {
    cur = NodeList::GetNode(firstHopId);
  } else {
    cur = pickNext(srcDstIt->second);
  }
  path.push_back(static_cast<uint16_t>(cur->GetId()));

  while (cur != dst) {
    auto it = nextHop.find(cur);
    if (it == nextHop.end())
      break;
    auto dstIt = it->second.find(dst);
    if (dstIt == it->second.end() || dstIt->second.empty())
      break;
    Ptr<Node> next = pickNext(dstIt->second);
    path.push_back(static_cast<uint16_t>(next->GetId()));
    cur = next;
  }
  return path;
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

// Populates each switch/NVSwitch's node-id -> local-port table used to
// resolve SRH segments (see BuildSourceRoute above and
// SwitchNode::GetOutDev). Built straight from nbr2if, which already has
// every direct adjacency (switch-to-switch and switch-to-host) regardless
// of whether it lies on a shortest path, so it's a strict superset of what
// BuildSourceRoute could ever emit as a segment.
inline void SetSourceRoutingEntries() {
  for (auto& i : nbr2if) {
    Ptr<Node> node = i.first;
    if (node->GetNodeType() != 1 && node->GetNodeType() != 2)
      continue; // only switches/NVSwitches forward via SRH
    for (auto& j : i.second) {
      Ptr<Node> neighbor = j.first;
      uint32_t interface = j.second.idx;
      if (node->GetNodeType() == 1) {
        DynamicCast<SwitchNode>(node)->AddSrNextHopEntry(neighbor->GetId(), interface);
      } else {
        DynamicCast<NVSwitchNode>(node)->AddSrNextHopEntry(neighbor->GetId(), interface);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Telemetry flow index (claude_reports/TELEMETRY_FLOW_INDEX_PLAN.md, step 2).
//
// A presimulation over the same (workload, topology) supplies ONE file: the
// flat list of 5-tuples the workload produces. Nothing about switches is read
// from disk. Which switches a flow can traverse, and what index it holds at
// each of them, are both derived here from `nextHop` -- the shortest-path DAG
// CalculateRoute already built -- so the input stays one small file and the
// index space is a pure function of (flow list, topology), independent of any
// randomness in the presimulation run that produced the list.
//
// Indices are dense PER SWITCH: switch s numbers the flows whose DAG includes
// it 0..N_s-1 with no gaps, so s can address exactly N_s counters. The global
// sort order only decides *which* flow gets s's index 0 versus index 1.
// ---------------------------------------------------------------------------

// Collapses 5-tuples that must share one index.
//
// Always: 0xFD (NACK) folds onto 0xFC (ACK). They are the same logical
// reverse-direction flow of the same QP, differing only in whether the
// receiver is acknowledging or reporting a gap, and a flow list generated from
// fct.txt contains no NACK rows at all -- so without this fold every NACK in a
// live run would miss its index.
//
// In PAIR mode, additionally: ports are zeroed, so every channel between a
// host pair shares one index. That makes the index space immune to the order
// in which entry.h's portNumber[src][dst]++ hands out ports, at the cost of
// per-channel resolution. Whether that costs anything is workload-dependent --
// check the workload's ports-per-pair histogram before choosing it.
inline FlowKey5Tuple NormalizeTelemetryFlowKey(FlowKey5Tuple f) {
  if (f.proto == 0xFD)
    f.proto = 0xFC;
  if (telemetry_flow_key_pair_mode) {
    f.sport = 0;
    f.dport = 0;
  }
  return f;
}

// Every real switch (GetNodeType() == 1) reachable from srcId on the
// shortest-path DAG toward dstId -- i.e. every switch BuildSourceRoute could
// ever return for this source/destination pair, over all of its random
// equal-cost choices. Returned sorted by node id and deduplicated.
//
// This is deliberately the DAG rather than the set a presimulation was
// *observed* to use: spraying makes the observed set a random sample of this
// one, so indexing on observations risks a live-run packet arriving at a
// switch where its flow has no index. NVSwitches and the destination host are
// excluded, matching RdmaHw::RecordExpectedFlowHits' filter.
inline std::vector<uint16_t> TelemetryDagSwitches(uint32_t srcId, uint32_t dstId) {
  std::vector<uint16_t> out;
  if (srcId == dstId)
    return out;
  Ptr<Node> dst = NodeList::GetNode(dstId);

  std::set<uint32_t> seen;
  std::vector<Ptr<Node>> frontier;
  seen.insert(srcId);
  frontier.push_back(NodeList::GetNode(srcId));

  while (!frontier.empty()) {
    Ptr<Node> cur = frontier.back();
    frontier.pop_back();
    if (cur == dst)
      continue;
    auto it = nextHop.find(cur);
    if (it == nextHop.end())
      continue;
    auto dstIt = it->second.find(dst);
    if (dstIt == it->second.end())
      continue;
    for (Ptr<Node> next : dstIt->second) {
      if (!seen.insert(next->GetId()).second)
        continue;
      if (next->GetNodeType() == 1)
        out.push_back(static_cast<uint16_t>(next->GetId()));
      frontier.push_back(next);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Parses the presimulation flow list. Format (see the plan's section 3):
//
//   # free-form comment lines
//   sip,dip,sport,dport,proto
//   0b000001,0b000101,10000,100,17
//
// sip/dip are the same 8-hex-digit encoding the oracle dump and fct.txt use.
// Any line that does not parse as five fields is skipped, which transparently
// absorbs the header line and comments whatever they are called -- 144_flows.txt's
// "srcIP,destIP,..." spelling included. Keys are normalized and deduplicated on
// load, so PAIR mode cannot hand the same key two different indices.
inline bool LoadTelemetryFlows() {
  FILE* f = fopen(telemetry_flow_file.c_str(), "r");
  if (!f) {
    std::cout << "ERROR: could not open TELEMETRY_FLOW_FILE '" << telemetry_flow_file << "'"
              << std::endl;
    return false;
  }
  char line[512];
  uint64_t parsed = 0, skipped = 0;
  telemetry_flows.clear();
  while (fgets(line, sizeof(line), f)) {
    uint32_t sip, dip, sport, dport, proto;
    if (sscanf(line, "%x,%x,%u,%u,%u", &sip, &dip, &sport, &dport, &proto) != 5) {
      skipped++;
      continue;
    }
    FlowKey5Tuple flow{sip, dip, static_cast<uint16_t>(sport), static_cast<uint16_t>(dport),
                       static_cast<uint8_t>(proto)};
    telemetry_flows.push_back(NormalizeTelemetryFlowKey(flow));
    parsed++;
  }
  fclose(f);

  std::sort(telemetry_flows.begin(), telemetry_flows.end());
  telemetry_flows.erase(std::unique(telemetry_flows.begin(), telemetry_flows.end(),
                                    [](const FlowKey5Tuple& a, const FlowKey5Tuple& b) {
                                      return !(a < b) && !(b < a);
                                    }),
                        telemetry_flows.end());

  std::cout << "telemetry: loaded " << parsed << " rows (" << skipped
            << " non-data lines skipped) from " << telemetry_flow_file << " -> "
            << telemetry_flows.size() << " distinct flow keys" << std::endl;
  return !telemetry_flows.empty();
}

// Assigns every flow its per-switch index. One pass over the canonically
// sorted flow list; for each flow, one DAG walk (cached per host pair, since a
// flow and its ACK flow -- and every channel of the same pair -- share one) and
// one post-increment of each of those switches' own counters.
//
// Must run after CalculateRoutes() has filled nextHop.
inline bool BuildTelemetryIndex() {
  telemetry_index.clear();
  telemetry_switch_cardinality.clear();

  std::map<std::pair<uint32_t, uint32_t>, std::vector<uint16_t>> dagCache;
  uint64_t entries = 0, badEndpoints = 0;

  for (const FlowKey5Tuple& flow : telemetry_flows) {
    const uint32_t src = ip_to_node_id(Ipv4Address(flow.sip));
    const uint32_t dst = ip_to_node_id(Ipv4Address(flow.dip));
    if (src >= n.GetN() || dst >= n.GetN() || n.Get(src)->GetNodeType() != 0 ||
        n.Get(dst)->GetNodeType() != 0) {
      badEndpoints++;
      continue;
    }
    const auto pairKey = std::make_pair(src, dst);
    auto it = dagCache.find(pairKey);
    if (it == dagCache.end())
      it = dagCache.emplace(pairKey, TelemetryDagSwitches(src, dst)).first;

    auto& vec = telemetry_index[flow];
    vec.reserve(it->second.size());
    for (uint16_t sw : it->second) {
      uint32_t& next = telemetry_switch_cardinality[sw];
      if (next >= TelemetryHeader::NO_INDEX) {
        std::cout << "ERROR: switch " << sw << " needs more than "
                  << TelemetryHeader::NO_INDEX
                  << " flow indices, which does not fit the telemetry header's 16-bit "
                     "fields. Reduce the flow list, or widen TelemetryHeader's fields."
                  << std::endl;
        return false;
      }
      vec.emplace_back(sw, static_cast<uint16_t>(next++));
      entries++;
    }
  }

  if (badEndpoints)
    std::cout << "WARNING: telemetry: " << badEndpoints
              << " flow rows named an endpoint that is not a host node in this topology; "
                 "skipped (is the flow list from a different topology?)"
              << std::endl;

  // Report only what carries information. A mean would be entries/switches,
  // both already on this line; a minimum cannot even be zero, since a switch no
  // flow reaches never enters the map. What matters is the largest array any
  // switch must allocate (the number that would hit the header's 16-bit limit)
  // and how much of the topology the flow list actually covers.
  uint32_t maxN = 0;
  for (const auto& e : telemetry_switch_cardinality)
    maxN = std::max(maxN, e.second);
  uint32_t totalSwitches = 0;
  for (uint32_t i = 0; i < n.GetN(); i++)
    if (n.Get(i)->GetNodeType() == 1)
      totalSwitches++;

  std::cout << "telemetry: indexed " << telemetry_index.size() << " flows across "
            << telemetry_switch_cardinality.size() << " of " << totalSwitches
            << " switches, " << entries
            << " (flow,switch) entries; largest per-switch counter array = " << maxN
            << std::endl;
  if (telemetry_switch_cardinality.size() < totalSwitches) {
    std::cout << "WARNING: telemetry: "
              << (totalSwitches - telemetry_switch_cardinality.size())
              << " switches lie on no flow's path. Expected for a workload that does not "
                 "use the whole fabric, but also what a flow list from a *different* "
                 "topology looks like -- worth checking." << std::endl;
  }
  return true;
}

// Optional debug dump of the whole index, one row per (switch, flow), sorted by
// switch then index. This is what the plan's section 11.3 check diffs against a
// per-switch presimulation file such as 144_flows.txt: the row count for a given
// switch_node_id is that switch's N_s, and any excess over the observed count is
// a counter that DAG membership allocates but the run never increments.
inline void DumpTelemetryIndex() {
  if (telemetry_index_dump_file.empty())
    return;
  std::map<uint16_t, std::vector<std::pair<uint16_t, const FlowKey5Tuple*>>> bySwitch;
  for (const auto& e : telemetry_index)
    for (const auto& si : e.second)
      bySwitch[si.first].emplace_back(si.second, &e.first);

  FILE* f = fopen(telemetry_index_dump_file.c_str(), "w");
  if (!f) {
    std::cout << "WARNING: could not open TELEMETRY_INDEX_DUMP_FILE '"
              << telemetry_index_dump_file << "'" << std::endl;
    return;
  }
  fprintf(f, "switch_node_id,index,sip,dip,sport,dport,proto\n");
  for (auto& e : bySwitch) {
    std::sort(e.second.begin(), e.second.end(),
              [](const std::pair<uint16_t, const FlowKey5Tuple*>& a,
                 const std::pair<uint16_t, const FlowKey5Tuple*>& b) { return a.first < b.first; });
    for (const auto& row : e.second)
      fprintf(f, "%u,%u,%08x,%08x,%u,%u,%u\n", e.first, row.first, row.second->sip,
              row.second->dip, row.second->sport, row.second->dport, row.second->proto);
  }
  fclose(f);
  std::cout << "telemetry: wrote index dump to " << telemetry_index_dump_file << std::endl;
}

// Loads the flow list and builds the index. Returns false if telemetry is
// enabled but unusable, so the caller can fail loudly rather than run a whole
// simulation that silently stamps nothing.
inline bool SetupTelemetryIndex() {
  if (!telemetry_flow_index)
    return true;
  if (!LoadTelemetryFlows())
    return false;
  if (!BuildTelemetryIndex())
    return false;
  DumpTelemetryIndex();
  return true;
}

// Registered on every host's RdmaHw as its TelemetryIndexCallback. Returns a
// borrowed pointer into telemetry_index -- valid for the whole run, since that
// map is built before Simulator::Run() and never mutated afterwards -- or
// nullptr when the presimulation's flow list did not contain this flow.
// Callers resolve this once per QP, never per packet.
// End-of-run dump of the switch-side counters, one row per (switch, index),
// alongside the 5-tuple that index belongs to so it can be joined directly
// against the host-side oracle without a second file. This is the closed-loop
// check of section 11.1: counted_packets here must equal the oracle's
// expected_packet_count for the same (switch, flow).
inline void DumpTelemetrySwitchCounters() {
  if (!telemetry_flow_index || telemetry_switch_counter_file.empty())
    return;
  // switch -> index -> flow, inverted from the index we already built.
  std::map<uint16_t, std::map<uint16_t, const FlowKey5Tuple*>> bySwitch;
  for (const auto& e : telemetry_index)
    for (const auto& si : e.second)
      bySwitch[si.first][si.second] = &e.first;

  FILE* f = fopen(telemetry_switch_counter_file.c_str(), "w");
  if (!f) {
    std::cout << "WARNING: could not open TELEMETRY_SWITCH_COUNTER_FILE '"
              << telemetry_switch_counter_file << "'" << std::endl;
    return;
  }
  fprintf(f, "switch_node_id,index,sip,dip,sport,dport,proto,counted_packets\n");
  uint64_t noIndex = 0, outOfRange = 0, noHeader = 0, counted = 0, srPtrBad = 0, srSelfBad = 0;
  for (uint32_t i = 0; i < n.GetN(); i++) {
    Ptr<Node> node = n.Get(i);
    if (node->GetNodeType() != 1)
      continue;
    Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(node);
    const auto& counters = sw->GetTelemetryCounters();
    noIndex += sw->GetTelemetryNoIndex();
    outOfRange += sw->GetTelemetryOutOfRange();
    noHeader += sw->GetTelemetryNoHeader();
    srPtrBad += sw->GetSrPtrOutOfRange();
    srSelfBad += sw->GetSrSelfMismatch();
    auto sit = bySwitch.find(static_cast<uint16_t>(node->GetId()));
    for (uint32_t k = 0; k < counters.size(); k++) {
      counted += counters[k];
      const FlowKey5Tuple* flow =
          (sit == bySwitch.end()) ? nullptr : (sit->second.count(k) ? sit->second.at(k) : nullptr);
      if (flow)
        fprintf(f, "%u,%u,%08x,%08x,%u,%u,%u,%lu\n", node->GetId(), k, flow->sip, flow->dip,
                flow->sport, flow->dport, flow->proto, counters[k]);
      else
        fprintf(f, "%u,%u,,,,,,%lu\n", node->GetId(), k, counters[k]);
    }
  }
  fclose(f);
  std::cout << "telemetry: switches counted " << counted << " packets; " << noIndex
            << " had no index for their hop, " << outOfRange << " were out of range, "
            << noHeader << " source-routed packets carried no telemetry header" << std::endl;
  if (srSelfBad) {
    std::cout << "WARNING: telemetry: " << srSelfBad
              << " packets arrived at a switch whose own node id is not the segment at "
                 "segs[ptr-1], i.e. the header describes a route the packet is not on. "
                 "That means the first hop the sender predicted differs from the one its "
                 "NIC actually used." << std::endl;
  }
  if (srPtrBad) {
    std::cout << "WARNING: telemetry: " << srPtrBad
              << " packets arrived with an SRH pointer past the end of their segment list "
                 "and were DROPPED. Before this was checked, that index read past the "
                 "segment array into adjacent header memory, which can be forwarded to an "
                 "arbitrary port and loop." << std::endl;
  }
  std::cout << "telemetry: wrote switch counters to " << telemetry_switch_counter_file
            << std::endl;
}

inline const std::vector<std::pair<uint16_t, uint16_t>>* TelemetryIndexLookup(uint32_t sip,
                                                                              uint32_t dip,
                                                                              uint16_t sport,
                                                                              uint16_t dport,
                                                                              uint8_t proto) {
  const FlowKey5Tuple key =
      NormalizeTelemetryFlowKey(FlowKey5Tuple{sip, dip, sport, dport, proto});
  auto it = telemetry_index.find(key);
  return it == telemetry_index.end() ? nullptr : &it->second;
}

// End-of-run diagnostics: how many packets were stamped, how many were not, and
// which 5-tuples were missing from the flow list entirely. A miss means the
// presimulation and the live run disagree about which flows exist -- the one
// real correctness risk in this design -- so it is counted and reported rather
// than asserted on. Must be called after Simulator::Destroy(), like the oracle
// dump it sits next to.
inline void DumpTelemetryStats() {
  if (!telemetry_flow_index)
    return;
  uint64_t stamped = 0, unstamped = 0;
  std::map<FlowKey5Tuple, uint64_t> misses;
  for (uint32_t i = 0; i < n.GetN(); i++) {
    Ptr<Node> node = n.Get(i);
    if (node->GetNodeType() != 0)
      continue;
    Ptr<RdmaDriver> rdma = node->GetObject<RdmaDriver>();
    if (!rdma)
      continue;
    stamped += rdma->m_rdma->GetTelemetryStamped();
    unstamped += rdma->m_rdma->GetTelemetryUnstamped();
    for (const auto& e : rdma->m_rdma->GetTelemetryMisses())
      misses[e.first] += e.second;
  }
  std::cout << "telemetry: stamped " << stamped << " packets, left " << unstamped
            << " unstamped (" << misses.size() << " distinct flows missing from the flow list)"
            << std::endl;
  if (misses.empty() || telemetry_miss_output_file.empty())
    return;
  FILE* f = fopen(telemetry_miss_output_file.c_str(), "w");
  if (!f) {
    std::cout << "WARNING: could not open TELEMETRY_MISS_OUTPUT_FILE '"
              << telemetry_miss_output_file << "'" << std::endl;
    return;
  }
  fprintf(f, "sip,dip,sport,dport,proto,unstamped_packets\n");
  for (const auto& e : misses)
    fprintf(f, "%08x,%08x,%u,%u,%u,%lu\n", e.first.sip, e.first.dip, e.first.sport,
            e.first.dport, e.first.proto, e.second);
  fclose(f);
  std::cout << "telemetry: wrote missing-flow report to " << telemetry_miss_output_file
            << std::endl;
}

// Derives one host-group's output path from the single configured base path.
// With EXPECTED_FLOW_HIT_GROUP_SIZE at its default of 1, startId == endId and
// this produces the original per-host name, e.g. "expected_flow_hits.txt" ->
// "expected_flow_hits.host42.txt"; a group of more than one host instead
// produces a range name, e.g. "expected_flow_hits.hosts0-7.txt". Used
// instead of a directory-based config key so the files land next to the
// other already-configured output files, with no new "create this directory
// if missing" step needed.
inline std::string PerHostOutputPath(const std::string& base, uint32_t startId, uint32_t endId) {
  size_t dot = base.find_last_of('.');
  std::string stem = (dot == std::string::npos) ? base : base.substr(0, dot);
  std::string ext = (dot == std::string::npos) ? "" : base.substr(dot);
  if (startId == endId)
    return stem + ".host" + std::to_string(startId) + ext;
  return stem + ".hosts" + std::to_string(startId) + "-" + std::to_string(endId) + ext;
}

// End-of-simulation dump of the expected-switch-hit oracle (see
// claude_reports/FLOW_HIT_ORACLE_PLAN.md). Hosts are bucketed into groups of
// EXPECTED_FLOW_HIT_GROUP_SIZE consecutive node ids (host node ids are
// always a contiguous 0..hostCount-1 range in this codebase's topology
// convention -- switches/NVSwitches occupy the ids above that range, see
// SetupNetwork's topology parsing), and each group writes one shared file.
// Merging multiple hosts' maps into one never actually combines two hosts'
// numbers under the same key: a FlowKey5Tuple's sip is always the
// contributing host's own address (data packets: the sender; ACK/NACK: the
// acking host, after RdmaHw's direction reversal), so no two different
// hosts ever produce the same key -- grouping only changes which file a
// host's rows land in, and the existing sip column still tells you which
// host any given row came from. Must be called after Simulator::Destroy(),
// once every MTP worker thread has been joined, so every host's map is
// quiescent.
inline void DumpExpectedFlowHits() {
  uint32_t groupSize = expected_flow_hit_group_size == 0 ? 1 : expected_flow_hit_group_size;

  // groupId -> switchId -> flow -> count, merged from every host in that group.
  std::map<uint32_t, std::map<uint16_t, std::map<FlowKey5Tuple, uint64_t>>> groups;

  for (uint32_t i = 0; i < n.GetN(); i++) {
    Ptr<Node> node = n.Get(i);
    if (node->GetNodeType() != 0)
      continue; // hosts only
    Ptr<RdmaDriver> rdma = node->GetObject<RdmaDriver>();
    if (!rdma)
      continue;
    const auto& hits = rdma->m_rdma->GetExpectedFlowHits();
    if (hits.empty())
      continue; // this host never sent a source-routed packet

    uint32_t groupId = node->GetId() / groupSize;
    auto& group = groups[groupId];
    for (auto& switchEntry : hits)
      for (auto& flowEntry : switchEntry.second)
        group[switchEntry.first][flowEntry.first] += flowEntry.second;
  }

  for (auto& groupEntry : groups) {
    uint32_t groupId = groupEntry.first;
    uint32_t startId = groupId * groupSize;
    uint32_t endId = startId + groupSize - 1;
    std::string path = PerHostOutputPath(expected_flow_hit_output_file, startId, endId);
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
      std::cout << "WARNING: could not open expected-flow-hit output file for hosts " << startId
                << "-" << endId << std::endl;
      continue;
    }
    fprintf(f, "switch_node_id,sip,dip,sport,dport,proto,expected_packet_count\n");
    for (auto& switchEntry : groupEntry.second) {
      uint16_t switchId = switchEntry.first;
      for (auto& flowEntry : switchEntry.second) {
        const FlowKey5Tuple& flow = flowEntry.first;
        uint64_t count = flowEntry.second;
        fprintf(f, "%u,%08x,%08x,%u,%u,%u,%lu\n", switchId, flow.sip, flow.dip, flow.sport,
                flow.dport, flow.proto, count);
      }
    }
    fclose(f);
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
    } else if (key.compare("ENABLE_SOURCE_ROUTING") == 0) {
      uint32_t v;
      conf >> v;
      source_routing = v;
    } else if (key.compare("ENABLE_EXPECTED_FLOW_HIT_TRACKING") == 0) {
      uint32_t v;
      conf >> v;
      expected_flow_hit_tracking = v;
    } else if (key.compare("EXPECTED_FLOW_HIT_OUTPUT_FILE") == 0) {
      conf >> expected_flow_hit_output_file;
      expected_flow_hit_output_file = extend_output_file_name(run_name, expected_flow_hit_output_file);
    } else if (key.compare("EXPECTED_FLOW_HIT_GROUP_SIZE") == 0) {
      conf >> expected_flow_hit_group_size;
    } else if (key.compare("ENABLE_TELEMETRY_FLOW_INDEX") == 0) {
      uint32_t v;
      conf >> v;
      telemetry_flow_index = v;
    } else if (key.compare("TELEMETRY_FLOW_FILE") == 0) {
      conf >> telemetry_flow_file;
    } else if (key.compare("TELEMETRY_FLOW_KEY") == 0) {
      std::string v;
      conf >> v;
      telemetry_flow_key_pair_mode = (v == "PAIR");
      if (v != "PAIR" && v != "5TUPLE") {
        std::cout << "WARNING: TELEMETRY_FLOW_KEY '" << v
                  << "' is neither 5TUPLE nor PAIR, defaulting to 5TUPLE." << std::endl;
      }
    } else if (key.compare("TELEMETRY_INDEX_DUMP_FILE") == 0) {
      conf >> telemetry_index_dump_file;
      telemetry_index_dump_file = extend_output_file_name(run_name, telemetry_index_dump_file);
    } else if (key.compare("TELEMETRY_ON_ACKS") == 0) {
      uint32_t v;
      conf >> v;
      telemetry_on_acks = v;
    } else if (key.compare("TELEMETRY_MISS_OUTPUT_FILE") == 0) {
      conf >> telemetry_miss_output_file;
      telemetry_miss_output_file = extend_output_file_name(run_name, telemetry_miss_output_file);
    } else if (key.compare("TELEMETRY_SWITCH_COUNTER_FILE") == 0) {
      conf >> telemetry_switch_counter_file;
      telemetry_switch_counter_file =
          extend_output_file_name(run_name, telemetry_switch_counter_file);
    } else if (key.compare("GLOBAL_T") == 0) {
      conf >> global_t;
      global_t = 1;
    } else if (key.compare("MI_THRESH") == 0) {
      conf >> mi_thresh;
    } else if (key.compare("VAR_WIN") == 0) {
      uint32_t v;
      conf >> v;
      var_win = v;
    } else if (key.compare("RTO") == 0) {
      uint32_t v;
      conf >> v;
      rto = v;
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
  if (expected_flow_hit_tracking && !source_routing) {
    std::cout << "WARNING: ENABLE_EXPECTED_FLOW_HIT_TRACKING is set but "
                 "ENABLE_SOURCE_ROUTING is not -- hosts never learn a packet's "
                 "path without source routing, so no oracle files will be produced."
              << std::endl;
  }
  if (telemetry_flow_index && !source_routing) {
    std::cout << "WARNING: ENABLE_TELEMETRY_FLOW_INDEX is set but ENABLE_SOURCE_ROUTING "
                 "is not -- the telemetry header rides on the SRH and is only attached to "
                 "source-routed packets, so nothing will be stamped."
              << std::endl;
  }
  if (telemetry_flow_index && !reuse_qps) {
    std::cout << "WARNING: ENABLE_TELEMETRY_FLOW_INDEX is set with REUSE_QPS 0 -- every "
                 "message then burns a fresh source port, so the set of 5-tuples grows "
                 "without bound and cannot match a presimulation's flow list. Use "
                 "REUSE_QPS 1, or TELEMETRY_FLOW_KEY PAIR."
              << std::endl;
  }
  if (expected_flow_hit_group_size == 0) {
    std::cout << "WARNING: EXPECTED_FLOW_HIT_GROUP_SIZE is 0, treating as 1 "
                 "(one file per host)."
              << std::endl;
  }
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
		  sw->SetAttribute("AckHighPrio", UintegerValue(1));
		  sw->SetAttribute("SourceRouting", BooleanValue(source_routing));
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

  CalculateRoutes(n);
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
        // uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())
        //                      ->GetDelay()
        //                      .GetTimeStep();
        // The "+ 2 packets" of PFC headroom must be TWO REAL PACKETS, not two
        // payloads: a packet on the wire also carries ppp+ip+udp+SeqTs+INT, and
        // under source routing an SrHeader, and under telemetry a
        // TelemetryHeader on top of that. Sizing this from packet_payload_size
        // alone under-reserves by the whole header stack, which on a workload
        // already running at the edge of BUFFER_SIZE is enough to tip the switch
        // into overflow. See the OPEN ISSUE section of
        // claude_reports/TELEMETRY_FLOW_INDEX_PLAN.md.
        uint32_t maxPktBytes = packet_payload_size + CustomHeader::GetStaticWholeHeaderSize();
        if (source_routing)
          maxPktBytes += SrHeader::GetStaticSize();
        if (telemetry_flow_index)
          maxPktBytes += TelemetryHeader::GetStaticSize();
        uint32_t headroom = rate * maxRtt / 8 / 1000000000 + maxPktBytes * 2;  // BDP + 2 packets
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
        uint32_t headroom = rate * delay / 8 / 1000000000 * 4;
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
      rdmaHw->SetAttribute("EnableRto", BooleanValue(rto));
      rdmaHw->SetAttribute("SourceRouting", BooleanValue(source_routing));
      rdmaHw->SetSourceRouteCb(MakeCallback(&BuildSourceRoute));
      rdmaHw->SetAttribute("ExpectedFlowHitTracking", BooleanValue(expected_flow_hit_tracking));
      rdmaHw->SetAttribute("TelemetryFlowIndex", BooleanValue(telemetry_flow_index));
      rdmaHw->SetAttribute("TelemetryOnAcks", BooleanValue(telemetry_on_acks));
      rdmaHw->SetTelemetryIndexCb(MakeCallback(&TelemetryIndexLookup));

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

  SetRoutingEntries();
  SetSourceRoutingEntries();
  // Needs nextHop, which CalculateRoutes filled above.
  if (!SetupTelemetryIndex()) {
    std::cout << "ERROR: telemetry flow index could not be built; aborting rather than "
                 "running a simulation that stamps nothing."
              << std::endl;
    exit(1);
  }
  //printRoutingEntries();

  for (uint32_t i = 0; i < node_num; i++) {
    if (n.Get(i)->GetNodeType() == 1) { 
      Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
      sw->SetAttribute("CcMode", UintegerValue(cc_mode));
      sw->SetAttribute("MaxRtt", UintegerValue(maxRtt));
      sw->SetAttribute("PacketSpraying", BooleanValue(packet_spraying));
      sw->SetAttribute("SourceRouting", BooleanValue(source_routing));
      // Each switch allocates exactly N_s counters -- the whole point of the
      // per-switch dense index. Zero (a switch no flow reaches, or telemetry
      // off) leaves counting disabled on that switch.
      if (telemetry_flow_index && !telemetry_switch_counter_file.empty()) {
        auto it = telemetry_switch_cardinality.find(static_cast<uint16_t>(n.Get(i)->GetId()));
        uint32_t ns = it == telemetry_switch_cardinality.end() ? 0 : it->second;
        sw->SetAttribute("TelemetryCounterCount", UintegerValue(ns));
      }
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
