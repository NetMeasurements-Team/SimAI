/*
* Copyright (c) 2024, Alibaba Group;
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "CollectiveGraph.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>

#include "SimCCL/mock/MockNcclLog.h"

namespace AstraSim {

namespace {

constexpr int kEvents = 6;
constexpr const char* kEventNames[kEvents] = {"posted", "released", "started", "sent", "arrived", "acked"};

struct FlowTimes {
  std::array<Tick, kEvents> tick{};
  std::array<bool, kEvents> set{};
};

struct GraphData {
  std::map<int, MockNccl::SingleFlow> flow_details;
  std::map<int, FlowTimes> times;
  int expected_ranks = 0;    // ranks of the group
  int finished_ranks = 0;
  int outstanding_acks = 0;  // released flows not acked yet (network events only)
};

using NetKey = std::tuple<int, int, int, int>;  // src, dst, flow_id, tag

// All state behind one lock: graph bookkeeping is cheap and only active with collective-graph on.
// Graphs still open at exit are written by the destructor.
struct Registry {
  std::mutex mtx;
  std::map<std::string, GraphData> graphs;
  std::map<NetKey, std::deque<std::string>> pending_start;
  std::map<NetKey, std::deque<std::string>> pending_arrival;
  std::map<NetKey, std::deque<std::string>> pending_ack;
  bool network_events = false;
  std::atomic<int> in_flight{0};  // fast path for the frontend hooks
  ~Registry();
};

Registry& registry() {
  static Registry r;
  return r;
}

std::string join(const std::vector<int>& v, char sep) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += sep;
    s += std::to_string(v[i]);
  }
  return s;
}

enum EventIndex { kPosted, kReleased, kStarted, kSent, kArrived, kAcked };

// Absolute time of an event, or "-" if it was not recorded.
std::string abs_str(const FlowTimes& t, int e) {
  return t.set[e] ? std::to_string(t.tick[e]) : std::string("-");
}

// collective_graph_<key>.csv: one row per flow, absolute times (for scripts).
void write_csv(const std::string& path, const GraphData& graph) {
  std::ofstream csv(path);
  if (!csv.is_open()) {
    std::cerr << "CollectiveGraph: failed to open " << path << " for writing" << std::endl;
    return;
  }
  static const FlowTimes kNoTimes{};
  csv << "flow_id,src,dest,channel_id,chunk_id,chunk_count,flow_size,conn_type,parent_flow_ids,child_flow_ids";
  for (const char* name : kEventNames) csv << ',' << name;
  csv << '\n';
  for (const auto& [flow_id, flow] : graph.flow_details) {
    auto it = graph.times.find(flow_id);
    const FlowTimes& t = it != graph.times.end() ? it->second : kNoTimes;
    csv << flow_id << ',' << flow.src << ',' << flow.dest << ',' << flow.channel_id << ',' << flow.chunk_id << ','
        << flow.chunk_count << ',' << flow.flow_size << ',' << flow.conn_type << ','
        << join(flow.parent_flow_id, ';') << ',' << join(flow.child_flow_id, ';');
    for (int e = 0; e < kEvents; ++e) csv << ',' << (t.set[e] ? std::to_string(t.tick[e]) : std::string());
    csv << '\n';
  }
}

// collective_graph_<key>.txt: one line per flow for reading, grouped by chunk (the path of each
// piece of data, hop by hop), with the absolute time of every event.
void write_txt(const std::string& path, const std::string& key, const GraphData& graph, bool complete) {
  std::ofstream txt(path);
  if (!txt.is_open()) {
    std::cerr << "CollectiveGraph: failed to open " << path << " for writing" << std::endl;
    return;
  }
  static const FlowTimes kNoTimes{};
  auto times_of = [&](int flow_id) -> const FlowTimes& {
    auto it = graph.times.find(flow_id);
    return it != graph.times.end() ? it->second : kNoTimes;
  };

  // Chunk by chunk; inside a chunk in posting order (= hop order along its path).
  std::vector<int> order;
  order.reserve(graph.flow_details.size());
  Tick first = 0, last = 0;
  bool any_posted = false, any_acked = false;
  for (const auto& [flow_id, flow] : graph.flow_details) {
    order.push_back(flow_id);
    const FlowTimes& t = times_of(flow_id);
    if (t.set[kPosted] && (!any_posted || t.tick[kPosted] < first)) first = t.tick[kPosted], any_posted = true;
    if (t.set[kAcked] && (!any_acked || t.tick[kAcked] > last)) last = t.tick[kAcked], any_acked = true;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const auto& fa = graph.flow_details.at(a);
    const auto& fb = graph.flow_details.at(b);
    if (fa.chunk_id != fb.chunk_id) return fa.chunk_id < fb.chunk_id;
    const FlowTimes& ta = times_of(a);
    const FlowTimes& tb = times_of(b);
    if (ta.set[kPosted] != tb.set[kPosted]) return ta.set[kPosted];  // unposted flows last
    if (ta.set[kPosted] && ta.tick[kPosted] != tb.tick[kPosted]) return ta.tick[kPosted] < tb.tick[kPosted];
    return a < b;
  });

  txt << "# collective " << key << (complete ? "" : "  (INCOMPLETE)") << "\n";
  txt << "# ranks " << graph.expected_ranks << ", flows " << graph.flow_details.size();
  if (any_posted) txt << ", first posted " << first << " ns";
  if (any_acked) txt << ", last acked " << last << " ns";
  if (any_posted && any_acked) txt << ", duration " << (last - first) << " ns";
  txt << "\n#\n";
  txt << "# times in ns:\n";
  txt << "#   posted    dependencies met, the model decides to send\n";
  txt << "#   released  MemBus done (endpoint-delay), handed to the network\n";
  txt << "#   started   the QP starts this message (after AS_SEND_LAT and the messages ahead on the same QP)\n";
  txt << "#   sent      last packet left the NIC\n";
  txt << "#   arrived   whole flow at the destination\n";
  txt << "#   acked     ACK of the whole flow back at the sender\n";
  txt << "# parents: flow ids this flow waits for;  '-' = not recorded\n";
  txt << "#\n";

  auto row = [&](std::ostream& os, const std::string& flow, const std::string& ch, const std::string& chunk,
                 const std::string& link, const std::string& bytes, const std::string& type,
                 const std::array<std::string, kEvents>& times, const std::string& parents) {
    os << std::right << std::setw(7) << flow << std::setw(4) << ch << std::setw(6) << chunk << "  "
       << std::left << std::setw(10) << link << std::right << std::setw(11) << bytes << "  " << std::left
       << std::setw(9) << type << std::right;
    for (const auto& time : times) os << std::setw(12) << time;
    os << "  " << parents << "\n";
  };
  std::array<std::string, kEvents> names;
  for (int e = 0; e < kEvents; ++e) names[e] = kEventNames[e];
  row(txt, "#  flow", "ch", "chunk", "src->dst", "bytes", "type", names, "parents");

  int current_chunk = -1;
  bool first_line = true;
  for (int flow_id : order) {
    const auto& flow = graph.flow_details.at(flow_id);
    const FlowTimes& t = times_of(flow_id);
    if (!first_line && flow.chunk_id != current_chunk) txt << "\n";
    first_line = false;
    current_chunk = flow.chunk_id;
    const std::string parents = flow.parent_flow_id.empty() ? "-" : join(flow.parent_flow_id, ',');
    std::array<std::string, kEvents> times;
    for (int e = 0; e < kEvents; ++e) times[e] = abs_str(t, e);
    row(txt, std::to_string(flow_id), std::to_string(flow.channel_id), std::to_string(flow.chunk_id),
        std::to_string(flow.src) + "->" + std::to_string(flow.dest), std::to_string(flow.flow_size),
        flow.conn_type, times, parents);
  }
}

void write_graph(const std::string& key, const GraphData& graph, bool complete) {
  const std::string base = "collective_graph_" + key + (complete ? "" : "_incomplete");
  write_txt(base + ".txt", key, graph, complete);
  write_csv(base + ".csv", graph);
}

// Must be called with the registry lock held. Writes and drops the graph once it is complete.
bool maybe_write(Registry& r, std::map<std::string, GraphData>::iterator it, int rank) {
  GraphData& graph = it->second;
  if (graph.expected_ranks <= 0 || graph.finished_ranks < graph.expected_ranks) return false;
  if (r.network_events && graph.outstanding_acks > 0) return false;
  write_graph(it->first, graph, true);
  MockNcclLog::getInstance()->writeLog(
      NcclLogLevel::INFO, "ID %d: Collective graph saved to collective_graph_%s.txt/.csv", rank, it->first.c_str());
  r.graphs.erase(it);
  return true;
}

void set_time(GraphData& graph, int flow_id, CollectiveGraph::Event event, Tick tick) {
  FlowTimes& t = graph.times[flow_id];
  t.tick[static_cast<int>(event)] = tick;
  t.set[static_cast<int>(event)] = true;
}

Registry::~Registry() {
  std::lock_guard<std::mutex> lg(mtx);
  for (const auto& [key, graph] : graphs) {
    write_graph(key, graph, false);
    std::cerr << "CollectiveGraph: collective " << key << " did not complete (" << graph.finished_ranks << "/"
              << graph.expected_ranks << " ranks finished, " << graph.outstanding_acks
              << " flows not acked); written as collective_graph_" << key << "_incomplete.txt/.csv" << std::endl;
  }
  graphs.clear();
}

} // namespace

void CollectiveGraph::begin(const std::string& key, int expected_ranks) {
  if (key.empty()) return;
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  r.graphs[key].expected_ranks = expected_ranks;  // same value from every rank
}

void CollectiveGraph::add_flows(const std::string& key, const MockNccl::FlowModels& flows) {
  if (key.empty()) return;
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  auto it = r.graphs.find(key);
  if (it == r.graphs.end()) return;  // not started, or already written
  for (const auto& flow_pair : flows) {
    it->second.flow_details.try_emplace(flow_pair.second.flow_id, flow_pair.second);
  }
}

void CollectiveGraph::record(const std::string& key, int flow_id, Event event, Tick tick) {
  if (key.empty()) return;
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  auto it = r.graphs.find(key);
  if (it != r.graphs.end()) set_time(it->second, flow_id, event, tick);
}

void CollectiveGraph::released(const std::string& key, int src, int dst, int flow_id, int tag, Tick tick) {
  if (key.empty()) return;
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  auto it = r.graphs.find(key);
  if (it == r.graphs.end()) return;
  set_time(it->second, flow_id, Event::Released, tick);
  if (!r.network_events) return;
  const NetKey nk{src, dst, flow_id, tag};
  r.pending_start[nk].push_back(key);
  r.pending_arrival[nk].push_back(key);
  r.pending_ack[nk].push_back(key);
  ++it->second.outstanding_acks;
  ++r.in_flight;
}

void CollectiveGraph::enable_network_events() {
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  r.network_events = true;
}

void CollectiveGraph::network_event(int src, int dst, int flow_id, int tag, Event event, Tick tick) {
  Registry& r = registry();
  if (r.in_flight.load(std::memory_order_relaxed) == 0) return;  // no graph waiting for this
  if (event != Event::Started && event != Event::Arrived && event != Event::Acked) return;
  std::lock_guard<std::mutex> lg(r.mtx);
  auto& pending = event == Event::Started   ? r.pending_start
                  : event == Event::Arrived ? r.pending_arrival
                                            : r.pending_ack;
  auto p = pending.find(NetKey{src, dst, flow_id, tag});
  if (p == pending.end()) return;  // not a recorded flow (or a later stripe of a started flow)
  const std::string key = p->second.front();  // messages of a QP start and complete in order
  p->second.pop_front();
  if (p->second.empty()) pending.erase(p);
  auto it = r.graphs.find(key);
  if (it == r.graphs.end()) return;
  set_time(it->second, flow_id, event, tick);
  if (event == Event::Acked) {
    --it->second.outstanding_acks;
    --r.in_flight;
    maybe_write(r, it, src);
  }
}

bool CollectiveGraph::rank_finished(const std::string& key, int rank) {
  if (key.empty()) return false;
  Registry& r = registry();
  std::lock_guard<std::mutex> lg(r.mtx);
  auto it = r.graphs.find(key);
  if (it == r.graphs.end()) return false;
  ++it->second.finished_ranks;
  return maybe_write(r, it, rank);
}

} // namespace AstraSim
