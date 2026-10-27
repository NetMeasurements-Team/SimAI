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

#ifndef __COLLECTIVE_GRAPH_HH__
#define __COLLECTIVE_GRAPH_HH__

#include <string>

#include "astra-sim/system/Common.hh"
#include "SimCCL/mock/MockNcclChannel.h"

namespace AstraSim {

// Timed flow graph of one collective (collective-graph: 1 in the system config), shared by
// NcclFlowModel and NcclCustom.
//
// A graph covers one group (TP/DP/EP replica) of one collective, identified by a key that every rank
// of the group computes identically (see Sys::next_collective_graph_key). Each rank opens it with
// begin() and the group size, adds the flows it sends or receives, and reports when it finished.
// For every flow six events are recorded:
//   posted    dependencies met, the model decides to send     (model)
//   released  MemBus done, flow handed to the network         (model, front_end_sim_send)
//   started   everything queued before it on its QP is sent,  (network frontend: pushed to an idle
//             the QP starts sending this message               QP, or previous message's send_finish)
//   sent      last packet of the flow left the NIC            (model, PacketSentFinshed)
//   arrived   whole flow received by the destination NIC      (network frontend, recv_finish)
//   acked     ACK of the whole flow back at the sender        (network frontend, message_finish)
// A flow striped over several QPs keeps the first start.
// The graph is written when every rank of the group has finished and, if the frontend reports
// network events (enable_network_events), every released flow has been acked:
//   collective_graph_<key>.txt   one line per flow for reading, grouped by chunk: the six times
//                                and the parents
//   collective_graph_<key>.csv   one row per flow, absolute times (for scripts)
// Graphs still open when the program exits are written with an "_incomplete" suffix.
// All calls are thread safe. An empty key disables every call.
class CollectiveGraph {
public:
  enum class Event { Posted, Released, Started, Sent, Arrived, Acked };

  static void begin(const std::string& key, int expected_ranks);
  static void add_flows(const std::string& key, const MockNccl::FlowModels& flows);
  static void record(const std::string& key, int flow_id, Event event, Tick tick);
  // Called by the model right before handing the flow to the network frontend: started/arrived/acked
  // events for (src, dst, flow_id, tag) are then attributed to this graph.
  static void released(const std::string& key, int src, int dst, int flow_id, int tag, Tick tick);
  // Network frontend hooks (no-ops unless a graph has a released flow in flight).
  static void enable_network_events();
  static void network_event(int src, int dst, int flow_id, int tag, Event event, Tick tick);
  // Returns true if this call wrote the graph.
  static bool rank_finished(const std::string& key, int rank);
};

} // namespace AstraSim
#endif
