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

#ifndef __ENTRY_H__
#define __ENTRY_H__

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/AstraNetworkAPI.hh"
#include "common.h"

#include <ns3/applications-module.h>
#include <ns3/core-module.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/sim-setting.h>

#include <fstream>
#include <iostream>
#include <map>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef NS3_MTP
  #include <ns3/mtp-interface.h>
#endif

using namespace ns3;
using namespace std;
inline std::unordered_map<std::string, ApplicationContainer> appCon;

struct task1 {
  int src;
  int dest;
  int type;
  /** Indicates the number of bytes remaining to be sent or received. */
  uint64_t count;
  void* fun_arg;
  void (*msg_handler)(void* fun_arg);
  double schTime;
};

// MsgEventKey is a key to uniquely identify each MsgEvent.
//  - Pair <Tag, Pair <src_id, dst_id>>
typedef pair<int, pair<int, int>> MsgEventKey;

inline std::map<std::pair<std::pair<int, int>, int>, AstraSim::ncclFlowTag> receiver_pending_queue;

/**
 * The ns3 RdmaClient structure cannot hold the 'tag' information, which is a
 * Astra-sim specific implementation. We use a mapping with the source port
 * number (another unique value) to hold tag information.
 *   - key: Pair <port_id, Pair <src_id, dst_id>>
 *   - value: tag
 * TODO: It seems we *can* obtain the tag through q->GetTag() at qp_finish.
 * Verify & Simplify.
 */
inline std::map<std::pair<uint64_t, std::pair<int, std::pair<int, int>>>, AstraSim::ncclFlowTag> sender_src_port_map;

/**
 * Holds messages where sim_recv has been called, but ns3 has not yet simulated the message arriving.
 *   - key: A MsgEventKey instance
 *   - value: A MsgEvent instance that indicates that Sys layer is waiting for a "receive" event to finish
 */
inline std::map<MsgEventKey, task1> expeRecvHash;

/**
 * Holds messages which ns3 has simulated the arrival, but sim_recv has not yet been called.
 *   - key: A MsgEventKey instance
 *   - value: The number of bytes that ns3 has simulated as completed
 */
inline std::map<MsgEventKey, uint64_t> recvHash;

/**
 * Stores a MsgEvent for sim_send events and its callback handler.
 *   - key: A pair of <MsgEventKey, port_id>.
 *          A single collective phase can be split into multiple sim_send messages, which all have the same MsgEventKey.
 *          TODO: Adding port_id as key is a hacky solution. The real solution would be to split this map, similar to expeRecvHash and recvHash.
 *   - value: A MsgEvent instance that indicates that Sys layer is waiting for a send event to finish
 */
inline std::map<MsgEventKey, task1> sentHash;

/**
 *  Used to count how many bytes were sent/received by this node.
 *    - key: Pair <node_id, send/receive>. Where 'send/receive' indicates if the value is for send or receive
 *    - value: Number of bytes this node has sent or received
 */
inline std::map<std::pair<int, int>, uint64_t> nodeHash;
inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;
inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_message_finish;
inline std::map<std::pair<int, std::pair<int, int>>, uint64_t> received_chunksize;
inline std::map<std::pair<int, std::pair<int, int>>, uint64_t> sent_chunksize;

static std::once_flag sim_finished;
static std::atomic<bool> waiting_sim_finish(false);

inline bool is_sending_finished(int src, int dst, const AstraSim::ncclFlowTag& flowTag) {
  int flow_id = flowTag.current_flow_id;
  if (waiting_to_sent_callback.count(std::make_pair(flow_id, std::make_pair(src, dst)))) {
    if (--waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))] == 0) {
      waiting_to_sent_callback.erase(std::make_pair(flow_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

inline bool is_receive_finished(int src, int dst, const AstraSim::ncclFlowTag& flowTag) {
  int flow_id = flowTag.current_flow_id;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (waiting_to_notify_receiver.count(std::make_pair(flow_id, std::make_pair(src, dst)))) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        " is_receive_finished waiting_to_notify_receiver  flow_id %d src %d dst %d count %d",
        flow_id,
        src,
        dst,
        waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]);
    if (--waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))] == 0) {
      waiting_to_notify_receiver.erase(std::make_pair(flow_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

inline bool is_message_finished(int src, int dst, int flow_id) {
  if (waiting_to_message_finish.count(std::make_pair(flow_id, std::make_pair(src, dst)))) {
    if (--waiting_to_message_finish[std::make_pair(flow_id, std::make_pair(src, dst))] == 0) {
      waiting_to_message_finish.erase(std::make_pair(flow_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

inline std::string get_hash_key(uint32_t src, uint32_t dst, uint32_t pg, uint32_t dport, uint64_t channel_id) {
  return std::to_string(src) + '_' + std::to_string(dst) + '_' + std::to_string(pg) + '_' + std::to_string(dport) +
      '_' + std::to_string(channel_id);
}

inline std::vector<Ptr<RdmaClient>> get_clients(
    uint32_t src,
    uint32_t dst,
    uint32_t pg,
    uint32_t dport,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg,
    int channel_id,
    int sendLat,
    bool nvls_on,
    size_t n_clients) {
  std::vector<Ptr<RdmaClient>> clients;
  const bool reuse = reuse_qps;
  std::string hashKey = get_hash_key(src, dst, pg, dport, channel_id);
#ifdef NS3_MTP
  MtpInterface::CriticalSection cs;
#endif
  if (appCon[hashKey].GetN() == 0) {
    for (int i = 0; i < n_clients; i++) {
      uint32_t port = portNumber[src][dst]++; // get a new port number
      RdmaClientHelper clientHelper(
          pg,
          serverAddress[src],
          serverAddress[dst],
          port,
          dport,
          0, // create a qp w/o message
          has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
          global_t == 1 ? maxRtt : pairRtt[src][dst],
          msg_handler,
          fun_arg,
          src,
          dst,
          !reuse);
      if (nvls_on) {
        clientHelper.SetAttribute("NVLS_enable", UintegerValue(1));
      }
      appCon[hashKey].Add(clientHelper.Install(n.Get(src)));
    }
    appCon[hashKey].Start(Time(sendLat));
  }
  for (int i = 0; i < n_clients; i++) {
    Ptr<RdmaClient> qp = DynamicCast<RdmaClient>(appCon[hashKey].Get(i));
    clients.push_back(qp);
  }
  if (!reuse) {
    appCon[hashKey] = ApplicationContainer();
  }
  return clients;
}

inline void push_msg_to_client(Ptr<RdmaClient> client, uint64_t size, uint64_t flow_id, uint64_t tag) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "push_msg_to_client, %u -> %u, flow_id %u, tag %u, at the tick %u",
      client->m_qp->m_src, client->m_qp->m_dest, flow_id, tag, AstraSim::Sys::boostedTick());
  client->PushMessageToQp(size, flow_id, tag);
}

/**
 * From AstraSim:
 * send_flow commands the ns3 simulator to schedule a RDMA message to be sent
 * between two pair of nodes. send_flow is triggered by sim_send.
 */
void send_flow(
    int src,
    int dst,
    uint64_t maxPacketCount,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg,
    int tag,
    AstraSim::sim_request* request) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();

  bool nvls_on = request->flowTag.nvls_on;
  int flow_id = request->flowTag.current_flow_id;
  int channel_id = request->flowTag.channel_id;
  int pg = 3, dport = 100;
  int send_lat = 6;
  if (const char* send_lat_env = std::getenv("AS_SEND_LAT")) {
    try {
      send_lat = std::stoi(send_lat_env);
    } catch (const std::invalid_argument&) {
      NcclLog->writeLog(NcclLogLevel::ERROR, "send_lat set error");
      exit(-1);
    }
  }
  send_lat *= 1000;

  if (maxPacketCount == 0) {
    maxPacketCount = 1;
  }

  // Create a MsgEvent instance and register callback function.
  task1 t;
  t.src = src;
  t.dest = dst;
  t.count = maxPacketCount;
  t.type = 0;
  t.fun_arg = fun_arg;
  t.msg_handler = msg_handler;
  t.schTime = 0;
  {
    #ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
    #endif
    sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
  }

  // Create (or reuse) one or more QPs for this flow
  uint16_t qps_per_flow = 1;
  if (flow_stripping && src/gpus_per_server != dst/gpus_per_server) {
    const size_t next_hops = nextHop[n.Get(src)][n.Get(dst)].size();
    // use four qps per each next hop to increase the chances of distributing them across all the interfaces
    qps_per_flow = next_hops*4;
  }
  std::vector<Ptr<RdmaClient>> clients =
      get_clients(src, dst, pg, dport, msg_handler, fun_arg, channel_id, send_lat, nvls_on, qps_per_flow);

  // Schedule the flow within the ns3 simulator
  const uint64_t base_size = maxPacketCount / qps_per_flow;
  const uint64_t last_size = base_size + maxPacketCount % qps_per_flow;
  for (int qp_index = 0; qp_index < qps_per_flow; qp_index++) {
    uint64_t size = qp_index == qps_per_flow - 1 ? last_size : base_size;
    uint32_t port = clients[qp_index]->GetSourcePort();

    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "[send_flow] %d -> %d on ch=%d. Port: %u. Flow ID: %d. NVLink: %d. Size: %llu. Tick: %ld",
        src, dst, channel_id, port, flow_id, nvls_on, size, AstraSim::Sys::boostedTick());

    {
#ifdef NS3_MTP
      MtpInterface::CriticalSection cs;
#endif
      // Store the association between this port and the tag
      sender_src_port_map[std::make_pair(flow_id, std::make_pair(port, std::make_pair(src, dst)))] = request->flowTag;
      waiting_to_sent_callback[std::make_pair(flow_id, std::make_pair(src, dst))]++;
      waiting_to_notify_receiver[std::make_pair(flow_id, std::make_pair(src, dst))]++;
      waiting_to_message_finish[std::make_pair(flow_id, std::make_pair(src, dst))]++;
    }

    Simulator::ScheduleWithContext(
        n.Get(src)->GetId(), Time(send_lat + 1), push_msg_to_client, clients[qp_index], size, flow_id, tag);
  }
  flow_input.idx++;
}

/**
 * From AstraSim:
 * notify_receiver_receive_data looks at whether the System layer has issued
 * sim_recv for this message. If the system layer is waiting for this message,
 * call the callback handler for the MsgEvent. If the system layer is not *yet*
 * waiting for this message, register that this message has arrived,
 * so that the system layer can later call the callback handler when sim_recv
 * is called.
 */
void notify_receiver_receive_data(
    int sender_node,
    int receiver_node,
    uint64_t message_size,
    AstraSim::ncclFlowTag flowTag) {
  {
    #ifdef NS3_MTP
    MtpInterface::ExplicitCriticalSection ecs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int tag = flowTag.tag_id;
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        " %d notify receiver %d, tag %u, message size %llu, channel_id %d, flow_id %u",
        sender_node, receiver_node, tag, message_size, flowTag.channel_id, flowTag.current_flow_id);
    if (expeRecvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
      // The Sys object is waiting for packets to arrive.
      task1 t2 = expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          " %d notify receiver %d, tag %u, message size %llu, t2.count %llu, channel_id %d",
          sender_node, receiver_node, tag, message_size, t2.count, flowTag.channel_id);
      const auto ehd = static_cast<AstraSim::RecvPacketEventHadndlerData*>(t2.fun_arg);
      if (message_size == t2.count) {
        // We received exactly the amount of data what Sys object was expecting.
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " message_size = t2.count expeRecvHash.erase %d notify receiver %d, tag %u, message size %llu",
            sender_node, receiver_node, tag, message_size);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        ecs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else if (message_size > t2.count) {
        // We received more packets than the Sys object is expecting.
        // Place task in recvHash and wait for Sys object to issue more sim_recv
        // calls. Call callback handler for the amount Sys object was waiting for.
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = message_size - t2.count;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            "message_size > t2.count expeRecvHash.erase, %d notify receiver %d, tag %u, message size %llu",
            sender_node, receiver_node, tag, message_size);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        ecs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else {
        // There are still packets to arrive.
        // Reduce the number of packets we are waiting for. Do not call callback handler.
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            "message_size < t2.count expeRecvHash.decrease, %d notify receiver %d, tag %u, message size %llu",
            sender_node, receiver_node, tag, message_size);
        t2.count -= message_size;
        expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = t2;
      }
    } else {
      // The Sys object is not yet waiting for packets to arrive.
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          " %d notify receiver %d, tag %u, message size %d, expeRecvHash not found",
          sender_node, receiver_node, tag, message_size);
      receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node), tag)] = flowTag;
      if (recvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) == recvHash.end()) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = message_size;
      } else {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] += message_size;
      }
    }
    #ifdef NS3_MTP
    ecs.ExitSection();
    #endif
  receiver_end_1st_section:
    {
      #ifdef NS3_MTP
      MtpInterface::CriticalSection cs;
      #endif
      // Add to the number of total bytes received.
      if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
        nodeHash[make_pair(receiver_node, 1)] = message_size;
      } else {
        nodeHash[make_pair(receiver_node, 1)] += message_size;
      }
    }
  }
}

void notify_sender_sending_finished(
  int sender_node,
  int receiver_node,
  uint64_t message_size,
  AstraSim::ncclFlowTag flowTag) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  #ifdef NS3_MTP
  MtpInterface::ExplicitCriticalSection ecs;
  #endif
  int tag = flowTag.tag_id;
  // Lookup the send_event registered at send_flow().
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    const auto ehd = static_cast<AstraSim::SendPacketEventHandlerData*>(t2.fun_arg);
    ehd->flowTag = flowTag;
    // Verify that the (ns3 identified) sent message size matches what was expected by the system layer.
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      // Add to the number of total bytes sent.
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      #ifdef NS3_MTP
      ecs.ExitSection();
      #endif
      t2.msg_handler(t2.fun_arg);
      return;
    } else {
      NcclLog->writeLog(NcclLogLevel::ERROR,
          "sentHash: msg size %u != expected bytes %u, %d -> %d, tag %u, flow_id %u",
          message_size, t2.count, sender_node, receiver_node, tag, flowTag.current_flow_id);
      cerr << "The message size does not match what is expected. Something is wrong. "
           << "tag, src_id, dst_id, expected msg_bytes, actual msg_bytes: "
           << tag << ", " << sender_node << ", " << receiver_node << ", " << t2.count << ", " << message_size
           << endl;
      exit(1);
    }
  } else {
    NcclLog->writeLog(NcclLogLevel::ERROR,
        "sentHash: cannot find sender_node %d receiver_node %d message_size %lu",
        sender_node, receiver_node, message_size);
    cerr << "Cannot find send_event in sentHash. Something is wrong."
         << "tag, src_id, dst_id: " << tag << ", " << sender_node << ", " << receiver_node
         << endl;
    exit(1);
  }
  #ifdef NS3_MTP
  ecs.ExitSection();
  #endif
}

[[maybe_unused]]
void notify_sender_packet_arrived_receiver(
    int sender_node,
    int receiver_node,
    uint64_t message_size,
    AstraSim::ncclFlowTag flowTag) {
#ifdef NS3_MTP
  MtpInterface::ExplicitCriticalSection ecs;
#endif
  int tag = flowTag.channel_id;
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) != sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    const auto ehd = static_cast<AstraSim::SendPacketEventHandlerData*>(t2.fun_arg);
    ehd->flowTag = flowTag;
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
#ifdef NS3_MTP
      ecs.ExitSection();
#endif
      t2.msg_handler(t2.fun_arg);
      return;
    }
  }
#ifdef NS3_MTP
  ecs.ExitSection();
#endif
}

void finish() {
  for (auto it : appCon) {
    for (int i = 0; i < it.second.GetN(); i++) {
      Ptr<RdmaClient> app = DynamicCast<RdmaClient>(it.second.Get(i));
      app->FinishQp();
    }
  }
}

void check_sim_finish() {
  if (waiting_sim_finish && waiting_to_notify_receiver.empty() && waiting_to_sent_callback.empty() &&
      waiting_to_message_finish.empty() && sentHash.empty() && expeRecvHash.empty()) {
    std::call_once(sim_finished, [] {
      std::cout << "All messages finished. Stopping simulation at time "
                << AstraSim::Sys::boostedTick() << "."
                << std::endl;
      finish();
    });
    Simulator::Stop();
  }
}

inline void print_fct_entry(FILE *fout, Ptr<RdmaQueuePair> q, const uint64_t msg_size) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  uint64_t size = msg_size;
  uint32_t total_bytes = size +
      ((size - 1) / packet_payload_size + 1) *
      (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes
                                                                               // required (with header but no INT)
  uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
  // sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
  fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
          q->sport, q->dport, size, q->startTime.GetTimeStep(),
          (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
  fflush(fout);
}

/**
 * Invoked when receiving the ack of the last packet for the current flow.
 */
void message_finish(FILE* fout, Ptr<RdmaQueuePair> q, const RdmaQueuePair::RdmaMessage& msg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "message_finish, %u -> %u, flow_id %u, tag %u, %u flows left in qp, at the tick %u",
      q->m_src, q->m_dest, msg.m_flow_id, msg.m_tag, q->m_messages.size(), AstraSim::Sys::boostedTick());
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  print_fct_entry(fout, q, msg.m_size);

  {
    #ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
    #endif
    sender_src_port_map.erase(make_pair(msg.m_flow_id, make_pair(q->sport, make_pair(sid, did))));
    if (!is_message_finished(sid, did, msg.m_flow_id)) {
      return;
    }
  }
  Simulator::Schedule(Time(0), &check_sim_finish);
}

/**
 * Invoked when a QP completes.
 * Since QPs are reused for multiple flows, this is only invoked at the end of the simulation.
 */
void qp_finish(FILE* fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  AstraSim::ncclFlowTag flowTag;
  {
#ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
#endif
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
  }
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "qp_finish, %d -> %d, port %d, at the tick %d",
      sid, did, q->sport, AstraSim::Sys::boostedTick());
}

/**
 * Invoked when the last packet of a flow has been sent.
 */
void send_finish(FILE* fout, Ptr<RdmaQueuePair> q, const RdmaQueuePair::RdmaMessage& msg) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  AstraSim::ncclFlowTag flowTag;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "send_finish, %d -> %d, flow_id %d, tag %u, port %d, total bytes %llu, at the tick %d",
      sid,  did, msg.m_flow_id, msg.m_tag, q->sport, msg.m_size, AstraSim::Sys::boostedTick());
  uint64_t all_sent_chunksize;
  {
#ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
#endif
    flowTag = sender_src_port_map[std::make_pair(
        msg.m_flow_id,
        std::make_pair(q->sport, std::make_pair(sid, did)))];
    if (flowTag.current_flow_id != msg.m_flow_id) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"Exit with unequal flow_id %d", flowTag.current_flow_id);
      std::cout << "[send_finish] Exit with unequal flow_id " << flowTag.current_flow_id << " in sender_src_port_map for flow " << msg.m_flow_id << "," << q->sport << "," << sid << "," << did << std::endl;
      exit(-1);
    }
    sent_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))] += msg.m_size;
    if (!is_sending_finished(sid, did, flowTag)) {
      return;
    }
    all_sent_chunksize = sent_chunksize[std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did))];
    sent_chunksize.erase(std::make_pair(flowTag.current_flow_id, std::make_pair(sid, did)));
  }
  notify_sender_sending_finished(sid, did, all_sent_chunksize, flowTag);
}


/**
 * Invoked when sending the last ack of the current flow.
 */
void recv_finish(FILE* fout, Ptr<RdmaRxQueuePair> rx_q, const RdmaRxQueuePair::RdmaMessage& msg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  const uint32_t sip = rx_q->dip, dip = rx_q->sip;
  uint16_t sport = rx_q->dport;
  uint32_t sid = ip_to_node_id(Ipv4Address(sip)), did = ip_to_node_id(Ipv4Address(dip));
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "recv_finish, %u -> %u, flow_id %u, tag %u, %u flows left in qp, at the tick %u",
      sid, did, msg.m_flow_id, msg.m_tag, rx_q->m_messages.size(), AstraSim::Sys::boostedTick());

  AstraSim::ncclFlowTag flowTag;
  uint64_t notify_size;
  {
    #ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
    #endif
    if (sender_src_port_map.find(make_pair(msg.m_flow_id, make_pair(sport, make_pair(sid, did)))) ==
        sender_src_port_map.end()) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"could not find the tag, there must be something wrong");
      exit(-1);
    }
    flowTag = sender_src_port_map[std::make_pair(
        msg.m_flow_id,
        std::make_pair(sport, std::make_pair(sid, did)))];
    if (flowTag.current_flow_id != msg.m_flow_id) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"Exit with unequal flow_id");
      std::cout << "[recv_finish] Exit with unequal flow_id " << flowTag.current_flow_id << " in sender_src_port_map (expected " << msg.m_flow_id << ")" << std::endl;
      exit(-1);
    }
    received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))] += msg.m_size;
    if (!is_receive_finished(sid,did,flowTag)) {
      return;
    }
    notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    received_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));
  }
  notify_receiver_receive_data(sid, did, notify_size, flowTag);
}

int setup_ns3_simulation(const string& network_topo, const string& network_conf, const string& run_name) {
  clock_t begint, endt;
  begint = clock();

  if (!ReadConf(network_topo, network_conf, run_name)) {
    std::cerr << "Unable to open configuration file: " << network_conf << std::endl;
    std::cerr << "This error is fatal." << std::endl;
    exit(1);
  }
  SetConfig();
  SetupNetwork(qp_finish, message_finish, send_finish, recv_finish);

  std::cout << "Running Simulation.\n";
  fflush(stdout);
  NS_LOG_INFO("Run Simulation.");

  endt = clock();
  return 0;
}
#endif
