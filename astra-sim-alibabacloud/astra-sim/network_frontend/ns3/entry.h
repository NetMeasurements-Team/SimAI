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

#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ns3/applications-module.h>
#include <ns3/core-module.h>
#include <ns3/rdma-client-helper.h>
#ifdef NS3_MTP
  #include <ns3/mtp-interface.h>
#endif

#include "astra-sim/system/MockNcclLog.h"
#include "common.h"

using namespace ns3;
using namespace std;

/**
 * @brief Global lookup table that associates application identifiers with their
 *        corresponding `ApplicationContainer` instances.
 *
 * This inline `std::unordered_map` enables fast retrieval of an
 * `ApplicationContainer` by its string key, allowing different translation units
 * to share a common registry of applications. The map is defined as `inline` to
 * avoid multiple-definition errors when included in multiple source files.
 */
inline std::unordered_map<std::string, ApplicationContainer> appCon;

class MsgEvent {
public:
  int src;
  int dst;
  int type;
  /**
   * Number of bytes remaining to be sent or received.
   * Initialized with the original size of the message and
   * incremented/decremented based on sent/received bytes.
   * Eventually, this value will reach zero when the event completes.
   */
  uint64_t remaining_bytes;
  void (*msg_handler)(void *fun_arg);
  void *fun_arg;

  MsgEvent(
      const int _src,
      const int _dst,
      const int _type,
      const int _remaining_msg_bytes,
      void (*_msg_handler)(void* fun_arg),
      void* _fun_arg) :
      src(_src), dst(_dst), type(_type), remaining_bytes(_remaining_msg_bytes),
      msg_handler(_msg_handler), fun_arg(_fun_arg) {}

  /**
   * Default constructor only declared to prevent compile errors.
   * When looking up MsgEvents from maps such as sentHash, we should always check that a MsgEvent exists
   * for the given key (i.e., this default constructor should not be called in runtime).
   */
  MsgEvent() : src(0), dst(0), type(0), remaining_bytes(0), msg_handler(nullptr), fun_arg(nullptr) {}

  /**
   * Invoke the callback handler associated with this MsgEvent.
   */
  void callHandler() const {
    msg_handler(fun_arg);
  }
};

/**
 * MsgEventKey is a key to uniquely identify each MsgEvent.
 *  - Pair <Tag, Pair <src_id, dst_id>>
 */
typedef pair<int, pair<int, int>> MsgEventKey;

/**
 * FlowIdKey is a key to uniquely identify a flow
 * (might refer to multiple messages in case of flow stripping).
 *  - Pair <flow_id, Pair <src_id, dst_id>>
 * TODO: Isn't flow_id enough?
 */
typedef pair<int, pair<int, int>> FlowIdKey;

/**
 * Stores a MsgEvent for sim_recv events and its callback handler.
 * Holds messages where sim_recv has been called, but ns3 has not yet simulated the message arriving.
 *   - key: A MsgEventKey instance
 *   - value: A MsgEvent instance that indicates that Sys layer is waiting for a "receive" event to finish
 */
inline map<MsgEventKey, MsgEvent> expeRecvHash;

/**
 * This is a helper structure that holds messages which ns3 has simulated the arrival,
 * but for which sim_recv has not yet been called. When the corresponding sim_recv is called,
 * the size stored in recvHash is removed from the MsgEvent before storing it in expeRecvHash.
 *   - key: A MsgEventKey instance
 *   - value: The number of bytes that ns3 has simulated as completed
 */
inline std::map<MsgEventKey, uint64_t> recvHash;

/**
 * Stores a MsgEvent for sim_send events and its callback handler.
 *   - key: A MsgEventKey instance
 *   - value: A MsgEvent instance that indicates that Sys layer is waiting for a "send" event to finish
 */
inline map<MsgEventKey, MsgEvent> sentHash;

/**
 *  Used to count how many bytes were sent/received by this node.
 *    - key: Pair <node_id, send/receive>. Where 'send/receive' indicates if the value is for send or receive
 *    - value: Number of bytes this node has sent or received
 */
inline std::map<std::pair<int, int>, uint64_t> nodeHash;

inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;
inline std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_message_finish;

/**
 * Cumulates the size of received messages before triggering notify_receiver_receive_data.
 * This is needed in the case the same flow_id was stripped into multiple flows.
 */
inline std::map<std::pair<int, std::pair<int, int>>, uint64_t> received_chunksize;

/**
 * Similar to received_chunksize, but sender-size (for triggering notify_sender_sending_finished).
 */
inline std::map<std::pair<int, std::pair<int, int>>, uint64_t> sent_chunksize;

static std::once_flag sim_finished;
static std::atomic<bool> waiting_sim_finish(false);

inline bool is_sending_finished(int src, int dst, int flow_id) {
  if (waiting_to_sent_callback.count(FlowIdKey{flow_id, {src, dst}})) {
    if (--waiting_to_sent_callback[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_sent_callback.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}

inline bool is_receive_finished(int src, int dst, int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (waiting_to_notify_receiver.count(FlowIdKey{flow_id, {src, dst}})) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        " is_receive_finished waiting_to_notify_receiver  flow_id %d src %d dst %d count %d",
        flow_id, src, dst, waiting_to_notify_receiver[FlowIdKey{flow_id, {src, dst}}]);
    if (--waiting_to_notify_receiver[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_notify_receiver.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}

inline bool is_message_finished(int src, int dst, int flow_id) {
  if (waiting_to_message_finish.count(FlowIdKey{flow_id, {src, dst}})) {
    if (--waiting_to_message_finish[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_message_finish.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}

inline std::string get_hash_key(
    const uint32_t src,
    const uint32_t dst,
    const uint32_t pg,
    const uint32_t dport,
    const uint64_t channel_id) {
  return std::to_string(src) + '_' + std::to_string(dst) + '_' + std::to_string(pg) + '_' + std::to_string(dport) +
      '_' + std::to_string(channel_id);
}

inline std::vector<Ptr<RdmaClient>> get_clients(
    const uint32_t src,
    const uint32_t dst,
    const uint32_t pg,
    const uint32_t dport,
    const int channel_id,
    const int send_lat,
    const bool nvls_on,
    const size_t n_clients) {
  std::vector<Ptr<RdmaClient>> clients;
  const bool reuse = reuse_qps;
  // Each channel gets a different QP
  const std::string hashKey = get_hash_key(src, dst, pg, dport, channel_id);
  #ifdef NS3_MTP
  MtpInterface::CriticalSection cs;
  #endif
  if (appCon[hashKey].GetN() == 0) {
    for (int i = 0; i < n_clients; i++) {
      const uint32_t port = portNumber[src][dst]++; // get a new port number
      RdmaClientHelper clientHelper(
          pg,
          serverAddress[src],
          serverAddress[dst],
          port,
          dport,
          0, // create a qp w/o message
          has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
          global_t == 1 ? maxRtt : pairRtt[src][dst],
          packet_spraying && src/gpus_per_server != dst/gpus_per_server,
          src,
          dst,
          !reuse);
      if (nvls_on) {
        clientHelper.SetAttribute("NVLS_enable", UintegerValue(1));
      }
      appCon[hashKey].Add(clientHelper.Install(n.Get(src)));
    }
    appCon[hashKey].Start(Time(send_lat));
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

inline void push_msg_to_client(
    Ptr<RdmaClient> client,
    const uint64_t size,
    const uint64_t flow_id,
    const uint64_t tag) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "push_msg_to_client, %u -> %u, flow_id %u, tag %u, port %u, at the tick %u",
      client->m_qp->m_src, client->m_qp->m_dest, flow_id, tag, client->m_qp->sport, AstraSim::Sys::boostedTick());
  client->PushMessageToQp(size, flow_id, tag);
}

/**
 * From AstraSim:
 * send_flow commands the ns3 simulator to schedule a RDMA message to be sent
 * between two pair of nodes. send_flow is triggered by sim_send.
 */
inline void send_flow(
    int src,
    int dst,
    uint64_t maxPacketCount,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg,
    int tag,
    int flow_id,
    bool nvls_on) {
  const auto ehd = static_cast<AstraSim::SendPacketEventHandlerData*>(fun_arg);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();

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
inline void notify_receiver_receive_data(
    int sender_node,
    int receiver_node,
    const uint64_t message_size,
    int tag) {
  #ifdef NS3_MTP
  MtpInterface::ExplicitCriticalSection ecs;
  #endif
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::DEBUG,
      " %d notify receiver %d, tag %u, message size %llu",
      sender_node, receiver_node, tag, message_size);
  if (expeRecvHash.find(MsgEventKey{tag, {sender_node, receiver_node}}) != expeRecvHash.end()) {
    // The Sys object is waiting for packets to arrive.
    MsgEvent recv_event = expeRecvHash[MsgEventKey{tag, {sender_node, receiver_node}}];
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        " %d notify receiver %d, tag %u, message size %llu, t2.count %llu",
        sender_node, receiver_node, tag, message_size, recv_event.remaining_bytes);
    if (message_size == recv_event.remaining_bytes) {
      // We received exactly the amount of data what Sys object was expecting.
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          " message_size = t2.count expeRecvHash.erase %d notify receiver %d, tag %u, message size %llu",
          sender_node, receiver_node, tag, message_size);
      expeRecvHash.erase(MsgEventKey{tag, {sender_node, receiver_node}});
      #ifdef NS3_MTP
      ecs.ExitSection();
      #endif
      recv_event.callHandler();
      goto receiver_end_1st_section;
    } else if (message_size > recv_event.remaining_bytes) {
      // We received more packets than the Sys object is expecting.
      // Place the task in recvHash and wait for Sys object to issue more sim_recv calls.
      // Call the callback handler for the amount Sys object was waiting for.
      recvHash[MsgEventKey{tag, {sender_node, receiver_node}}] = message_size - recv_event.remaining_bytes;
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "message_size > t2.count expeRecvHash.erase, %d notify receiver %d, tag %u, message size %llu",
          sender_node, receiver_node, tag, message_size);
      expeRecvHash.erase(MsgEventKey{tag, {sender_node, receiver_node}});
      #ifdef NS3_MTP
      ecs.ExitSection();
      #endif
      recv_event.callHandler();
      goto receiver_end_1st_section;
    } else {
      // There are still packets to arrive.
      // Reduce the number of packets we are waiting for. Do not call callback handler.
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "message_size < t2.count expeRecvHash.decrease, %d notify receiver %d, tag %u, message size %llu",
          sender_node, receiver_node, tag, message_size);
      recv_event.remaining_bytes -= message_size;
      expeRecvHash[MsgEventKey{tag, {sender_node, receiver_node}}] = recv_event;
    }
  } else {
    // The Sys object is not yet waiting for packets to arrive.
    // Wait for Sys object to issue more sim_recv calls.
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        " %d notify receiver %d, tag %u, message size %d, expeRecvHash not found",
        sender_node, receiver_node, tag, message_size);
    if (recvHash.find(MsgEventKey{tag, {sender_node, receiver_node}}) == recvHash.end()) {
      recvHash[MsgEventKey{tag, {sender_node, receiver_node}}] = message_size;
    } else {
      recvHash[MsgEventKey{tag, {sender_node, receiver_node}}] += message_size;
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

inline void notify_sender_sending_finished(
    int sender_node,
    int receiver_node,
    const uint64_t message_size,
    int tag) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  #ifdef NS3_MTP
  MtpInterface::ExplicitCriticalSection ecs;
  #endif
  // Find the send_event registered at sim_send.
  if (sentHash.find(MsgEventKey{tag, {sender_node, receiver_node}}) != sentHash.end()) {
    MsgEvent send_event  = sentHash[MsgEventKey{tag, {sender_node, receiver_node}}];
    // Verify that the (ns3 identified) sent message size matches what was expected by the system layer.
    if (send_event.remaining_bytes == message_size) {
      sentHash.erase(MsgEventKey{tag, {sender_node, receiver_node}});
      // Add to the number of total bytes sent.
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      #ifdef NS3_MTP
      ecs.ExitSection();
      #endif
      send_event.callHandler();
    } else {
      NcclLog->writeLog(NcclLogLevel::ERROR,
          "sentHash: msg size %u != expected bytes %u, %d -> %d, tag %u",
          message_size, send_event.remaining_bytes, sender_node, receiver_node, tag);
      cerr << "The message size does not match what is expected. Something is wrong. "
           << "tag, src_id, dst_id, expected msg_bytes, actual msg_bytes: "
           << tag << ", " << sender_node << ", " << receiver_node << ", " << send_event.remaining_bytes << ", "
           << message_size
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
}

inline void finish() {
  for (const auto& [_, client_helper] : appCon) {
    for (int i = 0; i < client_helper.GetN(); i++) {
      Ptr<RdmaClient> app = DynamicCast<RdmaClient>(client_helper.Get(i));
      app->FinishQp();
    }
  }
}

inline void check_sim_finish() {
  if (waiting_sim_finish && waiting_to_notify_receiver.empty() && waiting_to_sent_callback.empty() &&
      waiting_to_message_finish.empty() && sentHash.empty() && expeRecvHash.empty() && recvHash.empty()) {
    std::call_once(sim_finished, [] {
      {
        #ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
        #endif
        std::cout << "All messages finished. Stopping simulation at time "
                  << AstraSim::Sys::boostedTick() << "."
                  << std::endl;
      }
      finish();
    });
    Simulator::Stop();
  }
}

inline void print_fct_entry(FILE *fout, Ptr<RdmaQueuePair> q, const uint64_t msg_size) {
  const uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  const uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  const uint64_t size = msg_size;
  const uint32_t total_bytes = size +
      ((size - 1) / packet_payload_size + 1) *
      (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes
                                                                               // required (with header but no INT)
  const uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
  // sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
  fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
          q->sport, q->dport, size, q->startTime.GetTimeStep(),
          (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
  fflush(fout);
}

/**
 * Invoked when receiving the ack of the last packet for the current flow.
 */
inline void message_finish(FILE* fout, Ptr<RdmaQueuePair> q, const RdmaQueuePair::RdmaMessage& msg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "message_finish, %u -> %u, flow_id %u, tag %u, %u flows left in qp, at the tick %u",
      q->m_src, q->m_dest, msg.m_flow_id, msg.m_tag, q->m_messages.size(), AstraSim::Sys::boostedTick());
  const uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  print_fct_entry(fout, q, msg.m_size);

  {
    #ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
    #endif
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
inline void qp_finish(FILE* fout, Ptr<RdmaQueuePair> q) {
  const uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
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
inline void send_finish(FILE* fout, Ptr<RdmaQueuePair> q, const RdmaQueuePair::RdmaMessage& msg) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "send_finish, %d -> %d, flow_id %d, tag %u, port %d, total bytes %llu, at the tick %d",
      sid,  did, msg.m_flow_id, msg.m_tag, q->sport, msg.m_size, AstraSim::Sys::boostedTick());
  uint64_t notify_size;
  {
#ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
#endif
    sent_chunksize[FlowIdKey{msg.m_flow_id, {sid, did}}] += msg.m_size;
    if (!is_sending_finished(sid, did, msg.m_flow_id)) {
      return;
    }
    notify_size = sent_chunksize[FlowIdKey{msg.m_flow_id, {sid, did}}];
    sent_chunksize.erase(FlowIdKey{msg.m_flow_id, {sid, did}});
  }
  notify_sender_sending_finished(sid, did, notify_size, msg.m_tag);
}

/**
 * Invoked when sending the last ack of the current flow.
 */
inline void recv_finish(FILE* fout, Ptr<RdmaRxQueuePair> rx_q, const RdmaRxQueuePair::RdmaMessage& msg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  const uint32_t sip = rx_q->dip, dip = rx_q->sip;
  uint32_t sid = ip_to_node_id(Ipv4Address(sip)), did = ip_to_node_id(Ipv4Address(dip));
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "recv_finish, %u -> %u, flow_id %u, tag %u, %u flows left in qp, at the tick %u",
      sid, did, msg.m_flow_id, msg.m_tag, rx_q->m_messages.size(), AstraSim::Sys::boostedTick());

  uint64_t notify_size;
  {
    #ifdef NS3_MTP
    MtpInterface::CriticalSection cs;
    #endif
    received_chunksize[FlowIdKey{msg.m_flow_id, {sid, did}}] += msg.m_size;
    if (!is_receive_finished(sid,did,msg.m_flow_id)) {
      return;
    }
    notify_size = received_chunksize[FlowIdKey{msg.m_flow_id, {sid, did}}];
    received_chunksize.erase(FlowIdKey{msg.m_flow_id, {sid, did}});
  }
  notify_receiver_receive_data(sid, did, notify_size, msg.m_tag);
}

inline int setup_ns3_simulation(const string& network_topo, const string& network_conf, const string& run_name) {

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

  return 0;
}
#endif
