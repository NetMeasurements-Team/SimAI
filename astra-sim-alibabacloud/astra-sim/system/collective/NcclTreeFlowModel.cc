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

#ifdef PHY_MTP
  #include <mpi.h>
  #include "astra-sim/system/PhyMultiThread.hh"
#endif
#include <chrono>

#include "NcclTreeFlowModel.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#ifdef PHY_RDMA
  #include "astra-sim/system/SimAiFlowModelRdma.hh"
  extern FlowPhyRdma flow_rdma;
#endif


namespace AstraSim {
std::atomic<bool> NcclTreeFlowModel::g_flow_inCriticalSection(false);
NcclTreeFlowModel::NcclTreeFlowModel(
    ComType type,
    int id,
    int layer_num,
    RingTopology* ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode,
    std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
    int treechannels) : Algorithm(layer_num) {
  this->start_time = std::chrono::high_resolution_clock::now();
  this->end_time = std::chrono::high_resolution_clock::now();
  this->comType = type;
  this->id = id;
  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();
  this->parallel_reduce = 1;
  this->toggle = false;
  this->name = Name::Ring;
  this->enabled = true;
  this->m_channels = treechannels;
#if PHY_RDMA
  this->judge_exit_flag.store(false);
#endif
  // this->judge_exit_mutex.unlock();
  // this->judge_mutex.unlock();
  this->send_packets = 0;
  this->recv_packets = 0;
  zero_latency_packets = new std::map<int, int>();
  non_zero_latency_packets = new std::map<int, int>();
  if (boost_mode) {
    this->enabled = ring_topology->is_enabled();
  }
  if (ring_topology->dimension == RingTopology::Dimension::Local) {
    transmition = MemBus::Transmition::Fast;
  } else {
    transmition = MemBus::Transmition::Usual;
  }
  if (ptr_flow_models) {
    for (auto f : *ptr_flow_models) {
      if (f.second.dest == id) {
        this->free_packets[std::make_pair(f.second.channel_id, f.second.src)]++;
        this->_flow_models[f.first] = f.second;
        ++recv_packets;
      }
      if (f.second.src == id) {
        {
          FlowCriticalSection cs;
          this->_stream_count[f.second.channel_id] += 1;
        }
        assert(this->_flow_models.count(f.first) == 0);
        this->_flow_models[f.first] = f.second;
        ++send_packets;
      }
    }
  }
  for (uint32_t channel_id = 0; channel_id < m_channels; channel_id++) {
    assert(zero_latency_packets->find(channel_id) == zero_latency_packets->end());
    (*zero_latency_packets)[channel_id] = 0;
    assert(non_zero_latency_packets->find(channel_id) == non_zero_latency_packets->end());
    (*non_zero_latency_packets)[channel_id] = 0;
  }
  init_indegree_mapping();
  switch (type) {
  case ComType::All_Reduce:
    this->final_data_size = data_size;
    break;
  case ComType::All_Gather:
    this->final_data_size = data_size * nodes_in_ring;
    break;
  case ComType::Reduce_Scatter:
    this->final_data_size = data_size / nodes_in_ring;
    break;
  case ComType::All_to_All:
    this->final_data_size = data_size;
    break;
  default:;
  }
}

void NcclTreeFlowModel::init_indegree_mapping() {
  for (auto tree_it = _flow_models.begin(); tree_it != _flow_models.end(); ++tree_it) {
    if (tree_it->second.src != id) {
      continue;
    }
    indegree_mapping[tree_it->first.second] = tree_it->second.parent_flow_id.size();
  }
}

int NcclTreeFlowModel::get_non_zero_latency_packets() {
  return (nodes_in_ring - 1) * parallel_reduce * 1;
}

void NcclTreeFlowModel::run(EventType event, CallData* data) {
  auto* ehd = static_cast<BasicEventHandlerData*>(data);
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (event == EventType::General) {
    int channel_id = ehd->channel_id;
    int flow_id = ehd->flow_id;
    ready(channel_id, flow_id);
  } else if (event == EventType::PacketReceived) {
    auto rcv_ehd = static_cast<RecvPacketEventHadndlerData*>(ehd);
    auto& received_flow = _flow_models[std::make_pair(rcv_ehd->channel_id, rcv_ehd->flow_id)];
    std::vector<int> next_flow_list =  received_flow.child_flow_id;

#ifdef PHY_MTP
    recv_packets--;
    if (!phy_iteratable(channel_id)) {
      return;
    }
#else
    bool flow_exist = next_flow_list.empty();
    for (size_t i = 0; i < next_flow_list.size(); ++i) {
      int next_flow_id = next_flow_list[i];
      if (next_flow_id == -1 || _flow_models.count(std::make_pair(rcv_ehd->channel_id, next_flow_id)) != 0) {
        flow_exist = true;
      } else {
        flow_exist = false;
        break;
      }
    }
    assert(flow_exist == true);

    bool all_channel_finished = true;
    bool all_packets_freed = true;
    {
      FlowCriticalSection cs;
      free_packets[std::make_pair(received_flow.channel_id, received_flow.src)]--;
      for (uint32_t i = 0; i < m_channels; ++i) {
        if (_stream_count.count(i) != 0 && _stream_count[i] != 0) {
          all_channel_finished = false;
          break;
        }
      }
      if (all_channel_finished) {
        for (auto it = free_packets.begin(); it != free_packets.end(); ++it) {
          if (it->second != 0) {
            all_packets_freed = false;
            break;
          }
        }
      }
    }

    if (all_channel_finished) {
      ready(received_flow.channel_id, -1);
      if (all_packets_freed) {
        exit();
      }
      return;
    }
#endif
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "PacketReceived sender: %d, receiver: %d, current_flow_id: %d, channel_id: %d, tag_id: %d,"
        " free_packets: %d, next_flow_list.size: %d",
        received_flow.src,
        received_flow.dest,
        received_flow.flow_id,
        received_flow.channel_id,
        rcv_ehd->tag,
        free_packets.count(std::make_pair(received_flow.channel_id, received_flow.src))
            ? free_packets[std::make_pair(received_flow.channel_id, received_flow.src)]
            : -1,
        next_flow_list.size());

#ifdef PHY_MTP
    for (int next_flow_id : next_flow_list) {
      if (--indegree_mapping[next_flow_id] == 0) {
       ready(channel_id, next_flow_id);
      }
    }
#else
    flow_exist = true;
    bool recv_finished_tag = true;
    for (auto it = free_packets.begin(); it != free_packets.end(); ++it) {
      if (it->second != 0) {
        recv_finished_tag = false;
        break;
      }
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "recv_finished_tag: %s", recv_finished_tag ? "true" : "false");
    NcclLog->writeLog(NcclLogLevel::DEBUG, "next_flow_list.size %d", next_flow_list.size());

    for (int next_flow_id : next_flow_list) {
      FlowExplicitCriticalSection ecs;
      if (indegree_mapping.count(next_flow_id) == 0) {
        flow_exist = false;
        ecs.ExitSection();
        break;
      }
      if (--indegree_mapping[next_flow_id] == 0) {
        MockNccl::SingleFlow cur_flow = _flow_models[std::make_pair(received_flow.channel_id, next_flow_id)];
        ecs.ExitSection();
        insert_packets(received_flow.channel_id, next_flow_id);
      } else {
        ecs.ExitSection();
      }
    }
    assert(flow_exist == true);
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "NcclTreeFlowModel::run PacketReceived END. Channel ID: %d, Flow ID: %d",
        received_flow.channel_id,
        received_flow.flow_id);
#endif
  } else if (event == EventType::StreamInit) {
    NcclLog->writeLog(NcclLogLevel::INFO, "NcclTreeFlowModel::run StreamInit ID: %d", id);
#ifdef PHY_MTP
    MPI_Barrier(MPI_COMM_WORLD);
    for (auto single_flow : _flow_models) {
      if ((single_flow.second.src == id || single_flow.second.dest == id)) {
  #ifdef PHY_RDMA
        flow_rdma.ibv_create_peer_qp(
            id,
            single_flow.second.channel_id,
            single_flow.second.src,
            single_flow.second.dest,
            single_flow.second.chunk_count + 1,
            single_flow.second.chunk_id,
            single_flow.second.flow_size);
  #endif
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "streamInit time %lld", now_us);
    start_time = std::chrono::high_resolution_clock::now();
#endif

    for (int i = 0; i < parallel_reduce; i++) {
#ifndef PHY_MTP
      init_recv_ready();
#endif
      for (int j = 0; j < m_channels; j++) {
        for (const auto& flow_model : _flow_models) {
          if (flow_model.second.src != id
              && (comType != ComType::All_to_All || flow_model.second.dest != id)) {
            continue;
          }
          std::vector<int> parent_list = flow_model.second.parent_flow_id;
          if (parent_list.empty() && flow_model.second.channel_id == j) {
#ifdef PHY_MTP
            if (flow_model.second.chunk_id == 0) {
              ready(j, flow_model.second.flow_id);
            }
#else
            insert_packets(j, flow_model.second.flow_id);
#endif
          }
        }
      }
#ifdef PHY_MTP
      waiting_to_exit();
      NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel::waiting_to_exit end ");
#endif
    }
  } else if (event == EventType::PacketSentFinshed) {
    const auto snd_ehd = static_cast<SendPacketEventHandlerData*>(ehd);
    const auto& sent_flow = _flow_models[std::make_pair(snd_ehd->channel_id, snd_ehd->flow_id)];
    std::vector<int> next_flow_list = sent_flow.child_flow_id;
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "PacketSentFinshed src %d dst %d channel_id %d flow_id %d",
        sent_flow.src,
        sent_flow.dest,
        sent_flow.channel_id,
        sent_flow.flow_id);
    reduce(sent_flow.channel_id, sent_flow.flow_id);

    #ifndef PHY_MTP
    bool all_channel_finished = true;
    bool all_packets_freed = true;
    {
      FlowCriticalSection cs;
      for (uint32_t i = 0; i < m_channels; ++i) {
        if (_stream_count.count(i) != 0 && _stream_count[i] != 0) {
          all_channel_finished = false;
          break;
        }
      }
      if (all_channel_finished) {
        for (auto it = free_packets.begin(); it != free_packets.end(); ++it) {
          if (it->second != 0) {
            all_packets_freed = false;
            break;
          }
        }
      }
    }
    if (all_channel_finished && all_packets_freed) {
      exit();
    }
    #else
    phy_iteratable(channel_id);
    #endif
  }
}

bool NcclTreeFlowModel::init_recv_ready() {
  std::map<std::pair<int, std::vector<int>>, std::vector<int>> recv_ready_flows;
  for (auto flow : _flow_models) {
    if (flow.second.src != id && (flow.second.conn_type != "PTP_PXN_END" || flow.second.dest != id)) {
      continue;
    }
    if (flow.second.chunk_id != 0) {
      continue;
    }
    if (flow.second.parent_flow_id.empty()) {
      continue;
    }
    std::pair<int, std::vector<int>> cur = std::make_pair(flow.second.channel_id, flow.second.prev);
    if (recv_ready_flows.count(cur) == 0) {
      recv_ready_flows[cur] = {flow.second.flow_id};
    } else {
      std::vector<int> flow_ids = recv_ready_flows[cur];
      bool flag = true;
      for (int flow_id : flow_ids) {
        MockNccl::SingleFlow old_flow = _flow_models[std::make_pair(flow.second.channel_id, flow_id)];
        if (old_flow.parent_flow_id == flow.second.parent_flow_id) {
          flag = false;
          break;
        }
      }
      if (flag) {
        recv_ready_flows[cur].push_back(flow.second.flow_id);
      }
    }
  }
  for (auto recv_ready_flow_it = recv_ready_flows.begin();
       recv_ready_flow_it != recv_ready_flows.end();
       ++recv_ready_flow_it) {
    for (int flow_id : recv_ready_flow_it->second) {
      recv_ready(recv_ready_flow_it->first.first, flow_id);
    }
  }
  return true;
}

bool NcclTreeFlowModel::recv_ready(int channel_id, int flow_id) {
  const auto flow_model = _flow_models[std::make_pair(channel_id, flow_id)];
  std::vector<int> data_sources = flow_model.prev;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::INFO,
      "NcclTreeFlowModel::recv_ready called for channel_id: %d, flow_id: %d",
      channel_id,
      flow_id);

  if (flow_model.conn_type == "PTP_PXN_END" && flow_model.dest == id) {
    data_sources = { flow_model.src };
  }
  for (const int data_source : data_sources) {
    // find the source flow
    MockNccl::SingleFlow source_flow;
    if (data_source != id) {
      auto it = std::find_if(_flow_models.begin(), _flow_models.end(),
      [&](const std::pair<std::pair<int, int>, MockNccl::SingleFlow>& entry) {
         const auto& flow = entry.second;
         return flow.src == data_source &&
                flow.dest == id &&
                flow.channel_id == channel_id &&
                flow.chunk_id == flow_model.chunk_id;
         });
      if (it == _flow_models.end()) {
        std::cerr << "No flow to receive from when initializing a source for flow_id " << flow_id << std::endl;
        std::exit(-1);
      }
      source_flow = it->second;
    } else {
      source_flow = flow_model;
    }

    // FIXME this is never used
    sim_request rcv_req;

    // init the event handler
    auto* rcv_ehd = new RecvPacketEventHadndlerData(
        stream,
        source_flow.src,
        id,
        layer_num * flow_model.chunk_count * m_channels +
        flow_model.chunk_count * flow_model.channel_id
          + flow_model.chunk_id,
        EventType::PacketReceived,
        data_source,
        1);
    rcv_ehd->flow_id = source_flow.flow_id;
    rcv_ehd->channel_id = channel_id;
    rcv_ehd->flowTag.tag_id = rcv_ehd->tag;
    rcv_ehd->flowTag.channel_id = channel_id;
    rcv_ehd->flowTag.flow_size = source_flow.flow_size;
    rcv_ehd->flowTag.chunk_id = source_flow.chunk_id;
    rcv_ehd->flowTag.sender_node = data_source;
    rcv_ehd->flowTag.receiver_node = id;
    if (this->comType == ComType::All_Reduce_NVLS)
      rcv_ehd->flowTag.nvls_on = true;
    else
      rcv_ehd->flowTag.nvls_on = false;

    stream->owner->front_end_sim_recv(
        0,
        Sys::dummy_data,
        source_flow.flow_size,
        UINT8,
        data_source,
        rcv_ehd->tag,
        &rcv_req,
        &Sys::handleEvent,
        rcv_ehd);
  }
  return true;
}

void NcclTreeFlowModel::release_packets(int channel_id, int flow_id, uint64_t message_size) const {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (NPU_to_MA == true) {
    (new PacketBundle(
         stream->owner,
         stream,
         {},
         processed,
         send_back,
         message_size,
         transmition,
         channel_id,
         flow_id))->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         {},
         processed,
         send_back,
         message_size,
         transmition,
         channel_id,
         flow_id))->send_to_NPU();
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG, "id:  %d finish release_packets", id);
}

void NcclTreeFlowModel::process_stream_count(int channel_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
#ifdef PHY_MTP
  send_packets--;
#else
  FlowCriticalSection cs;
  if (_stream_count[channel_id] > 0) {
    _stream_count[channel_id]--;
  }
  NcclLog->writeLog(
      NcclLogLevel::DEBUG,
      "NcclTreeFlowModel::process_stream_count channel_id %d _stream_count %d",
      channel_id,
      _stream_count[channel_id]);
  if (_stream_count[channel_id] == 0 && stream->state != StreamState::Dead)
    stream->changeState(StreamState::Zombie);
#endif
}

void NcclTreeFlowModel::reduce(int channel_id, int flow_id) {
  process_stream_count(channel_id);
#ifndef PHY_MTP
  if (!packets[std::make_pair(channel_id, flow_id)].empty()) {
    packets[std::make_pair(channel_id, flow_id)].pop_front();
  }
#endif
}

void NcclTreeFlowModel::insert_packets(int channel_id, int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  assert(channel_id < m_channels);
  if (!enabled) {
    return;
  }
  assert(_flow_models.count(std::make_pair(channel_id, flow_id)) != 0);

  const MockNccl::SingleFlow f = _flow_models[std::make_pair(channel_id, flow_id)];
  assert(zero_latency_packets->count(channel_id) != 0 && non_zero_latency_packets->count(channel_id) != 0);
  if ((*zero_latency_packets)[channel_id] == 0 && (*non_zero_latency_packets)[channel_id] == 0) {
    (*zero_latency_packets)[channel_id] = parallel_reduce * 1;
    (*non_zero_latency_packets)[channel_id] = get_non_zero_latency_packets();
    toggle = !toggle;
  }
  const int current_receiver = f.dest;
  if ((*zero_latency_packets)[channel_id] > 0) {
    NcclLog->writeLog(NcclLogLevel::DEBUG, "id:  %d (*zero_latency_packets)[channel_id] > 0", id);
    const uint64_t message_size = f.flow_size;
    packets[std::make_pair(channel_id, flow_id)].push_back(
        MyPacket(
            stream->current_queue_id,
            -1 /* this is not used by NcclTreeFlowModel */,
            current_receiver,
            message_size,
            channel_id,
            flow_id));
    packets[std::make_pair(channel_id, flow_id)].back().set_flow_id(flow_id);
    packets[std::make_pair(channel_id, flow_id)].back().sender = nullptr;
    processed = false;
    send_back = false;
    NPU_to_MA = true;
    release_packets(channel_id, flow_id, message_size);
    (*zero_latency_packets)[channel_id]--;
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "id:  %d (*zero_latency_packets)[channel_id] : %d ",
        id,
        (*zero_latency_packets)[channel_id]);
    return;
  } else if ((*non_zero_latency_packets)[channel_id] > 0) {
    NcclLog->writeLog(NcclLogLevel::DEBUG, "id:  %d (*non_zero_latency_packets)[channel_id] > 0", id);
    uint64_t message_size = f.flow_size;
    packets[std::make_pair(channel_id, flow_id)].push_back(
        MyPacket(
            stream->current_queue_id,
            -1,
            current_receiver,
            message_size,
            channel_id,
            flow_id));
    packets[std::make_pair(channel_id, flow_id)].back().set_flow_id(flow_id);
    packets[std::make_pair(channel_id, flow_id)].back().sender = nullptr;
    if (comType == ComType::Reduce_Scatter || (comType == ComType::All_Reduce && toggle)) {
      processed = true;
    } else {
      processed = false;
    }
    if ((*non_zero_latency_packets)[channel_id] <= parallel_reduce * 1) {
      send_back = false;
    } else {
      send_back = true;
    }
    NPU_to_MA = false;
    release_packets(channel_id, flow_id, message_size);
    (*non_zero_latency_packets)[channel_id]--;
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "id:  %d (*non_zero_latency_packets)[channel_id] : %d ",
        id,
        (*non_zero_latency_packets)[channel_id]);
    return;
  }
  Sys::sys_panic("should not inject nothing!");
}

bool NcclTreeFlowModel::ready(int channel_id, int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  MyPacket packet;
  #ifndef PHY_RDMA
  {
    if (stream->state == StreamState::Created || stream->state == StreamState::Ready) {
      stream->changeState(StreamState::Executing);
    }
    if (!enabled || packets[std::make_pair(channel_id, flow_id)].empty() || _stream_count[channel_id] == 0) {
      NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel not ready!");
      return false;
    }
    packet = packets[std::make_pair(channel_id, flow_id)].front();
  }
  #endif
  MockNccl::SingleFlow flow_model = _flow_models[std::make_pair(channel_id, flow_id)];
  std::vector<int> data_sources = flow_model.prev;
  if (flow_model.conn_type == "PTP" && flow_model.dest == id) {
    // direct flows in AllToAll
    data_sources.push_back(flow_model.src);
  }
  NcclLog->writeLog(NcclLogLevel::INFO, "id %d ready() handle flow_id %u", id, flow_id);

  // if this is a root flow, call sim_recv for the flow with the same chunk id; otherwise, use the next chunk id
  bool is_first = false;
  if (flow_model.parent_flow_id.empty() || comType == ComType::All_to_All || flow_model.conn_type == "RING") {
    is_first = true;
  }

  for (int data_source : data_sources) {
    // find the source flow
    MockNccl::SingleFlow source_flow;
    if (data_source != id) {
      auto it = std::find_if(_flow_models.begin(), _flow_models.end(),
     [&](const std::pair<std::pair<int, int>, MockNccl::SingleFlow>& entry) {
         const auto& flow = entry.second;
         return flow.src == data_source &&
                flow.dest == id &&
                flow.channel_id == channel_id &&
                flow.chunk_id == flow_model.chunk_id + (is_first ? 0 : 1);
         });
      if (it == _flow_models.end()) {
        continue;
      }
      source_flow = it->second;
    } else {
      source_flow = flow_model;
    }

    // FIXME this is never used
    sim_request rcv_req;

    // init the event handler
    auto* ehd = new RecvPacketEventHadndlerData(
        stream,
        source_flow.src,
        id,
        layer_num * flow_model.chunk_count * m_channels +
        source_flow.chunk_count * source_flow.channel_id
          + source_flow.chunk_id,
        EventType::PacketReceived,
        #ifndef PHY_RDMA
        packet.preferred_vnet,
        packet.stream_num);
        #else
        stream->current_queue_id,
        1);
        #endif

    ehd->flow_id = source_flow.flow_id;
    ehd->channel_id = channel_id;
    ehd->flowTag.tag_id = ehd->tag;
    ehd->flowTag.channel_id = channel_id;
    ehd->flowTag.flow_size = source_flow.flow_size;
    ehd->flowTag.chunk_id = source_flow.chunk_id;
    ehd->flowTag.sender_node = data_source;
    ehd->flowTag.receiver_node = id;
    ehd->flowTag.nvls_on = comType == ComType::All_Reduce_NVLS;

    if (free_packets[std::make_pair(channel_id, data_source)] > 0) {
      stream->owner->front_end_sim_recv(
          0,
          Sys::dummy_data,
          source_flow.flow_size,
          UINT8,
          data_source,
          ehd->tag,
          &rcv_req,
          &Sys::handleEvent,
          ehd);
    }
  }
  if (flow_model.dest == id) {
    return true;
  }

  // TODO is this needed? It is used only by the physical frontend
  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = flow_model.dest;
  snd_req.reqType = UINT8;
  snd_req.vnet = stream->current_queue_id;
  snd_req.layerNum = layer_num;
  snd_req.reqCount = flow_model.flow_size;
  snd_req.flowTag.tag_id =
      layer_num * flow_model.chunk_count * m_channels +
      flow_model.channel_id * flow_model.chunk_count +
      flow_model.chunk_id;
  snd_req.flowTag.channel_id = channel_id;
  snd_req.flowTag.flow_size = flow_model.flow_size;
  snd_req.flowTag.current_flow_id = flow_id;
  snd_req.flowTag.chunk_id = flow_model.chunk_id;
  snd_req.flowTag.sender_node = id;
  snd_req.flowTag.receiver_node = flow_model.dest;
  snd_req.flowTag.nvls_on = comType == ComType::All_Reduce_NVLS;

  // init the event handler
  auto* snd_ehd = new SendPacketEventHandlerData(
      stream,
      id,
      flow_model.dest,
      snd_req.flowTag.tag_id,
      EventType::PacketSentFinshed);
  snd_ehd->flow_id = flow_id;
  snd_ehd->channel_id = channel_id;
  snd_ehd->flowTag = snd_req.flowTag;

  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      snd_req.reqCount,
      UINT8,
      flow_model.dest,
      snd_ehd->tag,
      &snd_req,
      &Sys::handleEvent,
      snd_ehd);
  return true;
}

void NcclTreeFlowModel::exit() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
#ifdef PHY_MTP
  auto now = std::chrono::system_clock::now();
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel exit time %lld", now_us);
  end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  NcclLog->writeLog(NcclLogLevel::DEBUG, "Communication Latency：%lld us", duration.count());
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);
#else
  for (std::pair<std::pair<int, int>, std::list<MyPacket>> packet : packets) {
    if (!packet.second.empty())
      packet.second.clear();
  }
#endif
  stream->owner->proceed_to_next_vnet_baseline(reinterpret_cast<StreamBaseline*>(stream));
  NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel exit");
}

#ifdef PHY_RDMA
bool NcclTreeFlowModel::phy_iteratable(int channel_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  bool all_send_finished = true, all_recv_finished = true;
  bool exit_flag = true;
  if (send_packets != 0 || recv_packets != 0) {
    exit_flag = false;
  }
  if (exit_flag) {
    judge_exit_flag.store(true);
    return false;
  } else {
    return true;
  }
}

void NcclTreeFlowModel::waiting_to_exit() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel::waiting_to_exit begin ");
  while (!judge_exit_flag) {
  };
  exit();
  return;
}
#endif
} // namespace AstraSim
