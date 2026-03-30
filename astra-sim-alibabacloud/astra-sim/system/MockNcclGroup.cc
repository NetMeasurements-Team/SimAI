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
#include "MockNcclGroup.h"
#include "MockNcclChannel.h"
#include <functional>
#include<vector>
#include<map>
#include<set>
#include <queue>
#include <cmath>
#include <algorithm>
#include "astra-sim/system/MockNcclLog.h"
#include <filesystem>
// #include <tuple>

using namespace std;

namespace MockNccl {

  struct ChunkInfo {
      int sub_chunk_idx;
      int src_offset;
      int count;
  };

  void logFlowModels(
      const NcclLogLevel level,
      const std::string& algorithmName,
      const std::map<int, std::shared_ptr<FlowModels>>& rank2pflowmodels) {

    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    // Header for the log output
    NcclLog->writeLog(level, "=======================================================================");
    NcclLog->writeLog(level, "--- Flow Model Details for: %s ---", algorithmName.c_str());
    NcclLog->writeLog(level, "=======================================================================");

    // Iterate over each rank that has associated flows.
    for (const auto& rank_pair : rank2pflowmodels) {
      int rank = rank_pair.first;
      const auto& models_ptr = rank_pair.second;

      if (!models_ptr)
        continue;

      std::stringstream header_ss;
      header_ss << "--- Flows visible to Rank " << rank << " ---";
      NcclLog->writeLog(level, "%s", header_ss.str().c_str());

      // Copy flows to a vector to sort them by flow_id for readability.
      std::vector<SingleFlow> sorted_flows;
      for (const auto& flow_pair : *models_ptr) {
        sorted_flows.push_back(flow_pair.second);
      }
      std::sort(sorted_flows.begin(), sorted_flows.end(), [](const SingleFlow& a, const SingleFlow& b) {
        return a.flow_id < b.flow_id;
      });

      // Log the details of each sorted flow.
      for (const auto& flow : sorted_flows) {
        std::stringstream flow_ss;
        flow_ss << "  Flow ID: " << flow.flow_id << "\t| Type: " << flow.conn_type << "\t| Ch: " << flow.channel_id
                << "\t| Chunk ID: " << flow.chunk_id << "\t| " << flow.src << " -> " << flow.dest
                << "\t| Size: " << flow.flow_size << "\t| Parents: " << AstraSim::asPrintable(flow.parent_flow_id)
                << "\t| Children: " << AstraSim::asPrintable(flow.child_flow_id)
                << "\t| prev: " << AstraSim::asPrintable(flow.prev);
        NcclLog->writeLog(level, "%s", flow_ss.str().c_str());
      }
    }
    NcclLog->writeLog(level, "=======================================================================");
  }

  MockNcclGroup::MockNcclGroup(int _ngpus,int _gpus_per_nodes,int _TP_size,int _DP_size,int _PP_size,int _EP_size,int _DP_EP_size,std::vector<int>_NVSwitch,GPUType _gpu_type):g_flow_id(0),gpu_type(_gpu_type){
    /*init groups
    */
    MockNcclLog *NcclLog = MockNcclLog::getInstance();
    if (_ngpus % _gpus_per_nodes != 0 || _ngpus / _gpus_per_nodes <= 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"The number of GPUs used is not a multiple of the number of GPUs per node.");
      return;
    }
    int all_group_idx = 0;
    int nNodes = _ngpus/_gpus_per_nodes;
    int nlocalranks = _gpus_per_nodes;
    int TP_nums = _ngpus/_TP_size;
    int DP_nums = _ngpus/_DP_size;
    int PP_nums = _ngpus/_PP_size;
    int EP_nums = _ngpus/_EP_size;
    int DP_EP_nums = _ngpus/_DP_EP_size;
    if (TP_nums <= 0 || DP_nums <= 0 || PP_nums <= 0 || EP_nums <= 0 || DP_EP_nums <= 0 || (_TP_size * _DP_size * _PP_size != _ngpus) || (_EP_size * _DP_EP_size != _DP_size)){
      NcclLog->writeLog(NcclLogLevel::ERROR,"The combination of parallelism groups is incorrect.");
      std::cerr << "The combination of parallelism groups is incorrect." << std::endl;
      exit(1);
    }
    int nNodesPerTPGroup = _TP_size / nlocalranks + (_TP_size % nlocalranks > 0 ? 1 : 0);
    std::vector<int>ranks;
    std::vector<int>NVSwitchs;
    // init TP group 
    if(_TP_size>1){
      std::set<int>TPnodes;
      for(int i =0;i<TP_nums;i++){
        ranks.clear();
        TPnodes.clear();
        for(int j =0;j<_TP_size;j++){
          int rank = i*_TP_size+j;
          ranks.push_back(rank);
          GroupIndex[std::make_pair(rank, TP)] = all_group_idx;
          int node_idx = rank / _gpus_per_nodes;
          TPnodes.insert(node_idx);
        }
        NVSwitchs.clear();
        for(int idx:TPnodes){
          NVSwitchs.push_back(_NVSwitch[idx]);
          GroupIndex[std::make_pair(_NVSwitch[idx],TP)] = all_group_idx;
        }
        AllGroups[all_group_idx]=GroupInfo(all_group_idx,TP,nNodesPerTPGroup,_TP_size,ranks,NVSwitchs);
        all_group_idx ++;
      }
    }
    // init DP group
    if(_DP_size>1){
      std::set<int>DPnodes;
      for(int i =0;i<DP_nums;i++){
        ranks.clear();
        DPnodes.clear();
        for(int j =0;j<_DP_size;j++){
          int rank = i+j*DP_nums;
          ranks.push_back(rank);
          GroupIndex[std::make_pair(rank, DP)] = all_group_idx;
          int node_idx = rank/_gpus_per_nodes;
          DPnodes.insert(node_idx);
        }
        NVSwitchs.clear();
        for(int idx:DPnodes){
          NVSwitchs.push_back(_NVSwitch[idx]);
          GroupIndex[std::make_pair(_NVSwitch[idx],DP)] = all_group_idx;
        }
        AllGroups[all_group_idx]=GroupInfo(all_group_idx,DP,DPnodes.size(),_DP_size,ranks,NVSwitchs);
        all_group_idx ++;
      }
    }
    // init PP group
    if(_PP_size > 1){

    }
    // init EP
    std::map<int,GroupInfo> AllTPGroups;
    if (_TP_size > 1) {
      for (auto it = AllGroups.begin(); it != AllGroups.end(); ++it) {
        if (it->second.type == TP) {
          AllTPGroups[it->second.group_index] = it->second;
        }
      }
    } else if (_TP_size == 1) {
      // init with single-node TP groups in case TP = 1
      NVSwitchs.clear();
      for (int i = 0; i < _ngpus; i++) {
        ranks.clear();
        ranks.push_back(i);
        AllTPGroups[i] = GroupInfo(i, TP, 1, 1, ranks, NVSwitchs);
      }
    }
    if(_EP_size>1){
      int TP_idx=0;
      std::set<int> EPnodes;
      for (int i = 0; i < TP_nums / _EP_size; i++){
        TP_idx = i*_EP_size;
        for(int k = 0;k<AllTPGroups[TP_idx].Ranks.size();k++){
          ranks.clear();
          EPnodes.clear();
          for(int l = TP_idx;l<TP_idx+_EP_size;l++){
            int tmp_rank = AllTPGroups[l].Ranks[k];
            int node_idx = tmp_rank/_gpus_per_nodes;
            ranks.push_back(tmp_rank);
            GroupIndex[std::make_pair(tmp_rank, EP)] = all_group_idx;
            EPnodes.insert(node_idx);
          }
          NVSwitchs.clear();
          for(int idx:EPnodes){
            NVSwitchs.push_back(_NVSwitch[idx]);
            GroupIndex[std::make_pair(_NVSwitch[idx],EP)] = all_group_idx;
          }
          AllGroups[all_group_idx] = GroupInfo(all_group_idx,EP,EPnodes.size(),_EP_size,ranks,NVSwitchs);
          all_group_idx++;
        }
      }
    }
    //init EP_DP
    if (_DP_EP_size > 1){
      int TP_idx = 0;
      std::set<int> DP_EP_nodes;
      for (int i = 0; i < TP_nums / _DP_EP_size; i++){
        TP_idx = i;
        for (int k = 0; k < AllTPGroups[TP_idx].Ranks.size(); k++){
          ranks.clear();
          DP_EP_nodes.clear();
          for (int l = TP_idx; l < TP_idx + _DP_EP_size * _EP_size; l += _EP_size){
            int tmp_rank = AllTPGroups[l].Ranks[k];
            int node_idx = tmp_rank / _gpus_per_nodes;
            ranks.push_back(tmp_rank);
            GroupIndex[std::make_pair(tmp_rank, DP_EP)] = all_group_idx;
            DP_EP_nodes.insert(node_idx);
          }
          NVSwitchs.clear();
          for (int idx : DP_EP_nodes){
            NVSwitchs.push_back(_NVSwitch[idx]);
            GroupIndex[std::make_pair(_NVSwitch[idx], DP_EP)] = all_group_idx;
          }
          AllGroups[all_group_idx] = GroupInfo(all_group_idx, DP_EP, DP_EP_nodes.size(), _DP_EP_size, ranks, NVSwitchs);
          all_group_idx++;
        }
      }
    }

    std::cout << "*********************    AllGroups:    *********************" << std::endl;
    for (const auto& [group_index, group_info] : AllGroups) {
      std::cout << "Group#" << group_info.group_index
                << " Type=" << to_cstr(group_info.type)
                << " Nodes=" << group_info.nNodes
                << " Ranks=" << group_info.nRanks
                << " " << AstraSim::asPrintable(group_info.Ranks)
                << " NVSwitches=" << AstraSim::asPrintable(group_info.NVSwitchs)
                << "\n";
    }
    std::cout << "************************************************************" << std::endl;

    return;
  }
  
  void MockNcclGroup::generateringchannels(std::map<int, std::vector<int>> localrings, MockNccl::GroupInfo* groupInfo, std::map<int, std::map<int, std::vector<int>>>& ringchannels) {
    std::map<int,std::vector<int>>::iterator ring_it;
    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes = groupInfo->nNodes;
    int nlocalRanks = groupInfo->nRanks/nNodes;
    int delta = nNodes > 1 ? groupInfo->Ranks[nlocalRanks]-groupInfo->Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];

    }
  }

  std::map<int, std::vector<int>> MockNcclGroup::gen_local_ring(int rank, GroupType type){
    GroupInfo gp_info;
    int gp_idx;
    std::vector<int>ranks;
    std::vector<int>localranks;
    std::map<int,std::vector<int>>localrings;
    int nNodes;
    int nlocalranks;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type)) == 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no relevant group info, resulting in an error in gen_local_ring");
      return {};
    } 
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    ranks = gp_info.Ranks;
    nNodes = gp_info.nNodes;
    nlocalranks = ranks.size()/nNodes;
    std::sort(ranks.begin(), ranks.end());
    for (int i = 0; i < nlocalranks; i++){
      localranks.push_back(ranks[i]);
    }
    for(int i =0;i<nlocalranks;i++){
      std::vector<int> vec;
      for (int j = 0; j < nlocalranks; ++j) {
        vec.push_back(localranks[(i + j) % nlocalranks]);
      }
      localrings[i] = vec;
    }
    return localrings;
  }

  RingChannels MockNcclGroup::genringchannels(int rank, MockNccl::GroupType type) {
    std::map<int,std::map<int,std::vector<int>>>ringchannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    int gp_idx;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes;
    int nlocalRanks;
    int delta;
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"No corresponding group information is generated, and there is an error in creating the ring channel.");
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[GroupIndex[std::make_pair(rank,type)]];
    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);

    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];
    }
    Allringchannels[gp_idx]=ringchannels;
    return ringchannels;
  }

  std::shared_ptr<void> MockNcclGroup::getFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size,int layer_num,State loopstate, bool& msccl, bool NCCL_Simple_LL_splitting){
    std::string flow_model_name;
    std::string flow_model_name_msccl;
    GroupInfo gp_info;
    int gp_idx;
    int end_rank;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return nullptr;
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    switch (type){
      case TP:
        flow_model_name = "TP";
        break;
      case DP:
        flow_model_name = "DP";
        break;
      case EP:
        flow_model_name = "EP";
        break;
      case DP_EP:
        flow_model_name = "DP_EP";
        break;
      default:
        break;
    }
    flow_model_name = flow_model_name + "_" + std::to_string(gp_idx) + "_" + std::to_string(layer_num) + "_" + std::to_string(static_cast<int>(loopstate)) + "_" + std::to_string(static_cast<int>(op)) + "_" + std::to_string(data_size);
    flow_model_name_msccl= flow_model_name + "_1";
    flow_model_name += "_0";
    // 0 is no msccl and 1 is with msccl
    if(flow_models.count(flow_model_name)){
      FlowName2nums[flow_model_name] ++;
      std::shared_ptr<void> presult;
      msccl = false;
      if(flow_models[flow_model_name].count(rank)!=0){
        // already found without msccl
        presult = flow_models[flow_model_name][rank];
      }else{
        presult = nullptr;
      }
      return presult;
    } else if(flow_models.count(flow_model_name_msccl)){
      NcclLog->writeLog(NcclLogLevel::DEBUG,"Found flow model with msccl for rank %d, type %d, op %d, data_size %lu, layer_num %d, loopstate %d.", rank, static_cast<int>(type), static_cast<int>(op), data_size, layer_num, static_cast<int>(loopstate));
      FlowName2nums[flow_model_name_msccl] ++;
      std::shared_ptr<void> presult;
      if(flow_models[flow_model_name_msccl].count(rank)!=0){
        // already found with msccl
        presult = flow_models[flow_model_name_msccl][rank];
      }else{
        presult = nullptr;
      }
      return presult;
    } else {
      auto result = genFlowModels(type,rank,op,data_size,msccl,NCCL_Simple_LL_splitting);
      if(msccl){
        NcclLog->writeLog(NcclLogLevel::DEBUG,"Generated flow model with msccl for rank %d, type %d, op %d, data_size %lu, layer_num %d, loopstate %d.", rank, static_cast<int>(type), static_cast<int>(op), data_size, layer_num, static_cast<int>(loopstate));
        //delete the old flow model without msccl
        if(flow_models.count(flow_model_name)){
          flow_models.erase(flow_model_name);
          FlowName2nums.erase(flow_model_name);
        }
        //add the new flow model with msccl
        flow_models[flow_model_name_msccl] = result;
        FlowName2nums[flow_model_name_msccl]= 1;
        return flow_models[flow_model_name_msccl][rank];
      }else{
        NcclLog->writeLog(NcclLogLevel::DEBUG,"Generated flow model without msccl for rank %d, type %d, op %d, data_size %lu, layer_num %d, loopstate %d.", rank, static_cast<int>(type), static_cast<int>(op), data_size, layer_num, static_cast<int>(loopstate));
        flow_models[flow_model_name] = result;
        FlowName2nums[flow_model_name]= 1;
        return flow_models[flow_model_name][rank];
      }
    }
  }

   unsigned int MockNcclGroup::countFlowsInFlowModels(const map<int, shared_ptr<FlowModels>>& flowModelsMap) {
    unsigned int count = 0;
    for (auto& [_, flowModels] : flowModelsMap) {
      count += flowModels->size();
    }
    return count/2;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genFlowModels(GroupType type, int rank, AstraSim::ComType op,uint64_t data_size, bool& msccl, bool NCCL_Simple_LL_splitting){
    switch (op) {
      case AstraSim::ComType::All_Reduce:
        return genAllReduceFlowModels(type,rank,data_size,msccl);
      case AstraSim::ComType::All_Gather:
        return genAllGatherFlowModels(type,rank,data_size,msccl,NCCL_Simple_LL_splitting);
      case AstraSim::ComType::Reduce_Scatter:
        return genReduceScatterFlowModels(type,rank,data_size,msccl);
      case AstraSim::ComType::All_to_All:
        return genAlltoAllFlowModels(type,rank,data_size,msccl);
      default:
        break;
    }
    return {};
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAlltoAllFlowModels(GroupType type, int rank, uint64_t data_size, bool& msccl){
    msccl = false;
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    nranks = gp_info.nRanks;
    size_t ranks_per_node = gp_info.nRanks / gp_info.nNodes;
    chunkcount = nranks - 1;
    chunksize = data_size / nranks;
    data_size = data_size / nranks;
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    for (int i = 0; i < gp_info.Ranks.size(); i++) {
      for (int j = 0; j < gp_info.Ranks.size(); j++) {
        if (i == j) {
          continue;
        }
        std::vector<int> prev;
        int src = gp_info.Ranks[i];
        vector<int> parent_flows = {};
        string connection_type = "PTP";
        uint32_t channel_id = 0;
        if (PXN_ENABLE
            && gp_info.Ranks[i] / ranks_per_node != gp_info.Ranks[j] / ranks_per_node
            && gp_info.Ranks[i] % ranks_per_node != gp_info.Ranks[j] % ranks_per_node) {
          int pxn_rank = gp_info.Ranks[j] % ranks_per_node + ranks_per_node * (gp_info.Ranks[i] / ranks_per_node);
          channel_id = (AstraSim::pos_mod(
              gp_info.Ranks[j] / ranks_per_node - gp_info.Ranks[i] / ranks_per_node,
              gp_info.nNodes) - 1) * ranks_per_node;
          channel_id += AstraSim::pos_mod(
              gp_info.Ranks[j] % ranks_per_node - gp_info.Ranks[i] % ranks_per_node,
              ranks_per_node);
          tmp_result = SingleFlow(
              g_flow_id,
              gp_info.Ranks[i],
              pxn_rank,
              chunksize,
              {},
              {},
              {g_flow_id + 1},
              channel_id,
              0,
              1,
              "PTP_PXN_START");
          result[std::make_pair(channel_id, g_flow_id)] = tmp_result;
          src = pxn_rank;
          parent_flows.push_back(g_flow_id);
          connection_type = "PTP_PXN_END";
          prev = {gp_info.Ranks[i]};
          g_flow_id++;
        }
        tmp_result = SingleFlow(
            g_flow_id,
            src,
            gp_info.Ranks[j],
            chunksize,
            prev,
            parent_flows,
            {},
            channel_id,
            0,
            1,
            connection_type);
        result[std::make_pair(channel_id, g_flow_id)] = tmp_result;
        g_flow_id++;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logFlowModels(NcclLogLevel::INFO, "AllToAll", rank2pflowmodels);
    return rank2pflowmodels;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genReduceScatterFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size, bool& msccl) {
    msccl = false;
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int, SingleFlow> task_list = {}; 
    std::map<int, SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    nranks = gp_info.nRanks;
    chunkcount = nranks - 1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    for (auto it = ringchannels.begin(); it != ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if (rank_it->second[0] != -1)
              prevranks = {rank_it->second[0]};
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            if (rank_it->first != -1) {
              prevranks = {rank_it->first};
            } else {
              prevranks = {};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              if(rank_it->first!=-1){
                prevranks={rank_it->first};
              }else{
                prevranks ={};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if (prenoderecvrank != -1) {
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id .push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks= {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceFlowModels(GroupType type , int rank,uint64_t data_size, bool& msccl){
    msccl = false;
    ncclInfo* ncc_info = get_algo_proto_info(type,rank,AstraSim::ComType::All_Reduce,data_size);
    switch (ncc_info->algorithm) {
      case NCCL_ALGO_TREE:
      case NCCL_ALGO_RING:
        return genAllReduceRingFlowModels(type, rank, data_size);
      case NCCL_ALGO_NVLS:
        return genAllreduceNVLSFlowModels(type,rank,data_size);
      case NCCL_ALGO_NVLS_TREE:
        return {};
      default:
        return {};
    }
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genallReduceNVLSTreeFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    GroupInfo gp_info;
    int gp_idx;
    int chunk_count = 1;
    int chunk_size;
    NVLStreechannels nvlstreechannels;
    NVLStreechannels::iterator nvlstree;
    FlowModels result = {};
    if(GroupIndex.count(std::make_pair(rank,type)) == 0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no relevant group info, resulting in an error in generating genallReduceNVLSTreeFlowModels.");
      return nullptr;
    } 
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    nvlstreechannels = AllNVLStreechannels[gp_idx];
    NcclLog->writeLog(NcclLogLevel::DEBUG," nvlstreechannels.size()  %d",nvlstreechannels.size());
    chunk_size = data_size / nvlstreechannels.size() / chunk_count;
    for (nvlstree = nvlstreechannels.begin();
         nvlstree != nvlstreechannels.end();
         nvlstree++) {
      std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
      if (rank == 0) {
        for (nvlstreenodes_it = nvlstree->second.begin();
             nvlstreenodes_it != nvlstree->second.end();
             nvlstreenodes_it++) {
          NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
          int i = 0;
          for (auto nvlstreenode : nvlstreenodes_it->second) {
            NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
            if(nvlstreenode->up!=nullptr){
              NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
            }
            NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
            for (auto down : nvlstreenode->down) {
              NcclLog->writeLog(NcclLogLevel::DEBUG,"%d ",down->rank);
            }
          }
        }
      }
      std::unordered_map<ncclChannelNode*, int> upinDegree;
      std::unordered_map<ncclChannelNode*, int> downinDegree;
      std::unordered_map<ncclChannelNode*, std::vector<int>> nodeprevs;
      for (int ck = 0; ck < chunk_count; ck++) {
        nodeprevs = {};
        std::vector<ncclChannelNode*> ncclchannelnodes;
        for (auto nvlstreenodes : nvlstree->second) {
          for (auto nvlstreenode : nvlstreenodes.second) {
            ncclchannelnodes.push_back(nvlstreenode);
            upinDegree[nvlstreenode] = nvlstreenode->down.size();
            if (nvlstreenode->up == nullptr)
              downinDegree[nvlstreenode] = 0;
            else
              downinDegree[nvlstreenode] = 1;
          }
        }
        generate_flow_model_nvls_tree_allreduce_up(
            ncclchannelnodes,
            upinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
        generate_flow_model_nvls_tree_allreduce_down(
            ncclchannelnodes,
            downinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result =
        std::make_shared<FlowModels>(result);
    return ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_up(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> upinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
        nodeprevs[entry.first] = {};
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();
      if (current->up != nullptr) {
        upinDegree[current->up]--;
        std::vector<int> _prev;
        if (current->down.size() == 0)
          _prev = {current->up->rank};
        else {
          for (auto down : current->down) {
            _prev.push_back(down->rank);
          }
        }
        tmp_result = SingleFlow(
            g_flow_id,
            current->rank,
            current->up->rank,
            chunk_size,
            _prev,
            nodeprevs[current],
            {},
            channle_id,
            chunk_id,
            chunk_count,
            conn_tag);
        for (int parent_flow_id : nodeprevs[current]) {
          result[std::make_pair(channle_id, parent_flow_id)]
              .child_flow_id.push_back(g_flow_id);
        }
        result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current->up].push_back(tmp_result.flow_id);
        nodeprevs.erase(current);
        if (upinDegree[current->up] == 0)
          q.push(current->up);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_down(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> downinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();

      if (current->down.size() > 0) {
        for (ncclChannelNode* down : current->down) {
          downinDegree[down]--;
          std::vector<int> _prev;
          if (current->up == nullptr) {
            for (ncclChannelNode* down1 : current->down) {
              _prev.push_back(down1->rank);
            }
          } else {
            _prev = {current->up->rank};
          }
          tmp_result = SingleFlow(
              g_flow_id,
              current->rank,
              down->rank,
              chunk_size,
              _prev,
              nodeprevs[current],
              {},
              channle_id,
              chunk_id,
              chunk_count,
              conn_tag);
          for (int parent_flow_id : nodeprevs[current]) {
            result[std::make_pair(channle_id, parent_flow_id)]
                .child_flow_id.push_back(g_flow_id);
          }
          result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id);
          if (downinDegree[down] == 0)
            q.push(down);
        }
      }
    }
    return result;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllreduceNVLSFlowModels(GroupType type,int rank,uint64_t data_size){
    GroupInfo gp_info;
    int gp_idx;
    int chunk_count = 4;
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info , resulting in an error in genAllreduceNVLSFlowModels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    FlowModels result={};
    SingleFlow treeflow;
    if(gp_info.nNodes == 1){  
      std::vector<int>NVswitchs = gp_info.NVSwitchs;
      std::vector<int>ranks = gp_info.Ranks;
      int chunk_size = data_size / chunk_count;
      for(int ck =0;ck<chunk_count;ck++){
        for(int j = 0;j<NVswitchs.size();j++){
          std::vector<int>prevs;
          std::vector<int>parents;
          for(int k = 0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,ranks[k],NVswitchs[j],chunk_size,{NVswitchs[j]},{},{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            prevs.push_back(ranks[k]);
            parents.push_back(g_flow_id);
            g_flow_id++;
          }
          for(int k =0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,NVswitchs[j],ranks[k],chunk_size,prevs,parents,{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            for(auto parent:parents){
              result[std::make_pair(0,parent)].child_flow_id.push_back(g_flow_id);
            }
            g_flow_id++;
          }
        }

      }

    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genAllReduceTreeFlowModels(GroupType type , int rank,uint64_t data_size){
    int chunk_count = 64;
    int chunk_size;
    SingleFlow tmp_result;
    FlowModels result1 = {};
    FlowModels result = {};
    std::map<int,int> task_list = {}; 
    std::map<int,std::map<int,ncclTree>>::iterator tree;
    GroupInfo gp_info;
    int gp_idx;
    TreeChannels treechannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    if(GroupIndex.count(std::make_pair(rank,type))==0||Alltreechannels.count(gp_idx)==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info , resulting in an error in genAllreduceNVLSFlowModels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    treechannels = Alltreechannels[gp_idx];
    chunk_size = data_size / treechannels.size() / chunk_count;
    for(tree = treechannels.begin(); tree !=treechannels.end(); tree++) {
      std::unordered_map<int, int> upinDegree;
      std::unordered_map<int, int> downinDegree;
      std::unordered_map<int,std::vector<int>> nodeprevs;
      for(int ck = 0; ck < chunk_count; ck++){
        nodeprevs = {};
        for(auto treenode:tree->second){
          upinDegree[treenode.first] = treenode.second.down.size();
          if(treenode.second.up == -1)
            downinDegree[treenode.first] = 0;
          else
            downinDegree[treenode.first] = 1;
        }
        generate_flow_model_tree_allreduce_up(tree->second,upinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
        generate_flow_model_tree_allreduce_down(tree->second,downinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result = std::make_shared<FlowModels>(result);
    return  ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_up(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> upinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
        nodeprevs[entry.first]={};
      }
    }
    std::string conn_tag = "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.up != -1) {
        upinDegree[current.up]--; 
        std::vector<int> _prev; 
        if (current.down.size() == 0)
          _prev = {current.up};
        else
          _prev = current.down;
        tmp_result = SingleFlow(g_flow_id,current.rank,current.up,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
        for(int parent_flow_id:nodeprevs[current.rank])
          result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);  
        result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current.up].push_back(tmp_result.flow_id); 
        nodeprevs.erase(current.rank); 
        if(upinDegree[current.up] == 0)
          q.push(nodes[current.up]);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_down(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> downinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
      }
    }
    std::string conn_tag =  "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.down.size() >0 ) {
        for(int down:current.down) {
          downinDegree[down] --;
          std::vector<int> _prev;
          if (current.up == -1) {
            _prev = current.down;
          } else {
            _prev = {current.up};
          }
          tmp_result = SingleFlow(g_flow_id,current.rank,down,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
          for(int parent_flow_id:nodeprevs[current.rank])
            result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);
          result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id); 
          if(downinDegree[down] == 0)
            q.push(nodes[down]);
        }
      }
    }
    return result;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceRingFlowModels(GroupType type , int rank,uint64_t data_size){
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {}; 
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }
    nranks = gp_info.nRanks;
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    int chunkcout = 2*(gp_info.nRanks-1);

    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      int chunk_id = 0;
      while (send_size < data_size)
      {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            prevranks = {rank_it->first};          
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if (prenoderecvrank != -1) {
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunk_id++;
        for(int i =0; i < nranks -1; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              prevranks={rank_it->first};
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1)
            {
              prevranks ={rank_it->second[0]};
            }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if (rank_it->first != -1) {
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        send_size += real_chunksize;
      }
    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logRankToFlowModels(NcclLogLevel::INFO, "RING AllReduce", rank2pflowmodels);
    return rank2pflowmodels;
  }

  void logRankToFlowModels(
    const NcclLogLevel level,
    const std::string& algorithmName,
    const std::map<int, std::shared_ptr<FlowModels>>& rank2pflowmodels) {

    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    // Header for the log output
    NcclLog->writeLog(level, "=======================================================================");
    NcclLog->writeLog(level, "--- Flow Model Details for: %s ---", algorithmName.c_str());
    NcclLog->writeLog(level, "=======================================================================");

    // Iterate over each rank that has associated flows.
    for (const auto& rank_pair : rank2pflowmodels) {
        int rank = rank_pair.first;
        const auto& models_ptr = rank_pair.second;

        if (!models_ptr) continue;

        std::stringstream header_ss;
        header_ss << "--- Flows visible to Rank " << rank << " ---";
        NcclLog->writeLog(level, "%s", header_ss.str().c_str());

        // Copy flows to a vector to sort them by flow_id for readability.
        std::vector<SingleFlow> sorted_flows;
        for (const auto& flow_pair : *models_ptr) {
            sorted_flows.push_back(flow_pair.second);
        }
        std::sort(sorted_flows.begin(), sorted_flows.end(),
                  [](const SingleFlow& a, const SingleFlow& b) {
                      return a.flow_id < b.flow_id;
                  });

        // Log the details of each sorted flow.
        for (const auto& flow : sorted_flows) {
            std::stringstream flow_ss;
            flow_ss << "  Flow ID: " << flow.flow_id
                    << "\t| Ch: " << flow.channel_id
                    << "\t| Chunk ID: " << flow.chunk_id
                    << "\t| " << flow.src << " -> " << flow.dest
                    << "\t| Size: " << flow.flow_size
                    << "\t| Parents: [" << flow.parent_flow_str() << "]"
                    << "\t| Children: [" << flow.child_flow_str() << "]";
            NcclLog->writeLog(level, "%s", flow_ss.str().c_str());
        }
    }
    NcclLog->writeLog(level, "=======================================================================");
  }

// --- Helper Struct for NCCL Constants and Dummy Values ---
// This struct holds realistic default values based on init.cc and primitives.h
// to make the simulation logic clear and tunable.
struct NcclSimulationConstants {
    // values taken from init.cc and primitives.h of MSCCL
    // From init.cc: #define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
    static constexpr uint64_t BUFFSIZE_SIMPLE = 4 * 1024 * 1024;

    // From init.cc: #define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
    // Dummy values for a typical configuration: 8 lines * 512 threads * 8 steps * 16 bytes/line
    static constexpr uint64_t BUFFSIZE_LL = 8 * 512 * 8 * 16; // Approx 512 KiB

    // From init.cc: #define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
    // Dummy values: 28 elems * 256 threads * 8 steps * 8 bytes/elem
    static constexpr uint64_t BUFFSIZE_LL128 = 28 * 256 * 8 * 8; // Approx 448 KiB

    // From common.h (or assumed default)
    static constexpr int NCCL_STEPS = 8;

    static constexpr int MSCCL_CHUNKSTEPS = NCCL_STEPS / 2; // Value is 4

    // From common.h for LL128 protocol
    static constexpr int NCCL_LL128_DATAELEMS = 14;
    static constexpr int NCCL_LL128_LINEELEMS = 16;
};

std::optional<std::map<int,std::shared_ptr<FlowModels>>> MockNcclGroup::genAllGatherCustomFlowModels(GroupType type , int rank,uint64_t data_size, int protocol) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    GroupInfo gp_info;
    int gp_idx;
    MscclAlgorithm msccl_algorithm;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    NcclLog->writeLog(NcclLogLevel::DEBUG,"Generating flow model for Custom AllGather with type: %d, rank: %d, data_size: %lu, protocol: %d", type, rank, data_size, protocol);

    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      gp_info = AllGroups[gp_idx];
    }

    // --- Find and load the correct MSCCL algorithm ---
    std::string folderName = "./msccl-algorithms";
    bool found_algo = false;
    NcclLog->writeLog(NcclLogLevel::DEBUG, "Searching for MSCCL XML algorithms in folder: %s", folderName.c_str());
    try {
        for (const auto& entry : std::filesystem::directory_iterator(folderName)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                MscclAlgorithm tmp = parseMscclXml(entry.path().string());
                 NcclLog->writeLog(NcclLogLevel::DEBUG, "Parsed MSCCL XML algorithm: %s with nchunksperloop: %d, minBytes: %lu, maxBytes: %lu, n_gpus: %d",
                                     entry.path().string().c_str(),
                                     tmp.nchunksperloop, tmp.minBytes, tmp.maxBytes, tmp.n_gpus);
                NcclLog->writeLog(NcclLogLevel::DEBUG, "Parameters to match - coll_type: %s, data_size: %lu, n_gpus: %d",
                                     tmp.coll_type.c_str(), data_size, gp_info.nRanks);
                if (tmp.coll_type == "allgather" && data_size >= tmp.minBytes && (data_size <= tmp.maxBytes || tmp.maxBytes==0) && tmp.nchunksperloop > 0 && tmp.n_gpus == gp_info.nRanks) {
                    msccl_algorithm = tmp;
                    found_algo = true;
                    NcclLog->writeLog(NcclLogLevel::INFO, "Found matching MSCCL XML algorithm: %s", entry.path().string().c_str());
                    std::cout << "Found matching MSCCL XML algorithm: " << entry.path().string() << std::endl;
                    break;
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "Filesystem error: %s. Current path: %s", e.what(), std::filesystem::current_path().string().c_str());
        return std::nullopt;
    }

    if (!found_algo) {
        NcclLog->writeLog(NcclLogLevel::INFO, "No matching MSCCL XML algorithm found for the given data_size. Fallback required.");
        return std::nullopt;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "Using MSCCL algorithm: %s for AllGather with data_size: %lu", msccl_algorithm.filename.c_str(), data_size);

    uint64_t pipeline_chunk_size;
    // might not be perfect, need to round up to the next multiple of the chunk size
    const uint64_t data_size_per_loop = (data_size + msccl_algorithm.nchunksperloop - 1) / msccl_algorithm.nchunksperloop;

    if (protocol == NCCL_PROTO_SIMPLE) {
        // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
        const uint64_t byte_per_step = NcclSimulationConstants::BUFFSIZE_SIMPLE / NcclSimulationConstants::NCCL_STEPS;
        pipeline_chunk_size = byte_per_step * NcclSimulationConstants::MSCCL_CHUNKSTEPS;
        NcclLog->writeLog(NcclLogLevel::INFO, "Simple protocol: Using coarse-grained pipeline chunk size of %lu bytes (Formula: (BuffSize/%d) * %d)",
                          pipeline_chunk_size, NcclSimulationConstants::NCCL_STEPS, NcclSimulationConstants::MSCCL_CHUNKSTEPS);
    } else if (protocol == NCCL_PROTO_LL) {
        // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / 2;
        pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL / NcclSimulationConstants::NCCL_STEPS) / 2;
        NcclLog->writeLog(NcclLogLevel::INFO, "LL protocol: Using fine-grained pipeline chunk size of %lu bytes (Formula: (%lu / %d) / 2)",
                          pipeline_chunk_size, NcclSimulationConstants::BUFFSIZE_LL, NcclSimulationConstants::NCCL_STEPS);
    } else if (protocol == NCCL_PROTO_LL128) { // Assuming LL128 or other fine-grained protocol
        // Formula from primitives.h: (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
        pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL128 / NcclSimulationConstants::NCCL_STEPS)
                              * NcclSimulationConstants::NCCL_LL128_DATAELEMS / NcclSimulationConstants::NCCL_LL128_LINEELEMS;
        NcclLog->writeLog(NcclLogLevel::INFO, "LL128 protocol: Using fine-grained pipeline chunk size of %lu bytes (Formula: (%lu / %d) * %d / %d)",
                          pipeline_chunk_size, NcclSimulationConstants::BUFFSIZE_LL128, NcclSimulationConstants::NCCL_STEPS,
                          NcclSimulationConstants::NCCL_LL128_DATAELEMS, NcclSimulationConstants::NCCL_LL128_LINEELEMS);
    }else{
        // undefined, put infinite in order to use the whole data_size_per_loop as a single chunk
        pipeline_chunk_size = UINT64_MAX;
        NcclLog->writeLog(NcclLogLevel::INFO, "Undefined protocol (%d). Using entire data_size_per_loop (%lu bytes) as a single chunk.",
                          protocol, data_size_per_loop);
    }

    // Unconditionally calculate the number of sub-chunks based on the chosen pipeline granularity.
    int num_sub_chunks = (data_size_per_loop > 0) ? (data_size_per_loop + pipeline_chunk_size - 1) / pipeline_chunk_size : 0;
    if (num_sub_chunks == 0 && data_size_per_loop > 0) num_sub_chunks = 1; // Ensure at least one chunk if there is data.

    NcclLog->writeLog(NcclLogLevel::INFO, "Data per loop (%lu) will be split into %d sub-chunks.", data_size_per_loop, num_sub_chunks);


    // --- Unique Chunk ID Remapping ---
    // Pre-calculate the mapping from (sub-chunk, local-offset) to a global unique ID
    std::vector<ChunkInfo> all_chunks;
    for (int sc = 0; sc < num_sub_chunks; ++sc) {
        for (const auto& [gpu_id, gpu] : msccl_algorithm.gpus) {
            for (const auto& tb : gpu.thread_blocks) {
                for (const auto& step : tb.steps) {
                    if (step.type == "s" || step.type == "rcs") {
                      all_chunks.push_back(ChunkInfo{sc, step.src_offset, step.count});
                    }
                }
            }
        }
    }
    std::sort(all_chunks.begin(), all_chunks.end(), [](const ChunkInfo& a, const ChunkInfo& b) {
        if (a.sub_chunk_idx != b.sub_chunk_idx) {
            return a.sub_chunk_idx < b.sub_chunk_idx;
        }
        return a.src_offset < b.src_offset;
    });

    // 2. MERGE OVERLAPPING CHUNKS
    std::vector<ChunkInfo> merged_chunks;
    if (!all_chunks.empty()) {
        merged_chunks.push_back(all_chunks[0]);

        for (size_t i = 1; i < all_chunks.size(); ++i) {
            ChunkInfo& last_merged = merged_chunks.back();
            const ChunkInfo& current_chunk = all_chunks[i];

            // Check if chunks belong to the same sub-chunk and overlap or are adjacent
            if (current_chunk.sub_chunk_idx == last_merged.sub_chunk_idx &&
                current_chunk.src_offset <= (last_merged.src_offset + last_merged.count)) {

                // They overlap, so merge them by extending the last merged chunk
                int last_merged_end = last_merged.src_offset + last_merged.count;
                int current_chunk_end = current_chunk.src_offset + current_chunk.count;

                // The new end is the maximum of the two ends
                int new_end = std::max(last_merged_end, current_chunk_end);

                // Update the count of the last merged chunk
                last_merged.count = new_end - last_merged.src_offset;

            } else {
                // No overlap, this is a new contiguous block
                merged_chunks.push_back(current_chunk);
            }
        }
    }

    // 3. ASSIGN CONTIGUOUS GLOBAL IDS
    // Now we have a clean, non-overlapping list of chunks.
    int unique_chunk_id_counter = 0;
    std::map<std::pair<int, int>, int> chunk_remap;

    for (const auto& chunk : merged_chunks) {
        // Process each block within the final, merged chunk
        for (int i = 0; i < chunk.count; ++i) {
            int local_offset = chunk.src_offset + i;
            auto key = std::make_pair(chunk.sub_chunk_idx, local_offset);

            // No need to check for existence here, as we know there are no overlaps.
            chunk_remap[key] = unique_chunk_id_counter++;

            // Optional: Simplified logging for the final mapping
            // NcclLog->writeLog(NcclLogLevel::DEBUG, "Mapping sub-chunk %d, Offset %d -> Global Chunk ID: %d",
            //                    chunk.sub_chunk_idx, local_offset, chunk_remap[key]);
        }
    }

    int total_unique_chunks = unique_chunk_id_counter;
    NcclLog->writeLog(NcclLogLevel::INFO, "Total unique chunk transfers identified: %d", total_unique_chunks);
    // print all the mappings
    for (const auto& [key, value] : chunk_remap) {
        NcclLog->writeLog(NcclLogLevel::INFO, "Sub-chunk %d, Offset %d -> Global Chunk ID: %d", key.first, key.second, value);
    }

    // Removed: prev_sc_step_to_flow_key (the old S->S pipeline tracker)
    // Added: tb_last_flow_map (tracks the tail of the chain for each TB to enforce sequential chunks)
    // Key: <gpu_id, tb_id>, Value: <channel_id, flow_id>
    std::map<std::pair<int, int>, std::pair<int, uint64_t>> tb_last_flow_map;

    // Outer loop for pipelining
    for (int sc_idx = 0; sc_idx < num_sub_chunks; ++sc_idx) {
        // Calculate the size of the data for the current sub-chunk
        uint64_t current_sc_size = (sc_idx == num_sub_chunks - 1)
                                   ? (data_size_per_loop - sc_idx * pipeline_chunk_size)
                                   : pipeline_chunk_size;

        // This is the size of data corresponding to a `count` of 1 in the XML step.
        uint64_t size_per_xml_count_unit = current_sc_size;
        if(size_per_xml_count_unit == 0) {
            NcclLog->writeLog(NcclLogLevel::WARNING, "Calculated MSCCL chunk size is 0 for sub-chunk %d. Skipping.", sc_idx);
            continue;
        }

        NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d/%d] Processing Size: %lu, Size per XML count unit: %lu", sc_idx + 1, num_sub_chunks, current_sc_size, size_per_xml_count_unit);

        // --- Intra-sub-chunk dependency trackers (reset for each sub-chunk) ---
        // Key: <gpu_id, tb_id, step_id, chunk_index_in_step>, Value: <channel_id, flow_id>
        std::map<std::tuple<int, int, int, int>, std::pair<int, uint64_t>> current_sc_step_to_flow_key;
        std::map<std::tuple<int,int>, std::pair<int, uint64_t>> current_sc_chkDstToFlowId;

        // **Pass 1 (per sub-chunk): Create all flow objects from 'send' operations**
        NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] --- Pass 1: Creating flows from 'send' operations ---", sc_idx);
        for (const auto& [gpu_id, gpu] : msccl_algorithm.gpus) {
            for (const auto& tb : gpu.thread_blocks) {
                for (const auto& step : tb.steps) {
                    if (step.type == "s" || step.type == "rcs") { // operations involving the send
                        if (tb.send_peer == -1) continue;

                        // --- START MODIFICATION ---
                        // Unroll the 'count' loop to create individual flows
                        for (int i = 0; i < step.count; ++i) {
                            int current_chunk_local_offset = step.src_offset + i;
                            int global_chunk_id = chunk_remap.at({sc_idx, current_chunk_local_offset});
                            
                            // Flow size is for a single chunk in this pipeline stage
                            uint64_t flow_size = size_per_xml_count_unit; 

                            SingleFlow tmp_result(g_flow_id, gpu.id, tb.send_peer, flow_size, {}, {}, {}, tb.channel, global_chunk_id, total_unique_chunks, "RING");

                            NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] Creating Flow ID: %lu (Src: %d, Dst: %d, Size: %lu, Channel: %d, ChunkID: %d) for step.count index %d", sc_idx, g_flow_id, gpu.id, tb.send_peer, flow_size, tb.channel, global_chunk_id, i);

                            auto result_key = std::make_pair(tb.channel, g_flow_id);
                            result[result_key] = tmp_result;

                            // Key now includes the chunk index 'i'
                            auto step_key = std::make_tuple(gpu_id, tb.id, step.id, i);
                            current_sc_step_to_flow_key[step_key] = result_key;
                            
                            auto dependency_key = std::make_tuple(current_chunk_local_offset, tb.send_peer);
                            current_sc_chkDstToFlowId[dependency_key] = result_key;

                            g_flow_id++;
                        }
                        // --- END MODIFICATION ---
                    }
                }
            }
        }

        // **Pass 2 (per sub-chunk): Link dependencies (parent/child relationships)**
        NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] --- Pass 2: Linking flow dependencies (Parent/Child) ---", sc_idx);
        for (const auto& [gpu_id, gpu] : msccl_algorithm.gpus) {
            for (const auto& tb : gpu.thread_blocks) {          
              // Load the last flow from the PREVIOUS chunk for this specific Thread Block
                std::optional<std::pair<int, uint64_t>> last_send_in_block_key;
                if (tb_last_flow_map.count({gpu_id, tb.id})) {
                    last_send_in_block_key = tb_last_flow_map[{gpu_id, tb.id}];
                    NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] GPU %d TB %d starts with dependency on previous chunk flow %lu", sc_idx, gpu_id, tb.id, last_send_in_block_key->second);
                }
                for (const auto& step : tb.steps) {
                    if (step.type == "s" || step.type == "rcs") {                        
                        // Unroll the 'count' loop to link individual flows
                        for (int i = 0; i < step.count; ++i) {
                            auto current_step_key = std::make_tuple(gpu_id, tb.id, step.id, i);
                            if(current_sc_step_to_flow_key.find(current_step_key) == current_sc_step_to_flow_key.end()) continue;
                            
                            auto& current_flow_key = current_sc_step_to_flow_key.at(current_step_key);
                            int current_chunk_local_offset = step.src_offset + i;

                            std::vector<int> parent_ids;
                            
                            // Dependency 1: Explicit Data Dependency (inter-block 'rcs' or 'depid')
                            auto parent_lookup_key = std::make_tuple(current_chunk_local_offset, gpu.id);
                            
                            if (step.type == "rcs") {
                                if (current_sc_chkDstToFlowId.count(parent_lookup_key)) {
                                    int p_id = current_sc_chkDstToFlowId.at(parent_lookup_key).second;
                                    parent_ids.push_back(p_id);
                                    NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] Linking rcs implicit data dependency for flow %lu -> parent %d", sc_idx, current_flow_key.second, p_id);
                                } else {
                                    NcclLog->writeLog(NcclLogLevel::WARNING, "[Sub-chunk %d] Could not find implicit parent for rcs flow %lu (lookup key: chunk offset %d, destination gpu %d)", sc_idx, current_flow_key.second, current_chunk_local_offset, gpu.id);
                                }
                            }

                            if (step.dep_id != -1) {
                                if (current_sc_chkDstToFlowId.count(parent_lookup_key)) {
                                    int dep_flow_id = current_sc_chkDstToFlowId.at(parent_lookup_key).second;
                                    if (std::find(parent_ids.begin(), parent_ids.end(), dep_flow_id) == parent_ids.end()) {
                                        parent_ids.push_back(dep_flow_id);
                                        NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] Linking explicit depid dependency for flow %lu -> parent %d", sc_idx, current_flow_key.second, dep_flow_id);
                                    }
                                }else{
                                    if(gpu.thread_blocks[step.dep_id].steps[step.dep_step].type != "cpy"){
                                      NcclLog->writeLog(NcclLogLevel::ERROR, "[Sub-chunk %d] Could not find explicit depid parent for flow %lu which is not a CPY (lookup key: chunk offset %d, destination gpu %d)", sc_idx, current_flow_key.second, current_chunk_local_offset, gpu.id);
                                      return std::nullopt; // If we can't find the parent, we can't proceed.
                                    }
                                }
                            }

                            // Dependency 2: Sequential Execution (intra-block AND intra-step)
                            // This now also handles Inter-Sub-chunk sequencing because last_send_in_block_key 
                            // is initialized from the previous chunk's last step.
                            if (last_send_in_block_key.has_value()) {
                                int p_id = last_send_in_block_key.value().second;
                                parent_ids.push_back(p_id);
                                NcclLog->writeLog(NcclLogLevel::INFO, "[Sub-chunk %d] Linking sequential dependency for flow %lu -> parent %d", sc_idx, current_flow_key.second, p_id);
                            }
                            // Add parents and update child links
                            if (!parent_ids.empty()) {
                                auto& current_flow = result.at(current_flow_key);
                                for (int p_id : parent_ids) {
                                    int nchannels = msccl_algorithm.n_channels;
                                    auto parent_it = result.find(std::make_pair(current_flow.channel_id, p_id));
                                    if(parent_it == result.end()){
                                      // If we didn't find the parent in the same channel, search across all channels
                                      for (int channel_idx = 0; channel_idx < nchannels; ++channel_idx) {
                                          parent_it = result.find(std::make_pair(channel_idx, p_id));
                                          // If we found the parent, stop searching
                                          if (parent_it != result.end()) {
                                              break;
                                          }
                                      }
                                    }
                                    if (parent_it != result.end()) {
                                        current_flow.parent_flow_id.push_back(p_id);
                                        parent_it->second.child_flow_id.push_back(current_flow.flow_id);
                                        current_flow.prev.push_back(parent_it->second.src);
                                    } else {
                                        NcclLog->writeLog(NcclLogLevel::WARNING, "Could not find parent flow with ID %d in the result map to establish dependency.", p_id);
                                    }
                                }
                            }
                            
                            // The current flow becomes the parent for the next flow in the block
                            last_send_in_block_key = current_flow_key;
                        }
                    }
                }
                // Store the last flow of this chunk to chain it to the next chunk
                if (last_send_in_block_key.has_value()) {
                    tb_last_flow_map[{gpu_id, tb.id}] = last_send_in_block_key.value();
                }
            }
        }
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "--- Pass 3: Validating flow dependency graph ---");
    if (!validate_flow_graph(result, msccl_algorithm, data_size, total_unique_chunks, NcclLog, protocol)) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "Flow graph validation failed. Aborting flow model generation.");
        return std::nullopt;
    }

    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    }

    // Calculate nlocalRanks as pointed out by the user
    int nlocalRanks = (gp_info.nNodes > 0) ? (gp_info.nRanks / gp_info.nNodes) : 0;

    if (PXN_ENABLE && gp_info.nNodes > 1 && nlocalRanks > 0) {
        NcclLog->writeLog(NcclLogLevel::INFO, "--- Pass 4: Applying PXN Transformation (nlocalRanks = %d) ---", nlocalRanks);

        // We must iterate over a snapshot of keys, as we are modifying the 'result' map
        std::vector<std::pair<int, uint64_t>> flow_keys;
        for (const auto& [key, flow] : result) {
            flow_keys.push_back(key);
        }
        
        // --- Pass 4a: Create proxy flows and modify original flows ---
        // These maps track the transformation.
        std::map<uint64_t, std::pair<int, uint64_t>> original_id_to_original_key;
        std::map<uint64_t, std::pair<int, uint64_t>> original_id_to_proxy_key;
        std::map<uint64_t, std::vector<int>> original_id_to_original_parents;
        std::map<uint64_t, std::vector<int>> original_id_to_original_children;
        FlowModels new_proxy_flows;
        std::set<uint64_t> transformed_flow_ids; // Set of original IDs that were transformed

        NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN Pass 4a: Identifying flows to transform and creating proxy flows...");
        for (const auto& key : flow_keys) {
            SingleFlow& flow = result.at(key);
            uint64_t original_id = flow.flow_id;

            // Store original state before modification
            original_id_to_original_key[original_id] = key;
            original_id_to_original_parents[original_id] = flow.parent_flow_id;
            original_id_to_original_children[original_id] = std::move(flow.child_flow_id);
            flow.child_flow_id.clear(); // Clear children, will be repopulated in Pass 4c

            int x = flow.src;
            int y = flow.dest;
            int node_x = x / nlocalRanks;
            int node_y = y / nlocalRanks;

            // Check if it's an inter-node transfer that needs transformation
            if (node_x != node_y) {
                int local_rank_y = y % nlocalRanks;
                int base_rank_x_node = node_x * nlocalRanks;
                int z_proxy_rank = base_rank_x_node + local_rank_y;

                if (z_proxy_rank != x) {
                    NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN: Flow %lu (%d -> %d) will be split via proxy %d", original_id, x, y, z_proxy_rank);
                    transformed_flow_ids.insert(original_id);

                    // 1. Create the new proxy flow (z -> y)
                    uint64_t proxy_flow_id = g_flow_id++;
                    SingleFlow proxy_flow = flow; // Copy properties
                    proxy_flow.flow_id = proxy_flow_id;
                    proxy_flow.src = z_proxy_rank;
                    proxy_flow.dest = y;
                    proxy_flow.prev = {x}; // Depends on the original sender
                    proxy_flow.parent_flow_id = {(int)original_id}; // Base dependency on original flow
                    proxy_flow.child_flow_id.clear(); // Children will be added in Pass 4c
                    proxy_flow.conn_type = "PXN";

                    // 2. Store the new proxy flow
                    auto proxy_key = std::make_pair(proxy_flow.channel_id, proxy_flow_id);
                    new_proxy_flows[proxy_key] = proxy_flow;
                    original_id_to_proxy_key[original_id] = proxy_key;

                    // 3. Modify the original flow to become (x -> z)
                    flow.dest = z_proxy_rank;
                    flow.child_flow_id.push_back(proxy_flow_id); // Add proxy as its child
                }
            }
        }

        // --- Pass 4b: Build master flow map ---
        // This map allows us to find any flow (original or new proxy) by its ID.
        NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN Pass 4b: Building master flow map...");
        std::map<uint64_t, SingleFlow*> all_flows_map;
        for (auto& [key, flow] : result) {
            all_flows_map[flow.flow_id] = &flow;
        }
        for (auto& [key, flow] : new_proxy_flows) {
            all_flows_map[flow.flow_id] = &flow;
        }

        // --- Pass 4c: Link proxy-to-proxy and partition children ---
        NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN Pass 4c: Linking proxy-to-proxy dependencies and partitioning children...");
        for (const auto& [original_id, original_key] : original_id_to_original_key) {
            
            // Find the original flow (which might now be x -> z)
            SingleFlow& original_flow = *all_flows_map.at(original_id);
            const bool was_transformed = transformed_flow_ids.count(original_id);
            const std::vector<int>& original_children = original_id_to_original_children.at(original_id);

            SingleFlow* proxy_flow_ptr = nullptr;
            if (was_transformed) {
                proxy_flow_ptr = all_flows_map.at(original_id_to_proxy_key.at(original_id).second);
                
                // --- [YOUR NEW LOGIC] ---
                NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN: Checking proxy-to-proxy dependencies for flow %lu (proxy %lu)", original_id, proxy_flow_ptr->flow_id);
                const std::vector<int>& original_parents = original_id_to_original_parents.at(original_id);
                for (int parent_id : original_parents) {
                    // Check if the parent was *also* transformed
                    if (transformed_flow_ids.count(parent_id)) {
                        uint64_t parent_proxy_id = original_id_to_proxy_key.at(parent_id).second;
                        
                        // Add parent's proxy as a parent to this flow's proxy
                        proxy_flow_ptr->parent_flow_id.push_back(parent_proxy_id);
                        
                        // Add this flow's proxy as a child to the parent's proxy
                        all_flows_map.at(parent_proxy_id)->child_flow_id.push_back(proxy_flow_ptr->flow_id);
                        
                        NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN: Linked proxy flow %lu (child) to parent proxy flow %lu (parent)", proxy_flow_ptr->flow_id, parent_proxy_id);
                    }
                }
                // --- [END OF NEW LOGIC] ---
            }

            // --- Child Partitioning and Re-linking ---
            for (uint64_t child_id : original_children) {
                if (all_flows_map.find(child_id) == all_flows_map.end()) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "PXN: Could not find child flow %lu in master map. Aborting.", child_id);
                    return std::nullopt;
                }
                SingleFlow& child_flow = *all_flows_map.at(child_id);
                
                uint64_t new_parent_id = 0;
                SingleFlow* new_parent_flow_ptr = nullptr;

                // Decide which flow is the new parent: the original (x->z) or the proxy (z->y)
                if (was_transformed && child_flow.src != original_flow.src) {
                    // Child starts elsewhere (e.g., at 'y'), so it should depend on the proxy flow (z -> y)
                    new_parent_flow_ptr = proxy_flow_ptr;
                    new_parent_id = proxy_flow_ptr->flow_id;
                } else {
                    // Child starts at 'x' OR the original flow wasn't transformed.
                    // It should depend on the original flow (x -> z or x -> y)
                    new_parent_flow_ptr = &original_flow;
                    new_parent_id = original_flow.flow_id;
                }

                // Add child to new parent's child list
                new_parent_flow_ptr->child_flow_id.push_back(child_id);

                // Update the child's parent list
                // Replace the *original_id* with the *new_parent_id*
                std::replace(child_flow.parent_flow_id.begin(), child_flow.parent_flow_id.end(), (int)original_id, (int)new_parent_id);
            }
        }

        // --- Pass 4d: Add all new flows to the main result map ---
        NcclLog->writeLog(NcclLogLevel::DEBUG, "PXN Pass 4d: Merging %zu new proxy flows into result map...", new_proxy_flows.size());
        result.insert(new_proxy_flows.begin(), new_proxy_flows.end());

        NcclLog->writeLog(NcclLogLevel::INFO, "--- Pass 5: Validating flow dependency graph after PXN (Structure & Connectivity) ---");
        if (!validate_flow_graph_PXN(result, msccl_algorithm, data_size, total_unique_chunks, NcclLog, protocol)) {
            NcclLog->writeLog(NcclLogLevel::ERROR, "PXN flow graph structural validation failed. Aborting flow model generation.");
            return std::nullopt;
        }
    }
    // --- Finalize and structure the data for return ---
    NcclLog->writeLog(NcclLogLevel::INFO, "--- Finalizing and structuring data for return ---");
    for(const auto& [key, flow] : result) {
        NcclLog->writeLog(NcclLogLevel::INFO, "Assigning flow %lu (Src: %d, Dst: %d) to rank2flowmodels for both source and destination ranks.", flow.flow_id, flow.src, flow.dest);
        rank2flowmodels[flow.src][key] = flow;
        if (flow.src != flow.dest) {
            rank2flowmodels[flow.dest][key] = flow;
        }
    }

    for(const auto& [curr_rank, models] : rank2flowmodels) {
        NcclLog->writeLog(NcclLogLevel::INFO, "Creating shared FlowModels object for rank %d with %zu models.", curr_rank, models.size());
        rank2pflowmodels[curr_rank] = std::make_shared<FlowModels>(models);
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "--- Flow generation complete for rank %d ---", rank);
    logRankToFlowModels(NcclLogLevel::INFO, "Custom MSCCL AllGather", rank2pflowmodels);
    return rank2pflowmodels;
}
bool MockNcclGroup::validate_flow_graph_PXN(const FlowModels& flows,
                                        const MscclAlgorithm& msccl_algorithm,
                                        uint64_t data_size,
                                        int total_unique_chunks,
                                        MockNcclLog* NcclLog,
                                        int protocol) {
    if (flows.empty()) {
        if (data_size == 0) {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "[VALIDATION_PXN] Flow graph is empty and data_size is 0, no validation needed.");
            return true;
        } else {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Flow graph is empty, but data_size > 0. This is an error.");
            return false;
        }
    }

    std::map<uint64_t, const SingleFlow*> flow_map;
    std::set<uint64_t> all_flow_ids;

    // --- Check 1: Duplicate Flow IDs & Map Key/Value Consistency ---
    for (const auto& pair : flows) {
        const auto& flow = pair.second;
        uint64_t flow_id = flow.flow_id;

        // Check if the flow_id in the map's key matches the flow_id in the SingleFlow struct
        if (pair.first.second != flow_id) {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Mismatch: Map key flow_id (%lu) != struct flow_id (%lu) for channel %d.", pair.first.second, flow_id, pair.first.first);
            return false;
        }

        // Check if this flow_id has already been seen (must be globally unique)
        if (flow_map.count(flow_id)) {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Duplicate Flow ID %lu found. Original in channel %d, duplicate in channel %d.",
                             flow_id, flow_map[flow_id]->channel_id, flow.channel_id);
            return false;
        }
        flow_map[flow_id] = &flow;
        all_flow_ids.insert(flow_id);
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "[VALIDATION_PXN] Check 1: No duplicate flow IDs found.");


    // --- Check 2: Dependency Existence & Adjacency List Build ---
    // Verifies that all parent/child IDs point to flows that actually exist.
    std::map<uint64_t, std::vector<uint64_t>> adj; // Adjacency list for connectivity check
    for (const auto& pair : flow_map) {
        const auto& flow = *pair.second;
        adj[flow.flow_id] = {}; // Initialize adjacency entry

        // Check parents
        for (uint64_t parent_id : flow.parent_flow_id) {
            if (all_flow_ids.count(parent_id) == 0) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Flow %lu has a dependency on non-existent parent flow %lu.", flow.flow_id, parent_id);
                return false;
            }
        }

        // Check children and build adjacency list
        for (uint64_t child_id : flow.child_flow_id) {
            if (all_flow_ids.count(child_id) == 0) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Flow %lu lists a child %lu which does not exist in the graph.", flow.flow_id, child_id);
                return false;
            }
            adj[flow.flow_id].push_back(child_id);
        }
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "[VALIDATION_PXN] Check 2: All parent/child dependencies point to existing flows.");


    // --- Check 3: Executability (Connectivity) ---
    // Finds all "root" nodes (no parents) and ensures all other nodes are reachable from them.
    // Also handles the case of a graph with no roots (e.g., a single cycle) by checking for disconnected components.
    std::set<uint64_t> root_nodes;
    for (const auto& id : all_flow_ids) {
        if (flow_map.at(id)->parent_flow_id.empty()) {
            root_nodes.insert(id);
        }
    }

    if (root_nodes.empty() && !all_flow_ids.empty()) {
        NcclLog->writeLog(NcclLogLevel::WARNING, "[VALIDATION_PXN] Schedule %s - No root nodes (flows with no parents) found. This may be valid with PXN loops. Checking for disconnected components.", msccl_algorithm.filename.c_str());
    }

    std::set<uint64_t> reachable_nodes;
    std::function<void(uint64_t)> find_reachable_util =
        [&](uint64_t u) {
        reachable_nodes.insert(u);
        if (adj.count(u)) {
            for (uint64_t v : adj.at(u)) {
                // all_flow_ids.count(v) is redundant due to Check 2, but is safe.
                if (all_flow_ids.count(v) && reachable_nodes.find(v) == reachable_nodes.end()) {
                    find_reachable_util(v);
                }
            }
        }
    };

    // Find all nodes reachable from the identified root nodes
    for (const auto& root_id : root_nodes) {
        if (reachable_nodes.find(root_id) == reachable_nodes.end()) {
            find_reachable_util(root_id);
        }
    }

    // Check if all nodes were reached
    if (reachable_nodes.size() < all_flow_ids.size()) {
        // If we had no roots, we need to check if the graph is just disconnected
        if (root_nodes.empty()) {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "[VALIDATION_PXN] No root nodes found. Starting graph traversal from arbitrary node to find components.");
            int component_count = 0;
            // We already traversed one component if any node was reachable
            if (!reachable_nodes.empty()) component_count = 1;
            
            for (const auto& id : all_flow_ids) {
                if (reachable_nodes.find(id) == reachable_nodes.end()) {
                    component_count++;
                    NcclLog->writeLog(NcclLogLevel::DEBUG, "[VALIDATION_PXN] Found new component starting with node %lu.", id);
                    find_reachable_util(id);
                }
            }
            if (component_count > 1) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Schedule %s - Graph is disconnected. Found %d components.", msccl_algorithm.filename.c_str(), component_count);
                return false;
            }
        } else {
            // We had roots, but some nodes were not reachable from them.
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION_PXN] Schedule %s - Some nodes are unreachable from root nodes (graph is disconnected).", msccl_algorithm.filename.c_str());
            for (const auto& id : all_flow_ids) {
                if (reachable_nodes.find(id) == reachable_nodes.end()) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, " -> Unreachable Flow ID: %lu (Parents: %s)", id, flow_map.at(id)->parent_flow_str().c_str());
                }
            }
            return false;
        }
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "[VALIDATION_PXN] Check 3: Graph is executable (all nodes are part of a single connected component).");


    // --- Data Integrity Validation Skipped ---
    NcclLog->writeLog(NcclLogLevel::INFO, "[VALIDATION_PXN] Skipping data integrity/chunk validation as requested for PXN.");

    NcclLog->writeLog(NcclLogLevel::INFO, "[VALIDATION_PXN] Flow graph structural validation successful.");
    return true;
}

bool MockNcclGroup::validate_flow_graph(const FlowModels& flows,
                                        const MscclAlgorithm& msccl_algorithm,
                                        uint64_t data_size,
                                        int total_unique_chunks,
                                        MockNcclLog* NcclLog,
                                        int protocol) {
    if (flows.empty()) {
        if (data_size == 0) {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "Flow graph is empty and data_size is 0, no validation needed.");
            return true;
        } else {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Flow graph is empty, but data_size > 0. This is an error.");
            return false;
        }
    }

    std::map<uint64_t, std::vector<uint64_t>> adj;
    std::map<uint64_t, const SingleFlow*> flow_map;
    std::set<uint64_t> all_flow_ids;

    for (const auto& pair : flows) {
        const auto& flow = pair.second;
        adj[flow.flow_id] = std::vector<uint64_t>(flow.child_flow_id.begin(), flow.child_flow_id.end());
        flow_map[flow.flow_id] = &flow;
        all_flow_ids.insert(flow.flow_id);
    }

    // --- Cycle Detection using DFS ---
    enum class NodeState { Unvisited, Visiting, Visited };
    std::map<uint64_t, NodeState> visited_state;
    for (const auto& id : all_flow_ids) {
        visited_state[id] = NodeState::Unvisited;
    }

    std::function<bool(uint64_t)> detect_cycle_util =
        [&](uint64_t u) -> bool {
        visited_state[u] = NodeState::Visiting;
        if (adj.count(u)) {
            for (uint64_t v : adj.at(u)) {
                if (visited_state.count(v) == 0) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a child dependency on non-existent flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                    return true;
                }
                if (visited_state.at(v) == NodeState::Visiting) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a circular dependency with flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                    return true;
                }
                if (visited_state.at(v) == NodeState::Unvisited) {
                    if (detect_cycle_util(v)) {
                      NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has a circular dependency with flow %lu.", msccl_algorithm.filename.c_str(), u, v);
                      return true;
                    }
                }
            }
        }
        visited_state[u] = NodeState::Visited;
        return false;
    };

    for (const auto& id : all_flow_ids) {
        if (visited_state.at(id) == NodeState::Unvisited) {
            if (detect_cycle_util(id)) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow graph contains circular dependencies.", msccl_algorithm.filename.c_str());
                return false;
            }
        }
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "No circular dependencies found in the flow graph.");

    // --- Unreachable Node Detection ---
    std::set<uint64_t> root_nodes;
    for (const auto& id : all_flow_ids) {
        if (flow_map.at(id)->parent_flow_id.empty()) {
            root_nodes.insert(id);
        }
    }

    if (root_nodes.empty() && !all_flow_ids.empty()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - No root nodes found, indicating a fully circular or disconnected graph.", msccl_algorithm.filename.c_str());
        return false;
    }

    std::set<uint64_t> reachable_nodes;
    std::function<void(uint64_t)> find_reachable_util =
        [&](uint64_t u) {
        reachable_nodes.insert(u);
        if (adj.count(u)) {
            for (uint64_t v : adj.at(u)) {
                if (reachable_nodes.find(v) == reachable_nodes.end()) {
                    find_reachable_util(v);
                }
            }
        }
    };

    for (const auto& root_id : root_nodes) {
        if (reachable_nodes.find(root_id) == reachable_nodes.end()) {
            find_reachable_util(root_id);
        }
    }

    if (reachable_nodes.size() < all_flow_ids.size()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Some nodes are unreachable from root nodes.", msccl_algorithm.filename.c_str());
        for (const auto& id : all_flow_ids) {
            if (reachable_nodes.find(id) == reachable_nodes.end()) {
                NcclLog->writeLog(NcclLogLevel::ERROR, " -> Unreachable Flow ID: %lu", id);
            }
        }
        return false;
    }
    NcclLog->writeLog(NcclLogLevel::INFO, "All nodes are reachable from root nodes.");

    // --- Collective-Specific Data Integrity Validation ---
    if (msccl_algorithm.coll_type == "allgather") {
        NcclLog->writeLog(NcclLogLevel::INFO, "--- Performing AllGather Data Integrity Validation ---");
        int n_gpus = msccl_algorithm.n_gpus;
        bool is_valid = true;

        // 1. Correctly determine the data size being validated, mirroring the generator.
        // The generator simulates ONE loop. We must validate against the data size for that single loop.
        const uint64_t data_size_per_chunk  = (data_size + msccl_algorithm.nchunksperloop - 1) / msccl_algorithm.nchunksperloop;
        NcclLog->writeLog(NcclLogLevel::DEBUG, "Validating against data size for a single loop: %lu bytes", data_size_per_chunk );

        // 2. Setup validation checks using the CORRECT data size
        const uint64_t expected_total_received_size = ((n_gpus > 1) ? (uint64_t)(n_gpus - 1) * data_size_per_chunk  : 0) * (msccl_algorithm.nchunksperloop/msccl_algorithm.n_gpus);

        if (total_unique_chunks % n_gpus != 0) {
            NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Total unique chunks %d is not divisible by number of GPUs %d.", msccl_algorithm.filename.c_str(), total_unique_chunks, n_gpus);
            return false;
        }

        size_t chunks_per_rank = total_unique_chunks / n_gpus;
        size_t data_size_per_rank = data_size_per_chunk  * chunks_per_rank;
        NcclLog->writeLog(NcclLogLevel::INFO, "Each GPU is expected to receive %zu unique chunks, totaling %lu bytes.", chunks_per_rank, data_size_per_rank);
        NcclLog->writeLog(NcclLogLevel::INFO, "Each Chunk is expected to be %lu bytes.", data_size_per_chunk);

        // Track what each GPU receives
        std::map<int, std::set<int>> gpu_received_chunks;   // Key: dest_gpu, Val: set of unique chunk IDs
        std::map<int, uint64_t> gpu_received_size;        // Key: dest_gpu, Val: total bytes received
        // find the data size per transfer based on the protocol
        size_t pipeline_chunk_size=-1;
        if (protocol == NCCL_PROTO_SIMPLE) {
            // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
            const uint64_t byte_per_step = NcclSimulationConstants::BUFFSIZE_SIMPLE / NcclSimulationConstants::NCCL_STEPS;
            pipeline_chunk_size = byte_per_step * NcclSimulationConstants::MSCCL_CHUNKSTEPS;
        } else if (protocol == NCCL_PROTO_LL) {
            // Formula from primitives.h: ncclShmem.comm.buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / 2;
            pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL / NcclSimulationConstants::NCCL_STEPS) / 2;
        } else { // Assuming LL128 or other fine-grained protocol
            // Formula from primitives.h: (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
            pipeline_chunk_size = (NcclSimulationConstants::BUFFSIZE_LL128 / NcclSimulationConstants::NCCL_STEPS)
                                  * NcclSimulationConstants::NCCL_LL128_DATAELEMS / NcclSimulationConstants::NCCL_LL128_LINEELEMS;
        }
        pipeline_chunk_size = std::min(pipeline_chunk_size, data_size_per_chunk);
        auto pipeline_leftover_size = data_size_per_chunk % pipeline_chunk_size;
        // Populate trackers by iterating through all generated flows
        NcclLog->writeLog(NcclLogLevel::INFO, "Each transfer will be split into chunks of size %lu bytes.", pipeline_chunk_size);
        for (const auto& [key, flow] : flows) {
            if (flow.src != flow.dest) { // In AllGather, we only care about inter-gpu traffic
                //for each chunk using cnt
                size_t chunk_count = -1;
                if( flow.flow_size % pipeline_chunk_size == 0) {
                  chunk_count = (flow.flow_size + pipeline_chunk_size - 1)  / pipeline_chunk_size;
                }else{
                  chunk_count = (flow.flow_size + pipeline_leftover_size - 1) / pipeline_leftover_size;
                }
                NcclLog->writeLog(NcclLogLevel::INFO, "Flow ID: %lu, Src: %d, Dest: %d, Size: %lu, Chunk ID: %d, Chunk Count: %d", flow.flow_id, flow.src, flow.dest, flow.flow_size, flow.chunk_id, chunk_count);
                for (int i = 0; i < chunk_count; ++i) {
                    if(gpu_received_chunks[flow.dest].count(flow.chunk_id + i) == 0) {
                        NcclLog->writeLog(NcclLogLevel::DEBUG, "[VALIDATION] Schedule %s - Flow %lu has received chunk ID %d for destination GPU %d.", msccl_algorithm.filename.c_str(), flow.flow_id, flow.chunk_id + i, flow.dest);
                      gpu_received_chunks[flow.dest].insert(flow.chunk_id + i);
                    }else{
                      NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - Flow %lu has received chunk ID %d for destination GPU %d 2 times.", msccl_algorithm.filename.c_str(), flow.flow_id, flow.chunk_id + i, flow.dest);
                      return false;
                    }
                }
                gpu_received_size[flow.dest] += flow.flow_size;
            }
        }

        // Now, validate each GPU
        for (int i = 0; i < n_gpus; ++i) {
            // Check 2: Is the total received data volume correct?
            if (gpu_received_size[i] != expected_total_received_size) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - GPU %d received %lu bytes, but expected %lu bytes.", msccl_algorithm.filename.c_str(), i, gpu_received_size[i], expected_total_received_size);
                return false;
            }

            // Check 3: Is the number of unique chunks received correct?
            // Each of the N-1 peers sends `chunks_per_rank` unique chunks.
            size_t expected_chunks_per_dest = (n_gpus - 1) * chunks_per_rank;
            if (gpu_received_chunks[i].size() != expected_chunks_per_dest) {
                 NcclLog->writeLog(NcclLogLevel::ERROR, "[VALIDATION] Schedule %s - GPU %d received %zu unique chunks, but expected %zu unique chunks.", msccl_algorithm.filename.c_str(), i, gpu_received_chunks[i].size(), expected_chunks_per_dest);
                 return false;
            }
        }
        NcclLog->writeLog(NcclLogLevel::INFO, "AllGather data integrity validation successful: all GPUs receive the correct data from all peers.");
    }

    NcclLog->writeLog(NcclLogLevel::INFO, "Flow graph validation successful.");
    return true;
  }

  std::map<int, std::shared_ptr<FlowModels>> MockNcclGroup::genAllGatherRingFlowModels(GroupType type, int rank, uint64_t data_size) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {}; 
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo gp_info;
    int gp_idx;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.");
      return {};
    } else {
      gp_idx = GroupIndex[std::make_pair(rank,type)];
      ringchannels = Allringchannels[gp_idx];
      gp_info = AllGroups[gp_idx];
    }

    nranks = gp_info.nRanks;
    chunkcount = gp_info.nRanks-1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto& ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            if(rank_it->first!=-1){
              prevranks = {rank_it->first};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if (rank_it->second[0] != -1) {
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
                curnodesendrank != rank_it->second[3]) {
              prenoderecvrank = curnoderecvrank;
              prenodesendrank = curnodesendrank;
              curnoderecvrank = rank_it->second[2];
              curnodesendrank = rank_it->second[3];
            }
            int cur_rank = rank_it->first;
            // for chunks > 0, set previous chunk as parent (e.g., 0 -> 1 chunk 0 is parent for 1 -> 2 chunk 1)
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if (rank_it->first != -1) {
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "PXN");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && gp_info.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    logFlowModels(NcclLogLevel::INFO, "AllGather", rank2pflowmodels);
    return rank2pflowmodels;
  }

  std::map<int, std::shared_ptr<FlowModels>> MockNcclGroup::genAllGatherFlowModels(GroupType type, int rank, uint64_t data_size,bool& msccl, bool NCCL_Simple_LL_splitting) {
      // Assuming AstraSim::ComType::All_Gather is an enum or constant
      ncclInfo* ncc_info = get_algo_proto_info(type, rank, AstraSim::ComType::All_Gather, data_size,msccl, NCCL_Simple_LL_splitting);
      MockNcclLog* NcclLog = MockNcclLog::getInstance();

      switch (ncc_info->algorithm) {
          case NCCL_ALGO_TREE:
          case NCCL_ALGO_RING:
              {
                NcclLog->writeLog(NcclLogLevel::DEBUG, "Using genAllGatherRingFlowModels for type %d, rank %d, data_size %lu", type, rank, data_size);
                return genAllGatherRingFlowModels(type, rank, data_size);
              }

          case NCCL_ALGO_NVLS:
              // It's generally better to throw an exception or handle the error gracefully than calling exit(1)
              // For example: throw std::runtime_error("NCCL_ALGO_NVLS is not supported");
              exit(1);

          case NCCL_ALGO_NVLS_TREE:
              return {};

          case NCCL_ALGO_MSCCL:
          {
              NcclLog->writeLog(NcclLogLevel::INFO, "Using genAllGatherCustomFlowModels for type %d, rank %d, data_size %lu, protocol %d", type, rank, data_size, ncc_info->protocol);
              auto retval = genAllGatherCustomFlowModels(type, rank, data_size, ncc_info->protocol);
              if (!retval) { // A more idiomatic check for std::optional
                  NcclLog->writeLog(NcclLogLevel::INFO, "genAllGatherCustomFlowModels failed, falling back to genAllGatherRingFlowModels for type %d, rank %d, data_size %lu", type, rank, data_size);
                  msccl = false;
                  return genAllGatherRingFlowModels(type, rank, data_size);
              } else {
                  return retval.value();
              }
          }

          default:
              NcclLog->writeLog(NcclLogLevel::ERROR, "Unsupported algorithm %d for AllGather in type %d, rank %d, data_size %lu", ncc_info->algorithm, type, rank, data_size);
              return {};
      }
      return {};
  }
  
  ncclChannelNode* MockNcclGroup::gen_nvls_tree_intra_channels(std::vector<int> intra_topo,std::map<int, vector<ncclChannelNode*>> &nvlstreechannel){
    ncclChannelNode* root = new ncclChannelNode(-1,intra_topo[0],nullptr,{});
    nvlstreechannel[root->rank].push_back(root);
    ncclChannelNode* nvswitch = new ncclChannelNode(-1,intra_topo[1],root,{});
    nvlstreechannel[nvswitch->rank].push_back(nvswitch);
    root->down.push_back(nvswitch);
    for(int i =2;i<intra_topo.size();i++){
      ncclChannelNode*leaf = new ncclChannelNode(-1,intra_topo[i],nvswitch,{});
      nvswitch->down.push_back(leaf);
      nvlstreechannel[leaf->rank].push_back(leaf);
    }
    return root;
  }

  TreeChannels MockNcclGroup::get_nvls_channels(int rank,GroupType type){
    GroupInfo gp_info;
    int gp_idx;
    TreeChannels nvlschannel;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in get_nvls_channels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    if (gp_info.nNodes > 1) {
      NcclLog->writeLog(NcclLogLevel::DEBUG," %d","error NVLS ALGO dont");
      return {};
    } else {
      std::vector<int> ranks = gp_info.Ranks;
      int NVswitch = gp_info.NVSwitchs[0];
      for (int i = 0; i < ranks.size(); i++) {
        nvlschannel[0][ranks[i]] = ncclTree(-1, ranks[i], NVswitch, {});
      }
      nvlschannel[0][ranks.size()] = ncclTree(-1, NVswitch, -1, ranks);
    }
    AllNVLSchannels[gp_idx] = nvlschannel;
    return nvlschannel;
  }

  NVLStreechannels MockNcclGroup::get_nvls_tree_channels(int rank,GroupType type){
    std::map<int,std::map<int,std::vector<ncclChannelNode*>>> nvlstreechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    int gp_idx;
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info , resulting in an error in get_nvls_tree_channels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    if(AllNVLStreechannels.count(gp_idx)){
      return AllNVLStreechannels[gp_idx];
    }
    std::vector<DoubleBinaryTreeNode*>roots;
    roots = genInterDouBinTree(gp_info);

    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);
    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = gp_info.nRanks / nNodes;
      for (int i = 0; i < gp_info.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }

    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    std::map<int, std::vector<int>> node2ranks = allnode2ranks[0];
    for (DoubleBinaryTreeNode* root : roots) {
      for (int index = 0; index < nlocalRanks; index++) {
        std::map<int, vector<ncclChannelNode*>> nvlstreechannel;
        std::map<int,ncclChannelNode*> nodencclchannlenodes;
        for (int i = 0; i < nNodes; i++) {
          std::vector<int> noderanks = node2ranks[i];
          std::vector<int> intra_topo;
          intra_topo.push_back(noderanks[index]);
          intra_topo.push_back(gp_info.NVSwitchs[i]);
          intra_topo.insert(
              intra_topo.end(), noderanks.begin(), noderanks.end());
          NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d intra_topo",i);
          for(auto num:intra_topo){
            NcclLog->writeLog(NcclLogLevel::DEBUG," %d",num);
          }
          ncclChannelNode* root =
              gen_nvls_tree_intra_channels(intra_topo, nvlstreechannel);
          nodencclchannlenodes[i] = root;
        }

        std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
        if (rank == 0) {
          for (nvlstreenodes_it = nvlstreechannel.begin();
               nvlstreenodes_it != nvlstreechannel.end();
               nvlstreenodes_it++) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
            int i = 0;
            for (auto nvlstreenode : nvlstreenodes_it->second) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
              if(nvlstreenode->up!=nullptr)
                NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
              NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
              for (auto down : nvlstreenode->down) {
                NcclLog->writeLog(NcclLogLevel::DEBUG," %d ",down->rank);
              }
            }
          }
        }

        gen_nvls_tree_inter_channels(
            root, nodencclchannlenodes, nvlstreechannel);

        nvlstreechannels[channel_id] = nvlstreechannel;
        channel_id++;
      }
    }
    AllNVLStreechannels[gp_idx] = nvlstreechannels;
    return nvlstreechannels;
  }

  ncclChannelNode* MockNcclGroup::gen_nvls_tree_inter_channels(
      DoubleBinaryTreeNode* root,
      std::map<int, ncclChannelNode*> nodencclchannlenodes,
      std::map<int, vector<ncclChannelNode*>>& nvlstreechannel) {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if (root == nullptr)
      return nullptr;
    else {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->right:  %d",root->right);
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->left:  %d",root->left);
      if (root->left != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->left:  %d",root->left);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* left = nodencclchannlenodes[root->left->node];
        cur->down.push_back(left);
        left->up = cur;
        gen_nvls_tree_inter_channels(root->left,nodencclchannlenodes,nvlstreechannel);
      }
      if (root->right != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->right:  %d",root->right);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* right = nodencclchannlenodes[root->right->node];
        cur->down.push_back(right);
        right->up = cur;
        gen_nvls_tree_inter_channels(root->right,nodencclchannlenodes,nvlstreechannel);
      }
    }
    return nodencclchannlenodes[root->node];
  }

  TreeChannels MockNcclGroup::gettreechannels(int rank, GroupType type){
    TreeChannels treechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo gp_info;
    int gp_idx;
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info and group ring channel, resulting in an error in gettreechannels.");
      return {};
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    if(Alltreechannels.count(gp_idx)){
      return Alltreechannels[gp_idx];
    }
  
    nNodes = gp_info.nNodes;
    nlocalRanks = gp_info.nRanks/nNodes;
    localrings = gen_local_ring(rank,type);
    delta = nNodes > 1 ? gp_info.Ranks[nlocalRanks]-gp_info.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta; 
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::vector<DoubleBinaryTreeNode*> roots;
    roots = genInterDouBinTree(gp_info);
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = gp_info.nRanks / nNodes;
      for (int i = 0; i < gp_info.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    for (allnode2ranks_it = allnode2ranks.begin();
         allnode2ranks_it != allnode2ranks.end();
         allnode2ranks_it++) {
      std::map<int, std::vector<int>> node2ranks = allnode2ranks_it->second;
      for (DoubleBinaryTreeNode* root : roots) {
        std::map<int, ncclTree> treechannel;
        for (int rank : gp_info.Ranks) {
          ncclTree cur =  ncclTree(-1, rank, -1, {});
          treechannel[rank] = cur;
        }
        ConnInterIntraTree(root, node2ranks, treechannel);
        treechannels[channel_id] = treechannel;
        channel_id++;
      }
      Alltreechannels[gp_idx] = treechannels;
    }
    return treechannels;
  }

  void MockNcclGroup::ConnInterIntraTree(DoubleBinaryTreeNode*root,std::map<int,std::vector<int>>node2ranks,std::map<int,ncclTree>&treechannel) {
    if(root == nullptr) return;
    std::vector<int>ranks = node2ranks[root->node];
    for(int i=0;i<ranks.size()-1;i++) {
      ncclTree *current = &treechannel[ranks[i]];
      ncclTree *down = &treechannel[ranks[i+1]];
      current->down.push_back(ranks[i+1]);
      down->up=ranks[i];
    }

    if(root->left!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->left->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->left,node2ranks,treechannel);
    }
    if(root->right!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->right->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->right,node2ranks,treechannel);
    }
  }

  std::vector<MockNcclGroup::DoubleBinaryTreeNode*> MockNcclGroup::genInterDouBinTree(GroupInfo gp_info){
    vector<DoubleBinaryTreeNode*> q;
    vector<DoubleBinaryTreeNode*> tmp_q;
    vector<DoubleBinaryTreeNode*> result;
    int nNodes = gp_info.nNodes; 
    std::vector<int> nodes;
    for(int i = 0;i < nNodes; i++)
      nodes.push_back(i);
    for(int i = 0;i < nodes.size();i++){
      q.push_back(new DoubleBinaryTreeNode(nodes[i]));
    }
    while (q.size() > 1){
      tmp_q = {};
      int i = 0;
      for(i = 0;(i + 2) < q.size();i +=4){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        DoubleBinaryTreeNode* node2 = q[i+2];
        node1->left = node0;
        node1->right = node2;
        tmp_q.push_back(node1);
        if(i+3 < q.size()) {
          DoubleBinaryTreeNode* node3 = q[i+3];
          tmp_q.push_back((node3));
        }
      }
      if(q.size() - i == 1) {
        DoubleBinaryTreeNode* node0 = q[i];
        tmp_q.push_back(node0);
      } else if(q.size() - i == 2){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        node1->left = node0;
        tmp_q.push_back(node1);
      }
      q = tmp_q;
    }
    DoubleBinaryTreeNode* root1 = InterDouBinTreeShift(q[0],nodes);
    int chunk_count = 1;
    for(int i =0;i<chunk_count;i++){
      result.push_back(q[0]);
      result.push_back(root1);
    }
    return result;
  }

  MockNcclGroup::DoubleBinaryTreeNode* MockNcclGroup::InterDouBinTreeShift(DoubleBinaryTreeNode* root,std::vector<int>nodes){
    std::map<int,DoubleBinaryTreeNode*>node2treenode;
    std::map<int,int>rank2index;
    std::queue<DoubleBinaryTreeNode*>q;
    for(int i =0 ;i<nodes.size();i++) {
      node2treenode[nodes[i]] = new DoubleBinaryTreeNode(nodes[i]);
      rank2index[nodes[i]] = i;
    }
    q.push(root);
    while (!q.empty())
    {
      DoubleBinaryTreeNode* current = q.front();
      q.pop();
      int node = current->node;
      int nodeshift = nodes[(rank2index[node] + 1) % nodes.size()];
      DoubleBinaryTreeNode* currentshift = node2treenode[nodeshift];
      if(current->left != nullptr) {
        int leftnode = current->left->node;
        int leftnodeshift = nodes[(rank2index[leftnode] + 1) % nodes.size()];
        currentshift->left = node2treenode[leftnodeshift];
        q.push(current->left);
      }
      if(current->right != nullptr) {
        int rightnode = current->right->node;
        int rightnodeshift = nodes[(rank2index[rightnode] + 1) % nodes.size()];
        currentshift->right = node2treenode[rightnodeshift];
        q.push(current->right);
      }
    }
    return node2treenode[nodes[(rank2index[root->node] + 1) % nodes.size()]];
  }

  ncclInfo* MockNcclGroup::get_algo_proto_info(
      GroupType type,
      int rank,
      AstraSim::ComType op,
      uint64_t data_size) {
    return this->get_algo_proto_info(type, rank, op, data_size, false, false);
  }
  ncclInfo* MockNcclGroup::get_algo_proto_info(
      GroupType type,
      int rank,
      AstraSim::ComType op,
      uint64_t data_size,
      bool msccl,
      bool NCCL_Simple_LL_splitting){
    std::string ncclInfoName ;
    int gp_idx;
    GroupInfo gp_info;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"get_algo_proto_info type %d rank %d op %d data_size %lu msccl %d",type,rank,op,data_size,msccl);
    if(GroupIndex.count(std::make_pair(rank,type))==0){
      NcclLog->writeLog(NcclLogLevel::ERROR,"There is no corresponding group info, resulting in an error with get_algo_proto_info.");
      return nullptr;
    }
    gp_idx = GroupIndex[std::make_pair(rank,type)];
    gp_info = AllGroups[gp_idx];
    switch (type)
    {
    case TP:
      ncclInfoName = "TP";
      break;
    case DP:
      ncclInfoName = "DP";
      break;
    case EP:
      ncclInfoName = "EP";
      break;
    case DP_EP:
      ncclInfoName = "DP_EP";
      break;
    default:
      break;
    }
    ncclInfoName+= "_"+std::to_string(static_cast<int>(op))+"_"+std::to_string(data_size)+"_"+std::to_string(msccl);
    if(nccl_infos.count(ncclInfoName)){
      return nccl_infos[ncclInfoName];
    }else{ 
      bool NVLSenable = false;
      const char* NVLSEnv = std::getenv("AS_NVLS_ENABLE");
      if (NVLSEnv && strcmp(NVLSEnv, "1")==0) {
        NVLSenable = true;
      } else {
        NVLSenable = false;
      }
    struct ncclInfo* info = new ncclInfo();
    if(NCCL_Simple_LL_splitting){
      const uint64_t simple_protocol_threshold = 0; // disabled 2 * 1024 * 1024;
      auto data_size_per_rank = data_size / gp_info.nRanks;
      if (data_size_per_rank > simple_protocol_threshold) {
          NcclLog->writeLog(NcclLogLevel::INFO,"Using NCCL_PROTO_SIMPLE for data_size %lu", data_size);
          info->protocol = NCCL_PROTO_SIMPLE;
      } else {
          NcclLog->writeLog(NcclLogLevel::INFO,"Using NCCL_PROTO_LL for data_size %lu", data_size);
          info->protocol = NCCL_PROTO_LL;
      }
    }else{
      NcclLog->writeLog(NcclLogLevel::INFO,"Using NCCL_PROTO_UNDEF for data_size %lu", data_size);
      info->protocol = NCCL_PROTO_UNDEF;
    }
    info->nBytes = data_size;
    info->nChannels = 0;
    info->coll = static_cast<ncclFunc_t>(op);
    if(msccl==true){
      info->algorithm = NCCL_ALGO_MSCCL;
      nccl_infos[ncclInfoName] = info;
      return info;
    }
    switch (op) {
      case AstraSim::ComType::All_Reduce:
          if(type==TP){
            if(gpu_type==GPUType::A100||gpu_type==GPUType::A800){
              info->algorithm = NCCL_ALGO_RING;
            }else if(gpu_type==GPUType::H100||gpu_type==GPUType::H800){
              if (gp_info.nRanks >= 8 && NVLSenable) {
                info->algorithm = NCCL_ALGO_NVLS;
              } else {
                info->algorithm = NCCL_ALGO_RING;
              }
            } else{
              info->algorithm = NCCL_ALGO_RING;
            }
          } else {
            info->algorithm = NCCL_ALGO_RING;
          }
          break;
      case AstraSim::ComType::All_Gather:
      case AstraSim::ComType::Reduce_Scatter:
      case AstraSim::ComType::All_to_All:
      default:
          info->algorithm = NCCL_ALGO_RING;
          break;
    }
    info->protocol = NCCL_PROTO_UNDEF;
    nccl_infos[ncclInfoName] = info;
    return info;
    }
  }
}