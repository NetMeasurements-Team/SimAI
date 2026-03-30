/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMON_HH__
#define __COMMON_HH__
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "AstraNetworkAPI.hh"

enum class GPUType { A100, A800, H100, H800, NONE, H20};

namespace AstraSim {
#define CLOCK_PERIOD 1
#define FREQ (1000.0 / CLOCK_PERIOD)
#define GBps 1.0 / (1024 * 1024 * 1024)
typedef unsigned long long Tick;
enum class ComType {
  None,
  Reduce_Scatter,
  All_Gather,
  All_Reduce,
  All_to_All,
  All_Reduce_All_to_All,
  All_Reduce_NVLS
};
constexpr const char* to_cstr(const ComType type) noexcept {
  switch (type) {
  case ComType::All_Gather:
    return "allgather";
  case ComType::All_Reduce:
    return "allreduce";
  case ComType::Reduce_Scatter:
    return "reducescatter";
  case ComType::All_to_All:
    return "alltoall";
  default:
    return "unknown";
  }
}
inline std::ostream& operator<<(std::ostream& os, const ComType type) {
  return os << to_cstr(type);
}
enum class CollectiveOptimization { Baseline, LocalBWAware };
enum class CollectiveImplementationType {
  Ring, 
  OneRing,
  Direct, 
  OneDirect,
  AllToAll,
  DoubleBinaryTreeLocalAllToAll,
  LocalRingNodeA2AGlobalDBT,
  HierarchicalRing,
  DoubleBinaryTree,
  HalvingDoubling,  
  OneHalvingDoubling,
  NcclFlowModel,
  NcclTreeFlowModel,
  MscclCustomFlow,
};

inline std::ostream& operator<<(std::ostream& os, CollectiveImplementationType type) { // <--- Add inline here
    switch (type) {
        case CollectiveImplementationType::Ring: os << "Ring"; break;
        case CollectiveImplementationType::OneRing: os << "OneRing"; break;
        case CollectiveImplementationType::Direct: os << "Direct"; break;
        case CollectiveImplementationType::OneDirect: os << "OneDirect"; break;
        case CollectiveImplementationType::AllToAll: os << "AllToAll"; break;
        case CollectiveImplementationType::DoubleBinaryTreeLocalAllToAll: os << "DoubleBinaryTreeLocalAllToAll"; break;
        case CollectiveImplementationType::LocalRingNodeA2AGlobalDBT: os << "LocalRingNodeA2AGlobalDBT"; break;
        case CollectiveImplementationType::HierarchicalRing: os << "HierarchicalRing"; break;
        case CollectiveImplementationType::DoubleBinaryTree: os << "DoubleBinaryTree"; break;
        case CollectiveImplementationType::HalvingDoubling: os << "HalvingDoubling"; break;
        case CollectiveImplementationType::OneHalvingDoubling: os << "OneHalvingDoubling"; break;
        case CollectiveImplementationType::NcclFlowModel: os << "NcclFlowModel"; break;
        case CollectiveImplementationType::NcclTreeFlowModel: os << "NcclTreeFlowModel"; break;
        case CollectiveImplementationType::MscclCustomFlow: os << "MscclCustomFlow"; break;
        default: os << "Unknown"; break; // Handle any future additions gracefully
    }
    return os;
}

enum class CollectiveBarrier { Blocking, Non_Blocking };
enum class SchedulingPolicy { LIFO, FIFO, HIGHEST, None };
enum class IntraDimensionScheduling {
  FIFO,
  RG,
  SmallestFirst,
  LessRemainingPhaseFirst
};
enum class InterDimensionScheduling {
  Ascending,
  OnlineGreedy,
  RoundRobin,
  OfflineGreedy,
  OfflineGreedyFlex
};
enum class InjectionPolicy {
  Infinite,
  Aggressive,
  SemiAggressive,
  ExtraAggressive,
  Normal
};
enum class PacketRouting { Hardware, Software };
enum class BusType { Both, Shared, Mem };
enum class StreamState {
  Created,
  Transferring,
  Ready,
  Executing,
  Zombie,
  Dead
};
enum class EventType {
  NONE,
  RendezvousSend,
  RendezvousRecv,
  CallEvents,
  PacketReceived,
  PacketSent,
  PacketSentFinshed,
  WaitForVnetTurn,
  NCCL_General,
  General,
  TX_DMA,
  RX_DMA,
  Wight_Grad_Comm_Finished,
  Input_Grad_Comm_Finished,
  Fwd_Comm_Finished,
  Wight_Grad_Comm_Finished_After_Delay,
  Input_Grad_Comm_Finished_After_Delay,
  Fwd_Comm_Finished_After_Delay,
  Workload_Wait,
  Reduction_Ready,
  Rec_Finished,
  Send_Finished,
  Processing_Finished,
  Delivered,
  NPU_to_MA,
  MA_to_NPU,
  Read_Port_Free,
  Write_Port_Free,
  Apply_Boost,
  Stream_Transfer_Started,
  Stream_Ready,
  Consider_Process,
  Consider_Retire,
  Consider_Send_Back,
  StreamInit,
  StreamsFinishedIncrease,
  CommProcessingFinished,
  NotInitialized
};

inline std::string getEventTypeString(EventType event) {
    switch (event) {
        case EventType::NONE:
            return "NONE";
        case EventType::RendezvousSend:
            return "RendezvousSend";
        case EventType::RendezvousRecv:
            return "RendezvousRecv";
        case EventType::CallEvents:
            return "CallEvents";
        case EventType::PacketReceived:
            return "PacketReceived";
        case EventType::PacketSent:
            return "PacketSent";
        case EventType::PacketSentFinshed:
            return "PacketSentFinshed";
        case EventType::WaitForVnetTurn:
            return "WaitForVnetTurn";
        case EventType::NCCL_General:
            return "NCCL_General";
        case EventType::General:
            return "General";
        case EventType::TX_DMA:
            return "TX_DMA";
        case EventType::RX_DMA:
            return "RX_DMA";
        case EventType::Wight_Grad_Comm_Finished:
            return "Wight_Grad_Comm_Finished";
        case EventType::Input_Grad_Comm_Finished:
            return "Input_Grad_Comm_Finished";
        case EventType::Fwd_Comm_Finished:
            return "Fwd_Comm_Finished";
        case EventType::Wight_Grad_Comm_Finished_After_Delay:
            return "Wight_Grad_Comm_Finished_After_Delay";
        case EventType::Input_Grad_Comm_Finished_After_Delay:
            return "Input_Grad_Comm_Finished_After_Delay";
        case EventType::Fwd_Comm_Finished_After_Delay:
            return "Fwd_Comm_Finished_After_Delay";
        case EventType::Workload_Wait:
            return "Workload_Wait";
        case EventType::Reduction_Ready:
            return "Reduction_Ready";
        case EventType::Rec_Finished:
            return "Rec_Finished";
        case EventType::Send_Finished:
            return "Send_Finished";
        case EventType::Processing_Finished:
            return "Processing_Finished";
        case EventType::Delivered:
            return "Delivered";
        case EventType::NPU_to_MA:
            return "NPU_to_MA";
        case EventType::MA_to_NPU:
            return "MA_to_NPU";
        case EventType::Read_Port_Free:
            return "Read_Port_Free";
        case EventType::Write_Port_Free:
            return "Write_Port_Free";
        case EventType::Apply_Boost:
            return "Apply_Boost";
        case EventType::Stream_Transfer_Started:
            return "Stream_Transfer_Started";
        case EventType::Stream_Ready:
            return "Stream_Ready";
        case EventType::Consider_Process:
            return "Consider_Process";
        case EventType::Consider_Retire:
            return "Consider_Retire";
        case EventType::Consider_Send_Back:
            return "Consider_Send_Back";
        case EventType::StreamInit:
            return "StreamInit";
        case EventType::StreamsFinishedIncrease:
            return "StreamsFinishedIncrease";
        case EventType::CommProcessingFinished:
            return "CommProcessingFinished";
        case EventType::NotInitialized:
            return "NotInitialized";
        default:
            return "Unknown EventType"; // Handle any unlisted values
    }
}

/**
 * @brief Overloads the stream insertion operator (<<) for EventType.
 *
 * This operator allows EventType enum values to be directly printed to
 * an output stream (like std::cout) using the `<<` operator. It internally
 * calls the `getEventTypeString` function to get the string representation.
 *
 * @param os The output stream to write to.
 * @param event The EventType enum value to print.
 * @return A reference to the output stream, allowing for chaining.
 */
inline std::ostream& operator<<(std::ostream& os, EventType event) {
    os << getEventTypeString(event);
    return os;
}

class CloneInterface {
 public:
  virtual CloneInterface* clone() const = 0;
  virtual ~CloneInterface() = default;
};
class CollectiveImplementation : public CloneInterface {
 public:
  CollectiveImplementationType type;
  CollectiveImplementation(CollectiveImplementationType type) {
    this->type = type;
  };
  virtual CloneInterface* clone() const {
    return new CollectiveImplementation(*this);
  }
  friend std::ostream& operator<<(std::ostream& os, const CollectiveImplementation& obj) {
      os << "CollectiveImplementationType: " << obj.type;
      return os;
  }
};
class DirectCollectiveImplementation : public CollectiveImplementation {
 public:
  int direct_collective_window;
  CloneInterface* clone() const {
    return new DirectCollectiveImplementation(*this);
  };
  DirectCollectiveImplementation(
      CollectiveImplementationType type,
      int direct_collective_window)
      : CollectiveImplementation(type) {
    this->direct_collective_window = direct_collective_window;
  }
};

template <class T>
struct PrintableVector {
  const std::vector<T>& v;
};

template <class T>
PrintableVector<T> asPrintable(const std::vector<T>& v) { return {v}; }

template <class T>
std::ostream& operator<<(std::ostream& os, PrintableVector<T> w) {
  os << "[";
  for (size_t i = 0; i < w.v.size(); ++i) {
    os << w.v[i];
    if (i + 1 != w.v.size()) os << ",";
  }
  os << "]";
  return os;
}

inline int pos_mod(const int a, const int m) {
  const int r = a % m;
  return r < 0 ? r + m : r;
}

} // namespace AstraSim
#endif
