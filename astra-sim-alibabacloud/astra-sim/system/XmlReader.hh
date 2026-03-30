#include <string>
#include <vector>
#include <map>
#include <algorithm>

// Represents a single step within a thread block <step>
struct MscclStep {
    int id;
    std::string type;
    std::string src_buffer;
    int src_offset;
    std::string dst_buffer;
    int dst_offset;
    int count;
    int dep_id;
    int dep_step;
    bool has_dependence;
};

// Represents a thread block <tb>
struct MscclThreadBlock {
    int id;
    int send_peer;
    int recv_peer;
    int channel;
    std::vector<MscclStep> steps;
};

// Represents the operations for a single GPU <gpu>
struct MscclGpu {
    int id;
    std::vector<MscclThreadBlock> thread_blocks;
};

// Represents the entire MSCCL algorithm <algo>
struct MscclAlgorithm {
    std::string name;
    std::string filename;
    std::string coll_type;
    int n_gpus;
    int n_channels;
    int nchunksperloop;
    int maxBytes;
    int minBytes;
    std::map<int, MscclGpu> gpus; // Map from GPU ID to its operations
};

MscclAlgorithm parseMscclXml(const std::string& filepath);