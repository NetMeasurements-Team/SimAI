## Enable tracing output

Tracing is disabled by default. Enable it through the `TRACE_FILE` file specified in `SimAI.conf`.

The default value for `TRACE_FILE` is `/etc/astra-sim/simulation/trace1.txt`. To enable tracing, you should use this
 file to specify which devices you want to trace. For instance:

```
2 4 5
```

means: _enable tracing for two nodes, those with ids 4 and 5_

This file is parsed in `common.h` and used in `SetupNetwork`, where tracing is enabled for the specified nodes:

```
qbb.EnableTracing(trace_output, trace_nodes);
```

This will generate an output file ending with `.tr` (`TRACE_OUTPUT_FILE` in `SimAI.conf`). This is a binary file listing 
 per-packet traces for the specified devices. To read it, do as follows:

```shell
$ cd ns-3-alibabacloud/analysis
$ make trace_reader
g++ trace_reader.cpp -o trace_reader -O3 -std=gnu++11
$ ./trace_reader /etc/astra-sim/simulation/test_mon.mix.tr
6001235 n:5 2:0 0 Recv ecn:0 0b000101 0b000001 10000 100 U 0 0 3 9052(9000)
6001235 n:5 1:3 0 Enqu ecn:0 0b000101 0b000001 10000 100 U 0 0 3 9052(9000)
6001235 n:5 1:3 0 Dequ ecn:0 0b000101 0b000001 10000 100 U 0 0 3 9052(9000)
6001235 n:4 1:0 0 Recv ecn:0 0b000001 0b000101 10000 100 U 0 0 3 9052(9000)
6001235 n:4 2:3 0 Enqu ecn:0 0b000001 0b000101 10000 100 U 0 0 3 9052(9000)
6001235 n:4 2:3 0 Dequ ecn:0 0b000001 0b000101 10000 100 U 0 0 3 9052(9000)
6001799 n:5 2:0 0 Recv ecn:0 0b000101 0b000001 10000 100 U 9000 0 3 7052(7000)
6001799 n:5 1:3 0 Enqu ecn:0 0b000101 0b000001 10000 100 U 9000 0 3 7052(7000)
6001799 n:4 1:0 0 Recv ecn:0 0b000001 0b000101 10000 100 U 9000 0 3 7052(7000)
6001799 n:4 2:3 0 Enqu ecn:0 0b000001 0b000101 10000 100 U 9000 0 3 7052(7000)
6001959 n:5 1:3 0 Dequ ecn:0 0b000101 0b000001 10000 100 U 9000 0 3 7052(7000)
6001959 n:4 2:3 0 Dequ ecn:0 0b000001 0b000101 10000 100 U 9000 0 3 7052(7000)
6002963 n:5 1:0 0 Recv ecn:0 0b000001 0b000101 100 10000 A 0x00 3 9000 0 60
6002963 n:5 2:3 0 Enqu ecn:0 0b000001 0b000101 100 10000 A 0x00 3 9000 0 60
6002963 n:5 2:3 0 Dequ ecn:0 0b000001 0b000101 100 10000 A 0x00 3 9000 0 60
```

Please refer to [this](ns-3-alibabacloud/analysis/README.format.md) for a full description of the trace format.
