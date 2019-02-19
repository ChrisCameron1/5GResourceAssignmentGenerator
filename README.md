# Problem Generator for 5G Network Resource Assignment

Copyright (C) 2018-2019 Huawei Technologies Ltd.

This is a generator for resource assignment problems on future 5G mobile networks.
5G mobile networking will allow for different services customized for different types of communications (e.g., low-latency, high bandwidth)
Rather than having dedicated hardware for different network functions, 5G networks will have virtualized 
functions that are dynamically allocated to a generic server pool based on network traffic.
These generated problems represent the network hardware and snapshots of network traffic needed to be allocated.
The main objectives of this generator are to:

1. provide encoding-agnostic problem specification in JSON/python both for dataset sharing and as a common source from which different problem encodings can be compared
2. model realistic network topologies and network functions
3. model realistic distributions of network traffic to build benchmarks for exploitation by algorithm configuration


Python3+. Tested in python3.6.

For a description of the problem statement, generator and encoding, please see:
```angular2html
See Cameron, C. and Hoos, H. H. and Leyton-Brown, K. and Mccormick, B.
Efficient CRAN Resource Assignment for Virtualized Slices in 5G Networks
Currently under review at SIGCOMM 2019
```

## Installation
On a linux machine:
```angular2html
tar -zxvf 5GResourceAssignmentGenerator.zip
cd 5GResourceAssignmentGenerator
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

## License

This project is under the GPL3 license. You should have received a copy of the GPL3 license along with this program (see license.txt file)
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

## Contact 

5G Resource Assignment Generator is developed at the University of British Columbia in partnership with Huawei Technologies Ltd.
If you found a bug or have a question, please email cchris13@cs.ubc.ca

## Overview

1. Problem Summary
2. Directory Structure
3. Quickstart
4. JSON Encoding
5. TODOs

## Problem Summary

Here is a brief description of the problem.

### Objective
The objective is to minimize the number of server resources in use, which corresponds th energy savings for network operator.
We cannot guarantee all traffic be serviced by the network. If the network is overloaded we must choose not to service some traffic.
In this case, we deny some network requests while maximizing quality of service subject to fairness constraints.

### Problem structure

What defines problem:

#### Traffic
- Small (3-6) number of "types" (or slices) of requests that a client might make. 
- Each request "type" is a service chain, which is a short (5-10) directed weighted acyclic graph of virtualized functions.
- Weightings on outgoing edges from any virtualized function in the directed graph corresponds to the proportion of traffic outgoing and must sum to 1
- Each slice has traffic bitrate representing a snapshot of the network traffic
- Each virtualized function has a set of resource requirements measured as a function of traffic bitrate for the slice
- A slice may have transit delay constraint that limits the number of server hops between functions in a slice

#### Network

- Set of remote radio heads with throughput capacities that directly connect to the clients 
- Set of data centres containing servers with specific resource capacities. There a small number of resource profiles a server might have.
- Some servers are dedicated as PhyProcessors which are the interface between remote radio heads and other servers. Always the first function within a slice.
- Each pair of servers has an associated delay depending if the servers are in same rack and/or same data centre.
- Pairs of data centres have bandwidth constraints


### Constraints for solving problem
This is the criteria for a valid allocation:

1. Each virtualized function needs to be allocated enough resources (e.g., CPU, MEM, I/O) on servers
2. Resource constraints on servers cannot be exceeded
3. Virtualized functions can be split across servers as long as the resources allocated are in the same proportion within every server (e.g., if function requires 2GB memory and 2 CPUs, we could allocate 1GB and 1 CPU to two different servers but not 2 CPUs to one server and 2 GB to another )
4. Bandwidth constraints between data centres cannot be exceeded
5. A RemoteRadioHead can be connected to at most one server (PhyProcessor) but server can connect to many remote radio heads
6. Some service chains have a delay constraint. If data jumps between servers along a service chain, a delay cost is incurred. For any chunk of data passing through service chain, the total sum of delays must not exceed some constant.


## Directory Structure
```angular2html
+-- main
|   +-- generator.py: main method for generating problems
|   +-- mip_encoding.py: main method for encoding generated problem into .lp format
+-- problem
|   +-- network: files for different pieces of network architecture
    |   +-- ...
|   +-- slices: files for setting up virtualized slices
    |   +-- ...
|   +-- traffic: files for sampling traffic across slices / remote radio heads
    |   +-- ...
    +-- AllocationProblem.py: class representing fill allocation problem
    +-- Options.py: class file representing options for optimization problem
README.md: description of generator
license.txt: license file
requirements.txt: required python libraries
setup.py: setup file for installing required libraries
```

## Quickstart

To generate some toy problems, type from the root directory:
```angular2html
python ./main/generator.py --num_datacentres 2 --racks_per_datacentre_range 2 2 --servers_per_rack_range 4 4 --name "5G_toy"
```
This will create problems `5G_toy_0.json`, ..., `5G_toy_11.json` allocation problems in ./instances
with 2 data centres, 2 racks per data centre, and 4 servers per rack.

For a full list of generator parameters, type
```angular2html
python ./main/generator.py --help
```

To build a MIP optimization from one of these problems, execute:

```angular2html
python ./main/mip_encoding.py --problem ./instances/5G_toy_0.json
```
This will store .lp file representing --problem into ./instances/5gAllocation_0.lp. The .lp can then be 
solved with a MIP solvers of your choice.


The user can also do this programmatically in python:
### Call from python 
```angular2html
import generator
from build_mip_problem import encode_distribution

# Generator 100 problems with default settings and store in specified directory
instance_data_directory = './instances'
parameters = ['--num_problems', 100] + \
             ['--output_directory', instance_data_directory]
problem_instances  = generator.main(parameters)

# Encode instances into .lp optimization file
mip_instance_names = encode_distribution(problems=problem_instances, directory=instance_data_directory)
```

## JSON Encoding

Below is a example .json problem. An allocation problem is a dictionary of 4 attributes:

1. network: dictionary of 3 attributes
    * Inter data centre bandwidth
    * Inter data centre delays
    * DataCentres: list of racks, which contain list of servers, which have dictionary of computational resource
2. slices: list of slice dictionary
    * each slice is a dictionary with attributes for: delay, importance, and function graph
    * function graph has nodes and edges attributes. nodes are list of virtual function which have dictionary of resource requirements as function of bitrate. edges are list of pairs of virtual function names
3. traffic: dataframe for bitrate of traffic for every slice and remote radio head
4. options: objective function, min resources function per server, minimum quality of service


```angular2html
﻿{
  "network": {
    "DataCentres": [
      {
        "name": "data_centre",
        "racks": [
          {
            "name": "Rack-0",
            "servers": [
              {
                "name": "Server-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              },
              {
                "name": "Server-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              }
            ],
            "phy_processors": [
              {
                "name": "PHY-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              },
              {
                "name": "PHY-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              }
            ],
            "inter_server_delay": 0
          },
          {
            "name": "Rack-1",
            "servers": [
              {
                "name": "Server-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              },
              {
                "name": "Server-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              }
            ],
            "phy_processors": [
              {
                "name": "PHY-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              },
              {
                "name": "PHY-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              }
            ],
            "inter_server_delay": 0
          },
          {
            "name": "Rack-2",
            "servers": [
              {
                "name": "Server-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              },
              {
                "name": "Server-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              }
            ],
            "phy_processors": [
              {
                "name": "PHY-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              },
              {
                "name": "PHY-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              }
            ],
            "inter_server_delay": 0
          },
          {
            "name": "Rack-3",
            "servers": [
              {
                "name": "Server-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              },
              {
                "name": "Server-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "Server"
              }
            ],
            "phy_processors": [
              {
                "name": "PHY-0",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              },
              {
                "name": "PHY-1",
                "resources": {
                  "CPU": 8,
                  "MEM": 8,
                  "IO": 10,
                  "Clock": 3
                },
                "type": "PHY"
              }
            ],
            "inter_server_delay": 0
          }
        ],
        "inter_rack_delay": 0,
        "radio_heads": [
          {
            "name": "0",
            "bandwidth": 100
          },
          {
            "name": "1",
            "bandwidth": 100
          },
          {
            "name": "2",
            "bandwidth": 100
          },
          {
            "name": "3",
            "bandwidth": 100
          },
          {
            "name": "4",
            "bandwidth": 100
          },
          {
            "name": "5",
            "bandwidth": 100
          },
          {
            "name": "6",
            "bandwidth": 100
          },
          {
            "name": "7",
            "bandwidth": 100
          },
          {
            "name": "8",
            "bandwidth": 100
          },
          {
            "name": "9",
            "bandwidth": 100
          },
          {
            "name": "10",
            "bandwidth": 100
          },
          {
            "name": "11",
            "bandwidth": 100
          }
        ],
        "internet_gateway": {
          "name": "InternetGateway",
          "bandwidth": 100
        }
      }
    ],
    "InterDataCentreBandwidth": [
      1000000
    ],
    "InterDataCentreDelay": [
      0
    ]
  },
  "slices": [
    {
      "function_graph": {
        "nodes": [
          [
            {
              "name": "F1",
              "resources": [
                {
                  "resource_name": "CPU",
                  "quantity_per_load": 1
                },
                {
                  "resource_name": "MEM",
                  "quantity_per_load": 1
                }
              ],
              "phy_processor_constrained": "True"
            },
            {}
          ],
          [
            {
              "name": "F2",
              "resources": [
                {
                  "resource_name": "CPU",
                  "quantity_per_load": 1
                },
                {
                  "resource_name": "MEM",
                  "quantity_per_load": 1
                }
              ],
              "phy_processor_constrained": "False"
            },
            {}
          ]
        ],
        "edges": [
          [
            {
              "name": "F1",
              "resources": [
                {
                  "resource_name": "CPU",
                  "quantity_per_load": 1
                },
                {
                  "resource_name": "MEM",
                  "quantity_per_load": 1
                }
              ],
              "phy_processor_constrained": "True"
            },
            {
              "name": "F2",
              "resources": [
                {
                  "resource_name": "CPU",
                  "quantity_per_load": 1
                },
                {
                  "resource_name": "MEM",
                  "quantity_per_load": 1
                }
              ],
              "phy_processor_constrained": "False"
            },
            {}
          ]
        ]
      },
      "delay_constraint": 0,
      "name": "example",
      "importance": 1
    }
  ],
  "traffic": {
    "traffic": {
      "DataCentre": {
        "0": "data_centre",
        "1": "data_centre",
        "2": "data_centre",
        "3": "data_centre",
        "4": "data_centre",
        "5": "data_centre",
        "6": "data_centre",
        "7": "data_centre",
        "8": "data_centre",
        "9": "data_centre",
        "10": "data_centre",
        "11": "data_centre"
      },
      "RRH": {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "10": "10",
        "11": "11"
      },
      "Slice": {
        "0": "example",
        "1": "example",
        "2": "example",
        "3": "example",
        "4": "example",
        "5": "example",
        "6": "example",
        "7": "example",
        "8": "example",
        "9": "example",
        "10": "example",
        "11": "example"
      },
      "bitrate": {
        "0": 4,
        "1": 4,
        "2": 4,
        "3": 4,
        "4": 4,
        "5": 4,
        "6": 4,
        "7": 4,
        "8": 4,
        "9": 4,
        "10": 4,
        "11": 4
      }
    }
  },
  "options": {
    "objective_function": "energy-QOS",
    "min_cpu_per_server": 0.1,
    "min_qos": 0.5
  }
}
```


## Major TODOs

### Improvements to generator

#### Add locality to remote radio heads. 

If traffic on a remote radio head exceeds capacity of any phy processor, overloaded traffic
should be allocated to geographically nearby remote radio heads.

#### Controlling traffic across experiments

Currently, it difficult to make comparisons about instance hardness between network settings because traffic is sampled independently across experiments.
But we can't just equate traffic because traffic may become overloaded on different networks settings or even infeasible
Need some way to give approximately proportional traffic to different network settings

#### Improve realism in slices, network, and traffic generally.

This will gradually happen as there are more concrete design decisions are made.


### Encoding

#### Independence assumption between CRAN data centres (allows parallelization)

Apparently, control plane function are constrained to the core data centre and user plane functions are constrained to CRAN data centres.
Control plane functions are not on the data plane. Therefore there is never delay between user plane and control plane functions
and no database to database delays. Our encoding allows consecutive user-plane functions to be allocated on different CRAN 
data centres but this means going through a core data centres given hub and spoke topology.

#### Speed up encoding

Building MIP encodings from JSON is very expensive for large problems. This is not necessary. We used a convenient python package
(pulp) to keep track of constraints but this library is quite expensive.

### Optimization

#### Adapt solutions from previous allocation problems given strong similarity

#### Reliability

Servers often fail is practice. Consider optimization procedure that that consider robustness to server failure.


