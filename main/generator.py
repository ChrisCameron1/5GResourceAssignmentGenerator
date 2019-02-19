################################################################################
# 5G Resource Allocation Generator
# Copyright (c) 2019 Huawei Technologies Ltd.
# All rights reserved.
################################################################################
#
#    This file is part of 5G Resource Allocation Generator.
#
#    5G Resource Allocation Generator is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    5G Resource Allocation Generator is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with 5G Resource Allocation Generator.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import argparse
import json
import os
import random
import sys

from problem.traffic.MultivariateNormalIncrementProcess import MultivariateNormalIncrementProcess
from problem.AllocationProblem import AllocationProblem
from problem.network.DataCentre import DataCentre
from problem.network.InternetGateway import InternetGateway
from problem.network.Network import Network
from problem.network.Rack import Rack
from problem.network.RemoteRadioHead import RemoteRadioHead
from problem.network.Server import Server
from problem.slices.Slice import Slice
from problem.traffic.TrafficDistribution import TrafficDistribution
from problem.traffic.Traffic import Traffic
from problem.Options import Options

def get_network_from_parameterization(num_datacentres=None,
                                      racks_per_datacentre_range=None,
                                      servers_per_rack_range=None,
                                      rack_types=None,
                                      server_types=None,
                                      server_cpu_settings=None,
                                      server_mem_settings=None,
                                      server_io_settings=None,
                                      cpu_clock_speeds=None,
                                      phy_processors_per_rack=None,
                                      remoteradioheads_per_datacentre_range=None,
                                      remoteradioheads_per_phy_processor=None,
                                      remoteradiohead_bandwidth=None,
                                      internet_gateway_bandwidth=None,
                                      inter_server_delay=None,
                                      inter_rack_delay=None,
                                      inter_datacentre_delay=None,
                                      inter_datacentre_bandwidth=None
                                      ):
    print("Generating Network...")

    # Build possible server configurations
    servers = []
    for i in range(server_types):
        servers.append(Server.get_random(server_cpu_settings=server_cpu_settings,
                                         server_mem_settings=server_mem_settings,
                                         server_IO_settings=server_io_settings,
                                         cpu_clock_speeds=cpu_clock_speeds))
    # Get single Phy Processor configuration
    phy_processor = Server.get_random(server_cpu_settings=server_cpu_settings,
                                      server_mem_settings=server_mem_settings,
                                      server_IO_settings=server_io_settings,
                                      cpu_clock_speeds=cpu_clock_speeds)

    # Build different rack configurations
    racks = []
    rack_counter = 0
    for rack_type in range(rack_types):
        # Each rack has homogeneous group of servers. Pick one of our server types
        random_server = random.choice(servers)
        # Select how many servers to have within this rack
        num_servers = random.randint(servers_per_rack_range[0], servers_per_rack_range[1])
        # Set number of phyprocessors
        num_phy_processors = max(1,
                                 int(num_servers * phy_processors_per_rack))

        phy_processors_in_rack = [phy_processor.copy_server(name='PHY-%d' % i) for i in range(num_phy_processors)]
        servers_in_rack = [random_server.copy_server(name='%d' % i) for i in
                           range(num_servers - num_phy_processors)] + phy_processors_in_rack
        racks.append(Rack(name='%d' % rack_counter,
                          servers=servers_in_rack,
                          phy_processors=phy_processors_in_rack,
                          inter_server_delay=inter_server_delay))
        rack_counter += 1

    # Build Datacentres
    data_centres = []
    # Core datacentre is central datacentre where others are connected to in hub-and-spoke network
    core_datacentre = None
    '''
          d
          |
    d--central--d
          |
          d
    '''
    if num_datacentres < 2:
        raise Exception("Must be at least 2 datacentres: {Radio Access Network, Core Network}")
    for data_centre_counter in range(num_datacentres):
        # Select number of racks
        num_racks = random.randint(racks_per_datacentre_range[0], racks_per_datacentre_range[1])
        datacentre_racks = []
        remote_radio_head = RemoteRadioHead(name='', bandwidth=remoteradiohead_bandwidth)
        # Select number for remote radio heads
        # 2 per RRH
        num_remote_radio_heads = None
        if remoteradioheads_per_datacentre_range:
            num_remote_radio_heads = random.randint(remoteradioheads_per_datacentre_range[0],
                                                    remoteradioheads_per_datacentre_range[1])
            remote_radio_heads = [remote_radio_head.get_copy(name='RRH-%d' % i) for i in range(num_remote_radio_heads)]
        else:
            remote_radio_heads = []

        rrh_counter = 0
        for j in range(num_racks):
            rack = random.choice(racks).get_copy(name='%d' % j)
            if not num_remote_radio_heads:
                rrh_per_rack = len(rack.phy_processors) * remoteradioheads_per_phy_processor
                remote_radio_heads += [remote_radio_head.get_copy(name='RRH-%d' % (i+rrh_counter)) for i in range(rrh_per_rack)]
                rrh_counter = rrh_per_rack + rrh_counter

            datacentre_racks.append(rack)

        internet_gateway = InternetGateway(name='%d' % (data_centre_counter), bandwidth=internet_gateway_bandwidth)

        if data_centre_counter == 0:

            datacentre = DataCentre(name='%d' % data_centre_counter,
                                    racks=datacentre_racks,
                                    inter_rack_delay=inter_rack_delay,
                                    remote_radio_heads=[],  # CORE networks have no connected remote radio heads
                                    internet_gateway=internet_gateway,
                                    type='CORE')
            core_datacentre = datacentre
            data_centres.append(datacentre)

        else:
            datacentre = DataCentre(name='%d' % data_centre_counter,
                                    racks=datacentre_racks,
                                    inter_rack_delay=inter_rack_delay,
                                    remote_radio_heads=remote_radio_heads,
                                    internet_gateway=internet_gateway,
                                    type='RAN')
            data_centres.append(datacentre)

    # Need to create matrix representing the bandwidth and delay between connected of servers.
    # Assumption: Core network connects to all datacentres while other C-RAN datacentres connect only to core network
    data_centre_names = []
    for data_centre in data_centres:
        data_centre_names.append(data_centre.name)

    connected_datacentre_pairs = []
    for i in range(len(data_centre_names) - 1):
        connected_datacentre_pairs.append([core_datacentre.name, data_centres[i + 1].name])

    inter_data_centre_bandwidth = (connected_datacentre_pairs, inter_datacentre_bandwidth)
    inter_data_centre_delay = (connected_datacentre_pairs, inter_datacentre_delay)

    # Create Network
    network = Network(data_centres=data_centres,
                      inter_data_centre_bandwidth=inter_data_centre_bandwidth,
                      inter_data_centre_delay=inter_data_centre_delay)

    # Check if network is valid json
    j = str(network.to_json())
    j = j.replace("'", '"')
    if not is_json(j):
        raise Exception("Network is not valid JSON!")

    network = Network.from_json(json.loads(j))
    return network


def get_slices_from_parameterization(slice_names=None, json_slices=None, random_slices=None):
    print("Generating Slices...")
    slices = []

    # Example Slices
    for slice_name in slice_names:
        slice = Slice.get_example_slice(name=slice_name)
        # Check if can be converted to valid JSON
        j = str(slice.to_json())

        j = j.replace("'", '"')
        if not is_json(j):
            print(j)
            raise Exception("Slice %s not able to be converted to JSON!")
        slice = Slice.from_json(json.loads(j))

        slices.append(slice)

    # JSON Slices
    if json_slices:
        for json_slice in json_slices:
            json_formatted_slice = json.load(json_slice)
            slices.append(Slice.from_json(json_formatted_slice))

    # Random Slices
    if random_slices:
        for random_slice in random_slices:
            # reweighting: [slice_name], num_functions: [number functions in slice], delay: []
            slice_type, parameter = random_slice.split(':')
            if slice_type == 'reweighting':
                slices.append(Slice.get_example_slice(name=parameter))
                raise Exception("Reweighting of example slice not implemented!")
            elif slice_type == 'num_functions':
                slice = Slice.get_random(length=parameter)
                if not is_json(slice.to_json()):
                    raise Exception("Slice %s not able to be converted to JSON!")
                slices.append(slice)

    return slices


def get_traffic_from_parameterization(correlations=None,
                                      traffic_proportions=None,
                                      proportion_variance_args=None,
                                      time_intervals=None,
                                      approx_network_load=None,
                                      slices=None,
                                      network=None):
    print("Generating Traffic Distribution...")
    parsed_correlations = []
    parsed_slice_proportions = {}
    proportion_variance = {}
    for tuple in (correlations):
        parsed_correlation = tuple.split(':')  # (slice,slice,correlation) triples
        parsed_correlation[2] = float(parsed_correlation[2])
        parsed_correlations.append(parsed_correlation)
    for slice_proportion in traffic_proportions:
        slice, proportion = slice_proportion.split(':')
        parsed_slice_proportions[slice] = float(
            proportion)  # If not one for each slice, make uniform. Make dictionary
    for slice_variance in proportion_variance_args:
        slice, proportion = slice_variance.split(':')
        proportion_variance[slice] = float(proportion)

    if (sum(parsed_slice_proportions.values()) != 1.0):
        print("Warning: Slice proportion do not sum to 1! Renormalizing")
        cumulative_slice_proportions = sum(parsed_slice_proportions.values())
        for slice in parsed_slice_proportions:
            parsed_slice_proportions[slice] = parsed_slice_proportions[slice] / cumulative_slice_proportions

    max_phy_traffic_load = 0
    for slice in slices:
        slice_phy_traffic_load = slice.phy_traffic_load()
        if slice_phy_traffic_load > max_phy_traffic_load:
            max_phy_traffic_load = slice_phy_traffic_load
    min_phy_traffic_capacity = network.min_phy_capacity() / max_phy_traffic_load

    traffic_distribution = MultivariateNormalIncrementProcess.parameterize(
        slices=slices,
        network=network,
        time_intervals=time_intervals,
        correlations=parsed_correlations,
        proportions=parsed_slice_proportions,
        proportion_variance=proportion_variance,
        constraining_resource='CPU',
        approx_network_load=approx_network_load,
        min_phy_traffic_capacity=min_phy_traffic_capacity)

    return traffic_distribution


def write_json_allocation_problem(network=None,
                                  slices=None,
                                  traffic=None,
                                  options=None,
                                  filename=None):

    traffic_json = str(traffic.to_json()).replace("'", '"')
    if not is_json(traffic_json):
        raise Exception("Traffic problem is not convertible into JSON format")
    traffic_sample = Traffic.from_json(json.loads(traffic_json))

    allocation_problem = AllocationProblem(network=network, slices=slices, traffic=traffic_sample, options=options)
    json_allocation_problem = str(allocation_problem.to_json()).replace("'", '"')

    if not is_json(json_allocation_problem):
        print(json_allocation_problem)
        raise Exception("Allocation problem is not convertible into JSON format")

    # If directory doesn't exist, create it
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
    f = open(filename, 'w')
    json.dump(json_allocation_problem, f)
    f.close()

def is_json(myjson: str):
    try:
        json_object = json.loads(myjson)
    except json.JSONDecodeError as err:
        print(err)
        print(myjson)
        return False
    return True


def parse_command_line_args(args):
    parser = argparse.ArgumentParser(description='Arguments for 5G Resource Allocation Generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.print_help()

    parser.add_argument('--seed', type=float, default=0.0, help='random seed for replication')
    ''' General options '''
    parser.add_argument('--objective_function', type=str, default='energy-prop-fairness',
                        help='{energy,QOS,energy-QOS,energy-prop-fairness}'
                             'energy: minimize racks in use. '
                             'QOS: maximize fraction of traffic allocated. '
                             'energy-QOS: minimize racks in use and maximize fraction of traffic allocated if network overloaded. '
                             'energy-prop-fairness: minimize racks in use and maximize *log* fraction of traffic allocated if network overloaded. ')
    parser.add_argument('--min_cpu_per_server', type=float, default=0.01,
                        help='Minimum CPU resources allocated to specific function on server')
    parser.add_argument('--min_qos', type=float, default=0.5,
                        help='Minimum quality of service for any slice')

    parser.add_argument('--num_problems', type=int, default=12,
                        help='Number of problems to generate. Each problem has fixed slices and network with varying traffic over remote radio heads')
    parser.add_argument('--output_directory', type=str, default="./instances",
                        help="Directory to save JSON generated files")
    parser.add_argument('--name', type=str, default="5gAllocation",
                        help="Prefix for name of generated instances. If verbose_filename turned off -> files saved to [output_directory]/[name]_[instance number].json")
    parser.add_argument('--verbose_filename', type=bool, default=False,
                        help="Include network and slice attributes in the JSON filenames")

    ''' Option 1: Set Network, Slices, and Traffic from existing JSONs'''
    parser.add_argument('--network_json', type=str, default=None,
                        help='Loads network object representing network resources / topology from provided JSON file. '
                             'This overrides all other network parameters. Do not set if you want to build network from parameterization!')
    parser.add_argument('--slices_json', type=str, default=None,
                        help='Loads Slices object representing network slices from provided JSON file. '
                             'This overrides all other slice parameters. Do not set if you want to build slices from parameterization!')
    parser.add_argument('--traffic_json', type=str, default=None,
                        help='Loads Traffic Object representing model of traffic from provided JSON file. '
                             'This overrides all other traffic parameters. Do not set if you want to build traffic from parameterization!')

    ''' Build Network, Slices, and Traffic from parameters
        Provide ability to give high level parameters with some level of detail without having to specify everything.
        Pieces that are not specified completely in options'''

    ## Network

    ## Topology
    parser.add_argument('--num_datacentres', type=int, default=2, help='Number of data centres. n-1 are designated as CRAN data centres. ')
    parser.add_argument('--racks_per_datacentre_range', nargs='+', type=int, default=[16, 64], help='Number of racks sampled uniformly over given range')
    parser.add_argument('--servers_per_rack_range', nargs='+', type=int, default=[16, 32], help='Number of servers per rack sampled uniformly over given range')
    parser.add_argument('--rack_types', type=int, default=3, help='Number of rack configurations. Each rack configuration samples server type and number of servers.')
    parser.add_argument('--server_types', type=int, default=3, help='Number of server configurations. Server configurations are randomly sampled from CPU,MEM,IO, and clock speed settings.')
    parser.add_argument('--server_cpu_settings', type=int, nargs='+', default=[8, 16], help='Server CPU setting options')
    parser.add_argument('--server_mem_settings', type=int, nargs='+', default=[8, 16], help='Server memory setting options')
    parser.add_argument('--server_io_settings', type=int, nargs='+', default=[10, 40], help='Server I/O setting options')
    parser.add_argument('--cpu_clock_speeds', type=float, nargs='+', default=[2.7, 3.3], help='Server Clock speed options')

    parser.add_argument('--phy_processors_per_rack', type=int, default=0.25, help='Proportion of phy processors per rack')
    parser.add_argument('--remoteradioheads_per_datacentre_range', nargs='+', type=int, default=None,
                        help='Range of number of remote radio heads per CRAN data centre')  # default=[10,100]
    parser.add_argument('--remoteradioheads_per_phy_processor', type=int, default=2,
                        help='Ratio of remote radio heads to phy processors for every data centre')
    parser.add_argument('--remoteradiohead_bandwidth', type=float, default=100.0,
                        help='Bandwidth of remote radio heads is Gb/s')
    parser.add_argument('--internet_gateway_bandwidth', type=float, default=100.0,
                        help='Bandwidth of remote radio heads is Gb/s')

    # Delay
    parser.add_argument('--inter_server_delay', type=float, default=1.0,
                        help='Delay of travelling across servers between connected functions')
    parser.add_argument('--inter_rack_delay', type=float, default=10.0,
                        help='Delay of travelling across racks between connected functions')
    parser.add_argument('--inter_datacentre_delay', type=float, default=30.0,
                        help='Delay of travelling across data centres between connected functions')
    parser.add_argument('--inter_datacentre_bandwidth', type=float, default=1000.0,
                        help='Bandwidth constraints between data centres')

    # parser.add_argument

    # Slices
    parser.add_argument('--slice_names', type=str, nargs='+',
                        default=["eMBB", "MMTC", "URLLC"],  # ["embb", "CachedVideo", "mMTC", "URLLC", "smartgrid"]
                        help='Comma separated lists of Slice names {embb,CachedVideo,mmtc,URLLC,smartgrid}, and/or json filename for slice and/or random option for slice with random weights')
    parser.add_argument('--additional_json_slices', type=str, nargs='+', default=None)
    parser.add_argument('--random_slices', type=str, nargs='+', default=None,
                        help="{reweighting:[slice_name], num_functions:[number functions in slice], delay:[]}")

    # Traffic
    parser.add_argument('--correlations', type=str, nargs='+', default=['MMTC:eMBB:0.5'],
                        help='Comma separated list of [slice_name]:[slice_name]:covariance triples. By default 0.')
    parser.add_argument('--traffic_proportions', type=str, nargs='+', default=['eMBB:0.5', 'MMTC:0.25', 'URLLC:0.25'],
                        help='Comma separated list of [slice_name]:[proportion] pairs. By default uniform proportion of traffic')
    parser.add_argument('--on_off_distribution', type=str, default='normal',
                        help='Distribution representing traffic arriving/leaving network. Always mean 0.')
    parser.add_argument('--proportion_variance', type=str, nargs='+', default=['eMBB:0.01','MMTC:0.01','URLLC:0.1'],
                        help='Variance of on/off distribution as a function of mean traffic. List of [slice_name]:[proportion] pairs.')
    parser.add_argument('--remote_radio_head_split', type=str, default='random_partition',
                        help='Partition of traffic over radioheads. By default traffic is randomly partitioned over radio heads. There are currently no other options implemented.')
    parser.add_argument('--approx_network_load', type=float, default=0.8,
                        help='Approx proportion of network loaded for optimal allocation')
    parser.add_argument('--time_intervals', type=int, default=3,
                        help='Number of time intervals samples over')

    if len(args) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args(args)


def main(args):
    print("Parsing Arguments...")

    args = parse_command_line_args(args)

    # Set random seed
    random.seed(args.seed)

    """
    If provided, translate JSON representation for any of network, slices, and traffic distribution
    Otherwise, generate representation given the parameterization. 
    """
    if args.network_json:
        json_network = json.load(args.network_json)
        network = Network.from_json(json_network)
    else:
        network = get_network_from_parameterization(num_datacentres=args.num_datacentres,
                                                    racks_per_datacentre_range=args.racks_per_datacentre_range,
                                                    servers_per_rack_range=args.servers_per_rack_range,
                                                    rack_types=args.rack_types,
                                                    server_types=args.server_types,
                                                    server_cpu_settings=args.server_cpu_settings,
                                                    server_mem_settings=args.server_mem_settings,
                                                    server_io_settings=args.server_io_settings,
                                                    cpu_clock_speeds=args.cpu_clock_speeds,
                                                    phy_processors_per_rack=args.phy_processors_per_rack,
                                                    remoteradioheads_per_datacentre_range=args.remoteradioheads_per_datacentre_range,
                                                    remoteradioheads_per_phy_processor=args.remoteradioheads_per_phy_processor,
                                                    remoteradiohead_bandwidth=args.remoteradiohead_bandwidth,
                                                    inter_server_delay=args.inter_server_delay,
                                                    inter_rack_delay=args.inter_rack_delay,
                                                    inter_datacentre_delay=args.inter_datacentre_delay,
                                                    inter_datacentre_bandwidth=args.inter_datacentre_bandwidth,
                                                    internet_gateway_bandwidth=args.internet_gateway_bandwidth)

    if args.slices_json:
        json_slices = json.load(args.slices_json)
        slices = Slice.from_json(json_slices)
    else:
        slices = get_slices_from_parameterization(slice_names=args.slice_names, json_slices=args.additional_json_slices,
                                                  random_slices=args.random_slices)

    if args.traffic_json:
        json_traffic = json.load(args.traffic_json)
        traffic_distribution = TrafficDistribution.from_json(json_traffic)
    else:
        traffic_distribution = get_traffic_from_parameterization(correlations=args.correlations,
                                                             traffic_proportions=args.traffic_proportions,
                                                             proportion_variance_args=args.proportion_variance,
                                                             time_intervals=args.time_intervals,
                                                             approx_network_load=args.approx_network_load,
                                                             slices=slices,
                                                             network=network
                                                             )



    traffic_samples = traffic_distribution.get_samples(num_samples=args.num_problems, network=network)

    options = Options(objective_function=args.objective_function,
                      min_cpu_per_server=args.min_cpu_per_server,
                      min_qos=args.min_qos)

    print("Writing %d instances to %s" % (len(traffic_samples), args.output_directory))
    output_instances = []
    instance_count = 0
    for traffic_sample in traffic_samples:
        if args.verbose_filename:
            filename = os.path.join(args.output_directory, '%s_%s_%s_%s_%s_%d.json' % (
                args.name, args.num_datacentres, args.racks_per_datacentre_range[1], args.servers_per_rack_range[1],
                args.approx_network_load, instance_count))
        else:
            filename = os.path.join(args.output_directory, '%s_%d.json' % (args.name, instance_count))

        instance_name = write_json_allocation_problem(network=network,
                                                      slices=slices,
                                                      traffic=traffic_sample,
                                                      options=options,
                                                      filename=filename)
        output_instances.append(instance_name)
        instance_count += 1

    return output_instances


if __name__ == '__main__':
    main(sys.argv[1:]) #[:1][1:]
