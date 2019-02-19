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

from __future__ import print_function

__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.AllocationProblem import AllocationProblem
from problem.slices.SliceGroup import SliceGroup
import pandas as pd
from pulp import *
from tqdm import tqdm
import time
import json
import argparse
#import matplotlib
#import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import multiprocessing as mp
import warnings
import random
import math
#matplotlib.use('agg')

#sys.setrecursionlimit(10000)

'''
*************Variables***********
'''
def set_resource_variables(network, slices):

    print('Setting Variables')
    start_time=time.time()
    num_vars = 0


    # Resources: slice,function, resource, server
    d=[]
    exist_non_phy_processor_constrained = False
    for data_centre in network.data_centres:
        for rack in data_centre.racks:
            phy_processor_names = [phy_processor.name for phy_processor in rack.phy_processors]
            non_phy_processor_servers = [server for server in rack.servers if server.name not in phy_processor_names]
            for server in non_phy_processor_servers:
                for slice in slices:
                    slice_name = slice.name
                    for function in slice.virtual_functions:
                        if not function.data_centre_type == data_centre.type:
                            continue
                        if not function.name == "NETWORK_GATEWAY": # special constant for network gateway function
                            if not function.phy_processor_constrained:
                                exist_non_phy_processor_constrained = True
                                for resource in server.resources:
                                    d.append({'DataCentre':data_centre.name,
                                              'Rack':rack.name,
                                              'Server': server.name,
                                              'PHYProcessorConstrained':function.phy_processor_constrained,
                                              'Slice': slice_name,
                                              'Function': function.name,
                                              'Resource': resource,
                                              'ResourceVariable': LpVariable("resource-%s-%s-%s-%s-%s-%s" % (data_centre.name, rack.name, server.name, slice_name, function.name, resource),
                                                                             lowBound=0,
                                                                             upBound=None,
                                                                             cat='Continuous')})
                                    num_vars+=1
            for server in rack.phy_processors:
            # Only add resource to phyProcessor function on PhyProcessor servers
            #if not isinstance(server,  PhyProcessor): # Special global type for PhyProcessor
                for slice in slices:
                    slice_name = slice.name
                    for function in slice.virtual_functions:
                        if not function.data_centre_type == data_centre.type:
                            continue
                        if not function.name == "NETWORK_GATEWAY":  # special constant for network gateway function
                            for resource in server.resources:
                                d.append({'DataCentre':data_centre.name,
                                          'Rack':rack.name,
                                          'Server': server.name,
                                          'PHYProcessorConstrained':function.phy_processor_constrained,
                                          'Slice': slice_name,
                                          'Function': function.name,
                                          'Resource': resource,
                                          'ResourceVariable': LpVariable("resource-%s-%s-%s-%s-%s-%s" % (data_centre.name, rack.name, server.name, slice_name, function.name, resource),
                                                                         lowBound=0,
                                                                         upBound=None,
                                                                         cat='Continuous')})
                                num_vars+=1


    if not exist_non_phy_processor_constrained:
        print("Warning: Every function is phy processor constrained! Check problem description")
        #raise Exception("Every function is phy processor constrained! Check problem description. This must be wrong!")

    # for server in range(NUM_SERVERS_PER_RACK * NUM_RACKS_PER_DATACENTRE * NUM_DATACENTRES):
    #     for slice_name in SLICES:
    #         for function in SLICES[slice_name].virtual_functions:
    #             example_server_type = 'TypeA'
    #             for resource in SERVER_CAPACITY[example_server_type]:
    #                 d.append({'Server': server, 'Slice': slice_name, 'Function': function.name, 'Resource': resource,
    #                           'ResourceVariable': LpVariable("resource-%s-%s-%s-%s" % (server, slice_name, function.name, resource), lowBound=0, upBound=None, cat='Continuous')})

    resource_variables = pd.DataFrame(d)

    # Special Variables for Network Gateway Function
    d=[]
    for data_centre in network.data_centres:
        if data_centre.internet_gateway: # If network gateway exists for data centre
            for slice in slices:
                slice_name = slice.name
                for function in slice.virtual_functions:
                    if function.name == "NETWORK_GATEWAY":
                        if function.phy_processor_constrained:
                            for resource in server.resources:
                                d.append({'DataCentre':data_centre.name,
                                          'Slice': slice_name,
                                          'Resource': resource,
                                          'ResourceVariable': LpVariable("network-gateway-resource-%s-%s-%s" % (data_centre.name,slice_name, resource),
                                                                         lowBound=0,
                                                                         upBound=None,
                                                                         cat='Continuous')})
    network_gateway_variables = pd.DataFrame(d)
    end_time = time.time()
    print("Built %d variables in %f sec" % (num_vars, end_time-start_time))
    return resource_variables, network_gateway_variables

def set_server_rack_variables(network):

    servers_in_use = []
    racks_in_use = []

    # Server in use variables
    for data_centre in network.data_centres:
        for rack in data_centre.racks:
            for server in rack.servers:# + rack.phy_processors:
                servers_in_use.append({'DataCentre':data_centre.name,
                                       'Rack':rack.name,
                                       'Server': server.name,
                                       'Variable': LpVariable("server_in_use-%s-%s-%s" % (data_centre.name, rack.name, server.name),lowBound=0,upBound=1,cat='Binary')})

    # Server in use variables
    for data_centre in network.data_centres:
        for rack in data_centre.racks:
            racks_in_use.append({'DataCentre':data_centre.name,
                                 'Rack':rack.name,
                                 'Variable':LpVariable("rack_in_use-%s-%s" % (data_centre.name, rack.name),lowBound=0,upBound=1,cat='Binary')})

    return pd.DataFrame(servers_in_use), pd.DataFrame(racks_in_use)

def set_slice_qos_vars(slices):
    slice_qos = {}
    slice_weights = {}
    # Slice QOS variable
    for slice in slices:
        slice_name = slice.name
        slice_qos[slice_name] = LpVariable("qos-%s" % (slice_name), lowBound=0, upBound=1, cat='Continuous')
        slice_weights[slice_name] = slice.importance

    return slice_qos, slice_weights

'''
*************Objective Function***********
'''
def set_objective_function(prob, racks_in_use, slice_qos, slice_weights, obj='energy-QOS', discretization=20):
    print("Setting Objective function")
    # Objective Function
    ALPHA = 1
    BETA = 1000
    weighted_qos = []
    for slice_name in slice_qos:
        weighted_qos.append(BETA*slice_weights[slice_name]*slice_qos[slice_name])

    racks_in_use_vars = racks_in_use['Variable']

    if obj == 'energy-QOS':
        prob += lpSum(ALPHA*racks_in_use_vars) - lpSum(weighted_qos)
    elif obj == 'energy':
        prob += lpSum(ALPHA*racks_in_use_vars)
    elif obj == 'QOS':
        prob += lpSum(weighted_qos)
    elif obj == 'energy-prop-fairness':
        log_value = {}
        max_slice_weight = max([slice_weights[i] for i in slice_weights])
        for interval in range(discretization):
            interval+=1
            log_value[interval] = math.log(max_slice_weight * interval / discretization)

        lambda_multipliers = []
        lambda_interpolators = {}
        lambdas = {}
        interval_identifiers = {}
        for slice in slice_qos:
            lambdas[slice] = []
            interval_identifiers[slice] = []
            lambda_interpolators[slice] = []
            lambdas[slice].append(LpVariable("lambda-%s-%s" % (slice, 0),
                                      lowBound=0,
                                      upBound=1,
                                      cat='Continuous'))
            #lambda_interpolators[slice].append(lambdas[slice][-1] * (interval + 1) / discretization))
            for interval in range(discretization):
                interval_identifiers[slice].append(LpVariable("interval-%s-%s" % (slice, interval+1),
                          lowBound=0,
                          upBound=1,
                          cat='Binary'))

                lambdas[slice].append(LpVariable("lambda-%s-%s" % (slice,interval+1),
                          lowBound=0,
                          upBound=1,
                          cat='Continuous'))

                lambda_multipliers.append(BETA * lambdas[slice][-1] * log_value[interval+1])
                # Multiply slice weights
                lambda_interpolators[slice].append(slice_weights[slice_name] * lambdas[slice][-1] * (interval+1)/discretization)


        # Objective
        prob += lpSum(ALPHA*racks_in_use_vars) - lpSum(lambda_multipliers)

        # Constrain interval identifiers to be indicator variables by ensuring that they sum to 1
        for slice in slice_qos:
            prob += lpSum(interval_identifiers[slice]) == 1, "slice-%s-indicator-constraint" % (slice)
            prob += lpSum(lambdas[slice]) == 1, "slice-%s-lambdas-constraint" % (slice)
            prob += lpSum(lambda_interpolators[slice]) == slice_weights[slice_name]*slice_qos[slice_name], 'constrain_lambdas_linear_interpolant-%s' % (slice)

            for interval in range(discretization):
                prob += interval_identifiers[slice][interval] <= lambdas[slice][interval] + lambdas[slice][interval+1], 'interval_identifier_interpolate_lambdas_%s_%d' % (slice, interval)

        return
    else:
        raise Exception("Objective function: %s not supported..." % (obj))

#RemoteRadioHead, PhyProcessor, are they connected
def set_remote_radio_head_variables(network):
    num_RRHs = 0
    for data_centre in network.data_centres:
        num_RRHs += len(data_centre.remote_radio_heads)

    print("Setting Remote Radio Head variables. %d RRHs" % (num_RRHs))
    d=[]
    for data_centre in network.data_centres:
        if data_centre.type == 'CORE':
            continue
        for rack in data_centre.racks:
            for phy_processor in rack.phy_processors:
                for RRH in data_centre.remote_radio_heads:
                    var = {'DataCentre': data_centre.name,
                           'Rack': rack.name,
                              'PHYProcessor': phy_processor.name,
                              'RRH':RRH.name,
                              'ConnectVariable':LpVariable("connect-%s-%s-%s-%s" % (data_centre.name, rack.name, phy_processor.name, RRH.name),
                                                           lowBound=0,
                                                           upBound=1,
                                                           cat='Binary')}
                    d.append(var)

    RRH_Phyprocesor_connections = pd.DataFrame(d)

    return RRH_Phyprocesor_connections

def set_delay_variables(network, slices):
    # Delay Variables


    # Inter Server Delay
    dd = []
    dr = []
    ds = []

    for slice in slices:
        slice_name = slice.name
        for connected_function_pair in slice.get_connected_functions():
            fl = connected_function_pair[0]
            fr = connected_function_pair[1]
            for data_centre in network.data_centres:

                dd.append({'DataCentre': data_centre.name,
                          'Slice': slice_name,
                          'Function1': fl.name,
                          'Function2': fr.name,
                          'InterDataCentreDelayVariable': LpVariable("interserverdelay-%s-%s-%s-%s" % (
                          data_centre.name, slice_name, fl.name, fr.name),
                                                                 lowBound=0,
                                                                 upBound=1,
                                                                 cat='Binary')})

                for rack in data_centre.racks:

                    dr.append({'DataCentre': data_centre.name,
                              'Rack': rack.name,
                              'Slice': slice_name,
                              'Function1': fl.name,
                              'Function2': fr.name,
                              'InterRackDelayVariable': LpVariable("interrackdelay-%s-%s-%s-%s-%s" % (
                              data_centre.name, rack.name, slice_name, fl.name, fr.name),
                                                                     lowBound=0,
                                                                     upBound=1,
                                                                     cat='Binary')})

                    for server in rack.servers:


                        ds.append({'DataCentre':data_centre.name,
                              'Rack':rack.name,
                              'Server':server.name,
                              'Slice':slice_name,
                              'Function1': fl.name,
                              'Function2': fr.name,
                              'InterServerDelayVariable':LpVariable("interserverdelay-%s-%s-%s-%s-%s-%s" % (data_centre.name, rack.name, server.name,slice_name,fl.name,fr.name),
                                                               lowBound=0,
                                                               upBound=1,
                                                               cat='Binary')})

    inter_server_delays = pd.DataFrame(ds)
    inter_rack_delays = pd.DataFrame(dr)
    inter_datacentre_delays = pd.DataFrame(dd)

    # Max Function to function Delay
    d = []
    for slice in slices:
        slice_name = slice.name
        for connected_functions in slice.get_connected_functions():
            fl = connected_functions[0]
            fr = connected_functions[1]
            d.append({'Slice': slice,
                      'Function1': fl.name,
                      'Function2': fr.name,
                      'MaxFunctionDelayVariable':LpVariable("max-function-delay-%s-%s-%s" % (slice_name,fl.name,fr.name),
                                                            lowBound=0,
                                                            upBound=1,
                                                            cat='Binary')})

    max_function_to_function_delays = pd.DataFrame(d)

    # Max Slice Delays
    for slice in slices:
        slice_name = slice.name
        d.append({'Slice':slice_name,
                  'SliceVariable':LpVariable("slice-delay-%s" % (slice_name),
                                             lowBound=0,
                                             upBound=1,
                                             cat='Binary')})

    max_slice_delay = pd.DataFrame(d)

    return inter_server_delays, inter_rack_delays, inter_datacentre_delays, max_function_to_function_delays, max_slice_delay

def set_qos_proportionality_constraints(prob, slice_qos):

    ''' Optional constraints whether to equate quality of service across slices'''
    slice_names = list(slice_qos.keys())
    for i in range(len(slice_names)-1):
        slice_left = slice_names[i]
        slice_right = slice_names[i+1]
        prob += slice_qos[slice_left] == slice_qos[slice_right], "qos_%s_equals_%s" % (slice_left, slice_right)

def set_remote_radio_head_phy_processor_constraints(prob, network, RRH_Phyprocesor_connections):
    print('RadioHead and PhyProcessor Constraints')
    num_constraints = 0

    # Connecting PhyProcessors and RemoteRadioHeads
    for data_centre in network.data_centres:
        if data_centre.type == 'CORE':
            continue
        for RRH in data_centre.remote_radio_heads:
            num_connections_for_RRH = lpSum(RRH_Phyprocesor_connections[(RRH_Phyprocesor_connections['RRH']==RRH.name)&
                                                                        (RRH_Phyprocesor_connections['DataCentre']==data_centre.name)]['ConnectVariable'])
            constraint_name = "Datacentre_%s_RRH_%s_connect_to_1_PHY" % (data_centre.name, RRH.name)
            prob += num_connections_for_RRH == 1, constraint_name
            num_constraints += 1

    print("Added %d RemoteRadioHead / PhyProcessor constraints" % (num_constraints))

def setting_rack_variables_to_turn_on_with_contained_servers(prob, network, resource_variables, servers_in_use, racks_in_use):
    # Aligning resource and indicator variables. When use is 0, so should resource variable
    print('Setting rack variable to turn on with contained servers...')
    MAX_CPU = 0
    num_constraints = 0
    for data_centre in tqdm(network.data_centres):
        for rack in data_centre.racks:
            for server in rack.servers:# + rack.phy_processors:
                MAX_CPU = max(MAX_CPU, server.resources['CPU'])

    if (MAX_CPU <= 0):
        raise("Exception: Maximimum CPU of any resource is not above 0!")

    for data_centre in tqdm(network.data_centres):
        for rack in data_centre.racks:
            for server in rack.servers:# + rack.phy_processors:#range(NUM_SERVERS_PER_RACK * NUM_RACKS_PER_DATACENTRE * NUM_DATACENTRES):
                reference_resource = 'CPU'

                resources_on_server = resource_variables[(resource_variables['DataCentre'] == data_centre.name) &
                                                         (resource_variables['Rack'] == rack.name) &
                                                         (resource_variables['Server'] == server.name) &
                                                         (resource_variables['Resource'] == reference_resource)]['ResourceVariable']
                boolean_server_in_use = servers_in_use[(servers_in_use['DataCentre'] == data_centre.name) &
                                                        (servers_in_use['Rack'] == rack.name) &
                                                        (servers_in_use['Server'] == server.name)]['Variable']

                prob += lpSum(resources_on_server) <= boolean_server_in_use * MAX_CPU, "align_continuous_and_discrete_resource_variables_datacentre_%s_rack_%s_server_%s" % (data_centre.name, rack.name, server.name)
                num_constraints+=1

    print("Adding rack constraints ensuring rack is in use if at least one of its servers are in use...")

    # Make sure that a rack boolean variable is turned on if one of it's servers are.
    # Note that this constraint doesn't ensure that rack is not on when all server are not, but our objective function wants to minimize this, so optimum would enforce that

    for data_centre in tqdm(network.data_centres):
        for rack in data_centre.racks:
            boolean_rack_in_use = racks_in_use[(racks_in_use['DataCentre'] == data_centre.name) &
                                               (racks_in_use['Rack'] == rack.name)]['Variable']
            #print(boolean_rack_in_use)

            for server in rack.servers:# + rack.phy_processors:
                boolean_server_in_use = servers_in_use[(servers_in_use['DataCentre'] == data_centre.name) &
                                                       (servers_in_use['Rack'] == rack.name) &
                                                       (servers_in_use['Server'] == server.name)]['Variable']
                #print(boolean_server_in_use)

                prob += lpSum(boolean_rack_in_use) >= lpSum(boolean_server_in_use), "datacentre_%s_rack_%s_in_use_if_server_%s_in_use" % (data_centre.name, rack.name, server.name)
                num_constraints += 1
    print("Added %d Server / rack interaction constraints" % (num_constraints))

def set_resource_proportionality_constraints(prob, network, slices, resource_variables):

    print('Resource Proportionality Constraints')
    num_constraints=0
    start_time = time.time()

    # Resource Proportionality
    reference_resource = 'CPU'

    # Transitivity argument to reduce constraints here
    for data_centre in tqdm(network.data_centres):
        data_centre_df = resource_variables[(resource_variables['DataCentre'] == data_centre.name)]
        for rack in data_centre.racks:
            rack_df = data_centre_df[(data_centre_df['Rack'] == rack.name)]
            for server in rack.servers:# + rack.phy_processors:
                server_df = rack_df[(rack_df['Server'] == server.name)]
                for slice in slices:
                    slice_name = slice.name
                    slice_df = server_df[(server_df['Slice'] == slice_name) ]

                    for function in slice.virtual_functions:
                        if not function.data_centre_type == data_centre.type:
                            continue
                        #print("CPU_resource_usage_%s_%s_%s_%s_%s" % (data_centre.name, rack.name, server.name, slice_name, function.name))
                        q = LpVariable("CPU_used-%s-%s-%s-%s-%s" % (data_centre.name, rack.name, server.name,slice_name,function.name), lowBound=0, upBound=None, cat='Continuous')

                        reference_resource_variable = slice_df[(slice_df['Function'] == function.name) &
                                                               (slice_df['Resource'] == reference_resource)]['ResourceVariable']

                        function_df = slice_df[(slice_df['Function']==function.name)]
                        prob += q == reference_resource_variable, "CPU_resource_usage_reference_%s_%s_%s_%s_%s" % (data_centre.name, rack.name, server.name, slice_name, function.name)
                        for resource in function.resource_names:
                            if resource == reference_resource:
                                # No self-referential proportionality
                                continue
                            # Test that resource type is in function
                            if function.get_quantity_per_rate(resource) == 0.0:
                                # No resource required for this type
                                continue
                            server_slice_function_resource_variable = function_df[(function_df['Resource'] == resource)]['ResourceVariable']
                            #print(slice.name, function.name)
                            traffic_proportion = function.get_quantity_per_rate(reference_resource)/ function.get_quantity_per_rate(resource)#slice.get_traffic_proportion(function.name, resource, reference_resource='CPU') # TODO: int??
                            prob += q == server_slice_function_resource_variable * traffic_proportion, "Resource_proportionality_%s_%s_%s_%s_%s_%s" % (data_centre.name,
                                                                                                                                                       rack.name,
                                                                                                                                                       server.name,
                                                                                                                                                       slice_name,
                                                                                                                                                       function.name,
                                                                                                                                                       resource)
                            num_constraints += 1

    print("Added %d resource proportionality constraints" % (num_constraints))
    end_time = time.time()
    print("time building constraints: %f" % (end_time - start_time))

def set_lower_bound_function_resource_per_server(prob, network, slices, resource_variables, min_cpu_function_per_server):
    print('Lower bounding function resources per server Constraints')
    start_time = time.time()
    num_constraints=0
    # Minimum Resource Per Function. Resources on a server must be 0 or > greater than some minimum amount
    large_constant = 10000
    reference_resource = 'CPU'

    for data_centre in tqdm(network.data_centres):
        data_centre_df = resource_variables[(resource_variables['DataCentre'] == data_centre.name)]
        for rack in data_centre.racks:
            rack_df = data_centre_df[(data_centre_df['Rack'] == rack.name)]
            for server in rack.servers:# + rack.phy_processors:
                server_df = rack_df[(rack_df['Server'] == server.name)]
                for slice in slices:
                    slice_name = slice.name
                    slice_df = server_df[(server_df['Slice'] == slice_name) ]
                    for function in slice.virtual_functions:
                        if not function.data_centre_type == data_centre.type:
                            continue
                        q = LpVariable("zero-resource-indicator-%s-%s-%s-%s-%s" % (data_centre.name,
                                                                                   rack.name,
                                                                                   server.name,
                                                                                   slice_name,
                                                                                   function.name), lowBound=0, upBound=1, cat='Binary')

                        var = slice_df[(slice_df['Function'] == function.name) &
                                                 (slice_df['Resource'] == reference_resource)]['ResourceVariable']
                        prob += lpSum(var) + large_constant*q >= min_cpu_function_per_server, "lower_bound_slice_%s_function_%s_datacentre_%s_rack_%s_server_%s" % (slice_name,
                                                                                                                                                                    function.name,
                                                                                                                                                                    data_centre.name,
                                                                                                                                                                    rack.name,
                                                                                                                                                                    server.name)
                        # If q is 1, must be 0, otherwise must be greater than MIN_CPU_FUNCTION_PER_SERVER
                        prob += lpSum(var) <= 0 + large_constant * (1-q), "activate_lower_bound_when_resource_non_zero_%s_%s_%s_%s_%s" % (slice_name,
                                                                                                                                          function.name,
                                                                                                                                          data_centre.name,
                                                                                                                                          rack.name,
                                                                                                                                          server.name)
                        num_constraints += 1

    end_time = time.time()
    print("Added %d lower bound function per server constraints in %f sec" % (num_constraints, end_time-start_time))

def set_server_capacity_constraints(prob, network, resource_variables):

    print('Adding Server Capacity Constraints...')
    start_time = time.time()
    num_constraints=0
    # Assigned resources must be constrained by infrastructure capacity
    for data_centre in tqdm(network.data_centres):
        data_centre_df = resource_variables[(resource_variables['DataCentre'] == data_centre.name)]
        for rack in data_centre.racks:
            rack_df = data_centre_df[(data_centre_df['Rack'] ==rack.name)]
            for server in rack.servers:# + rack.phy_processors:
                server_df = rack_df[(rack_df['Server'] == server.name)]
                for resource_name in server.resources:
                    resources_on_server = lpSum(server_df[(server_df['Resource'] == resource_name)]['ResourceVariable'])
                    server_capacity = server.resources[resource_name]
                    prob += resources_on_server <= server_capacity, "server_%s_%s_%s_resource_%s_capacity" % (data_centre.name, rack.name, server.name, resource_name)
                    num_constraints += 1

    end_time = time.time()
    print("Added %d server capacity constraints in %f sec" % (num_constraints, end_time-start_time))

def set_slice_qos_minimum_requirements(prob, slices, qos_variables, min_qos):
    # QOS must not drop below some constant
    for slice in slices:
        slice_name = slice.name
        prob += qos_variables[slice_name] >= min_qos, "slice_%s_qos_min" % (slice.name)

def set_traffic_demand_constraints(prob, slices, traffic, resource_variables, qos_variables):
    print('Traffic Demands Constraints')
    num_constraints=0
    # Resource must exceed traffic demands
    # TODO: sort out Traffic vs. bitrate per slice
    for slice in slices:
        slice_name = slice.name
        for function in slice.virtual_functions:
            for resource in function.resource_names:
                sum_of_function_resources = lpSum(resource_variables[(resource_variables['Resource'] == resource) &
                                                                     (resource_variables['Slice'] == slice_name) &
                                                                     (resource_variables['Function'] == function.name)]['ResourceVariable'])
                # Should be traffic proportion * resources / traffic * traffic
                required_function_resources = function.get_quantity_per_rate(resource) * lpSum(traffic[(traffic['Slice'] == slice_name)]['bitrate']) * qos_variables[slice_name]
                prob += sum_of_function_resources >= required_function_resources, "resource_demand_function_%s_slice_%s_resource_%s" % (function.name, slice_name, resource)
                num_constraints += 1

    print("Added %d traffic Demand constraints" % (num_constraints))

def set_phy_processor_demand_constraints(prob, slices, network, traffic, resource_variables, RRH_Phyprocesor_connections):
    print("PhyProcessor demand constraints...")
    start_time = time.time()
    num_constraints = 0
    for data_centre in tqdm(network.data_centres):
        if data_centre.type == 'CORE':
            continue
        data_centre_df = resource_variables[(resource_variables['DataCentre'] == data_centre.name)]
        for rack in data_centre.racks:
            rack_df = data_centre_df[(data_centre_df['Rack'] == rack.name)]
            for phy_processor in rack.phy_processors:
                phy_processor_df = rack_df[(rack_df['Server'] == phy_processor.name)]

                for resource in phy_processor.resources:
                    phy_processor_function_resources_slice = LpAffineExpression()
                    resource_df = phy_processor_df[(phy_processor_df['Resource'] == resource)]
                    for slice in slices:
                        slice_name = slice.name
                        slice_df = resource_df[(resource_df['Slice'] == slice_name)]

                        required_function_resources_per_phy_processor = LpAffineExpression()
                        Phy_connections = RRH_Phyprocesor_connections[
                            (RRH_Phyprocesor_connections['PHYProcessor'] == phy_processor.name) &
                            (RRH_Phyprocesor_connections['Rack'] == rack.name) &
                            (RRH_Phyprocesor_connections['DataCentre'] == data_centre.name)]

                        slice_traffic = traffic[(traffic['Slice'] == slice_name) & (traffic['DataCentre'] == data_centre.name)]
                        # TODO: Explain how this merge works
                        joined = pd.merge(Phy_connections,slice_traffic)  # pd.concat([a,RRH_TRAFFIC],axis=1,join='outer')#join_axes='RRH')
                        for function in slice.virtual_functions:

                            if function.phy_processor_constrained:
                                phy_processor_function_resources_slice = LpAffineExpression()
                                required_function_resources_per_phy_processor = LpAffineExpression()
                                if resource not in function.resource_names:
                                    # Continue if function doesn't require resource
                                    continue

                                phy_processor_function_resources_slice += lpSum(
                                    slice_df[(slice_df['Function'] == function.name)]['ResourceVariable'])

                                required_function_resources_per_phy_processor += lpSum(joined['ConnectVariable'] * joined['bitrate']) * function.get_quantity_per_rate(resource)

                                prob += phy_processor_function_resources_slice - required_function_resources_per_phy_processor >= 0.0, "phy_processor_demand_datacentre_%s_rack_%s_phy_%s_slice_%s_function_%s_resource_%s" % (data_centre.name, rack.name, phy_processor.name, slice.name, function.name, resource)
                                num_constraints += 1

    end_time = time.time()
    print("Added %d phy processor demand constraints in %f sec" % (num_constraints, end_time-start_time))

def set_delay_constraints(prob, network, slices, resource_variables, inter_server_delays, inter_rack_delays, inter_datacentre_delays, max_function_to_function_delays, max_slice_delay, max_proportion_difference, delay_between_datacentres, delay_between_racks, delay_within_racks):
    # Delay Constraints
    start_time = time.time()
    print('Delay Constraints')
    num_constraints=0

    # Assume traffic proportion are on same scale for all functions in slice (i.e. If function A and B have 1 traffic proportion, then they have equal traffic proportion
    for slice in slices:
        slice_name = slice.name
        slice_df = resource_variables[(resource_variables['Slice'] == slice_name)]

        if not slice.delay_constraint:
            # Ignore slices with no delay constraint
            continue
        print('Slice:%s, delay constraint:%f' % (slice_name, slice.delay_constraint))
        for connected_function_pair in tqdm(slice.get_connected_functions()):
            functionA = connected_function_pair[0]
            functionB = connected_function_pair[1]
            print("Setting delay variables for connected functions: %s, %s" % (functionA.name, functionB.name))

            resource = 'CPU'
            functionA_df = slice_df[(slice_df['Function'] == functionA.name)]
            functionB_df = slice_df[(slice_df['Function'] == functionB.name)]

            for data_centre in network.data_centres:
                if not functionA.data_centre_type == data_centre.type:
                    continue
                data_centre_df_A = functionA_df[(functionA_df['DataCentre'] == data_centre.name)]
                data_centre_df_B = functionB_df[(functionB_df['DataCentre'] == data_centre.name)]
                for rack in data_centre.racks:
                    rack_df_A = data_centre_df_A[(data_centre_df_A['Rack'] == rack.name)]
                    rack_df_B = data_centre_df_B[(data_centre_df_B['Rack'] == rack.name)]
                    for server in rack.servers:# + rack.phy_processors:
                        server_df_A = rack_df_A[(rack_df_A['Server'] == server.name)]
                        server_df_B = rack_df_B[(rack_df_B['Server'] == server.name)]
                        delay_variable = inter_server_delays[(inter_server_delays['DataCentre'] == data_centre.name) &
                                                             (inter_server_delays['Rack'] == rack.name) &
                                                             (inter_server_delays['Server'] == server.name) &
                                                             (inter_server_delays['Slice'] == slice.name) &
                                                             (inter_server_delays['Function1'] == functionA.name) &
                                                             (inter_server_delays['Function2'] == functionB.name)]['InterServerDelayVariable']
                        resource_variable_l = server_df_A[(server_df_A['Resource'] == resource)]['ResourceVariable']
                        resource_variable_r = server_df_B[(server_df_B['Resource'] == resource)]['ResourceVariable']

                        a = LpAffineExpression()
                        a += (resource_variable_r * functionB.get_quantity_per_rate(resource))
                        a -= (resource_variable_l * functionA.get_quantity_per_rate(resource))
                        a += (delay_variable * max_proportion_difference)
                        name = "data_centre_%s_rack_%s_server_%s_slice_%s_func_%s_func_%s_resource_%s_proportion_delay" % (data_centre.name,
                                                                                                                    rack.name,
                                                                                                                    server.name,
                                                                                                                    slice.name,
                                                                                                                    functionA.name,
                                                                                                                    functionB.name,
                                                                                                                    resource)
                        prob += LpConstraint(a, sense=LpConstraintGE, rhs=0, name=name)
                        num_constraints += 1


                    # Racks
                    delay_variable = inter_rack_delays[(inter_rack_delays['DataCentre'] == data_centre.name) &
                                                        (inter_rack_delays['Rack'] == rack.name) &
                                                         (inter_rack_delays['Slice'] == slice.name) &
                                                         (inter_rack_delays['Function1'] == functionA.name) &
                                                         (inter_rack_delays['Function2'] == functionB.name)]['InterRackDelayVariable']
                    resource_variable_l = rack_df_A[(rack_df_A['Resource'] == resource)]['ResourceVariable']
                    resource_variable_r = rack_df_B[(rack_df_B['Resource'] == resource)]['ResourceVariable']

                    a = LpAffineExpression()
                    a += (resource_variable_r * functionB.get_quantity_per_rate(resource))
                    a -= (resource_variable_l * functionA.get_quantity_per_rate(resource))
                    a += (delay_variable * max_proportion_difference)
                    name = "data_centre_%s_rack_%s_slice_%s_func_%s_func_%s_resource_%s_proportion_delay" % (
                    data_centre.name,
                    rack.name,
                    slice.name,
                    functionA.name,
                    functionB.name,
                    resource)
                    prob += LpConstraint(a, sense=LpConstraintGE, rhs=0, name=name)
                    num_constraints += 1



                # DataCentre
                delay_variable = inter_datacentre_delays[(inter_datacentre_delays['DataCentre'] == data_centre.name) &
                                                     (inter_datacentre_delays['Slice'] == slice.name) &
                                                     (inter_datacentre_delays['Function1'] == functionA.name) &
                                                     (inter_datacentre_delays['Function2'] == functionB.name)]['InterDataCentreDelayVariable']
                resource_variable_l = lpSum(data_centre_df_A[(data_centre_df_A['Resource'] == resource)]['ResourceVariable'])
                resource_variable_r = lpSum(data_centre_df_B[(data_centre_df_B['Resource'] == resource)]['ResourceVariable'])

                a = LpAffineExpression()
                a += (resource_variable_r * functionB.get_quantity_per_rate(resource))
                a -= (resource_variable_l * functionA.get_quantity_per_rate(resource))
                a += (delay_variable * max_proportion_difference)
                name = "data_centre_%s_slice_%s_func_%s_func_%s_resource_%s_proportion_delay" % (
                data_centre.name,
                slice.name,
                functionA.name,
                functionB.name,
                resource)
                prob += LpConstraint(a, sense=LpConstraintGE, rhs=0,name=name)
                num_constraints += 1



            # Max delay over servers/racks/datacentres

            slack_variable = LpVariable("%s-%s-%s-slack" % (slice.name,
                                                      functionA.name,
                                                      functionB.name), lowBound=0, upBound=None, cat='Continuous')
            for data_centre in network.data_centres:
                if not functionA.data_centre_type == data_centre.type:
                    continue
                prob += slack_variable >= delay_between_datacentres * lpSum(inter_datacentre_delays[(inter_datacentre_delays['DataCentre'] == data_centre.name) &
                                                                                                   (inter_datacentre_delays['Slice'] == slice.name) &
                                                                                                   (inter_datacentre_delays['Function1'] == functionA.name) &
                                                                                                   (inter_datacentre_delays['Function2'] == functionB.name)]['InterDataCentreDelayVariable'])
                num_constraints += 1
                for rack in data_centre.racks:
                    prob += slack_variable >= delay_between_racks * lpSum(
                        inter_rack_delays[(inter_rack_delays['DataCentre'] == data_centre.name) &
                                          (inter_rack_delays['Rack'] == rack.name) &
                                          (inter_rack_delays['Slice'] == slice.name) &
                                          (inter_rack_delays['Function1'] == functionA.name) &
                                          (inter_rack_delays['Function2'] == functionB.name)]['InterRackDelayVariable'])
                    num_constraints += 1
                    for server in rack.servers:# + rack.phy_processors:
                        prob += slack_variable >= delay_within_racks * lpSum(
                            inter_server_delays[(inter_server_delays['DataCentre'] == data_centre.name) &
                                                (inter_server_delays['Rack'] == rack.name) &
                                                (inter_server_delays['Server'] == server.name) &
                                                (inter_server_delays['Slice'] == slice.name) &
                                                (inter_server_delays['Function1'] == functionA.name) &
                                                (inter_server_delays['Function2'] == functionB.name)]['InterServerDelayVariable'])
                        num_constraints += 1


            prob += lpSum(max_function_to_function_delays[(max_function_to_function_delays['Slice'] == slice.name) &
                                                    (max_function_to_function_delays['Function1'] == functionA.name) &
                                                    (max_function_to_function_delays['Function2'] == functionB.name)]['MaxFunctionDelayVariable']) >= LpAffineExpression(slack_variable), "slice_%s_function_%s_function_%s_max_delay" % (slice_name, functionA.name, functionB.name)
            num_constraints += 1

        slice_delay_variable = LpVariable("slice_delay_variable_%s" % (slice_name), lowBound=0, upBound=None, cat='Continuous')

        # For now, just sum up delays across all connected functions.
        prob += slice_delay_variable >= lpSum(max_function_to_function_delays[max_function_to_function_delays['Slice'] == slice.name]['MaxFunctionDelayVariable'])
        num_constraints += 1


        # Bound delay of each slice
        prob += slice_delay_variable <= slice.delay_constraint, "slice_delay_%s" % (slice_name)
        num_constraints += 1

    end_time = time.time()
    print("Added %d delay constraints in %f sec" % (num_constraints, end_time-start_time))

def solve_problem(prob=None, solver_path="/ubc/cs/research/arrow/software/CPLEX_Studio126/cplex/bin/x86-64_linux/cplex", print_output=False, keep_solver_output_files=True, options=[]):
    # For linux
    if not solver_path:
        cplex_path = "/ubc/cs/research/arrow/software/CPLEX_Studio126/cplex/bin/x86-64_linux/cplex"
                     #/ubc/cs/research/arrow/software/CPLEX_Studio_12_6_2/cplex/bin/x86-64_linux/cplex
        # For OSX
        #cplex_path = "/ubc/cs/research/arrow/software/CPLEX_Studio_12_6_2/cplex/bin/x86-64_osx/cplex"
        #[ "-c \"read 5gAllocation.lp\" \"optimize\" "]
        print_solver_output = False
        solver = CPLEX(path=cplex_path, options=options, keepFiles=True, msg=print_solver_output)#, keepFile=0, mip=1, msg=0, options=[], timelimit=None)
    else:
        solver = CPLEX(path=solver_path, options=options, keepFiles=keep_solver_output_files, msg=print_output)
    # Check if solver is available
    if not solver.available():
       raise Exception('Solver is not available!')

    print("Solving LP...")
    start_time = time.time()
    prob.solve(solver=solver)
    end_time = time.time()

    print("Total solving time: %f" % (end_time - start_time))

    runtime = end_time - start_time
    status = prob.status

    # For printing CPLEX output
    # with open('cplex.log') as f:
    #     for line in f.readlines():
    #         print(line)

    #prob.objective.value()

    return runtime, status

def analyze_solution(rack_variables, qos_variables, prob=None):
    print("Status:", LpStatus[prob.status])

    if LpStatus[prob.status] == "Optimal":

        # Display Objective
        print("Objective: %f" % (value(prob.objective)))
        #exit()

        # QOS check: Check that we have full QOS if not all servers in use
        num_racks = int(len(rack_variables))
        # for rack in rack_variables:
        #     rack = rack.values[0]
        #     print(rack)
        #     print(type(rack))
        num_racks_in_use = int(sum([value(rack.values[0]) for rack in rack_variables]))
        if int(len(qos_variables.keys())) > int(sum([value(qos_variables[slice_name]) for slice_name in qos_variables])):
            print("QOS reduced: Some traffic not handled by network!")
            if num_racks > num_racks_in_use:
                print("Warning: QOS reduced but still resource available!")
        else:
            print("QOS maximized: Slices serve all traffic!")

        # Objective: prob += lpSum(ALPHA*servers_in_use) - lpSum(weighted_qos)
        # Display Rack decision variables

        print("%d/%d racks in use" % (num_racks_in_use, num_racks))
        # for server in servers_in_use:
        #     print("Server in us:%f" % (value(server)))

        # Display Quality of Service decision variables
        for slice_name in qos_variables:
            print("Slice: %s, QOS: %f" % (slice_name, value(qos_variables[slice_name])))


        #value(variable)


# def stacked_barplot(server_df, slices, server_capacity, server_name, add_label=False, axes=None):
#     warnings.warn("Call to deprecated function stacked_barplot")
#     """
#     This function is deprecated. Only works for specific settings!
#     :param server_df:
#     :param slices:
#     :param server_capacity:
#     :param server_name:
#     :param add_label:
#     :param axes:
#     :return:
#     """
#     ind = (1)
#     previous_val = [0.0]
#     slice_palette = itertools.cycle(sns.color_palette())
#
#     for slice in slices:
#         slice_color = next(slice_palette)
#         pal = itertools.cycle(sns.color_palette("hls",len(slice.get_functions())))
#         slice_df = server_df[(server_df["Slice"] == slice.name)]
#         for function in slice.get_functions():
#             val = [slice_df[(slice_df["Function"] == function.name)]['ResourceVariable'].values[0]]
#             label=None
#             if add_label:
#                 label = "%s-%s"%(slice.name, function.name)
#             axes.bar(ind, val, width=1.0, bottom=previous_val, color=next(pal), edgecolor=slice_color, label=label)
#             previous_val = val
#     plt.ylim((0, server_capacity))
#     axes.xaxis.set_ticklabels([])
#     axes.set_title(server_name,fontsize=2.0)
#
#     if add_label:
#         axes.legend(loc='center left', bbox_to_anchor=(1, 2), ncol=3)


# def visualize_solution(network, slices, resource_variables):
#     # Update resource variable to change 'ResourceVaraible' column
#     resource_variables['ResourceVariable'] = resource_variables['ResourceVariable'].apply(lambda x: value(x))
#
#     resource_variables = resource_variables[(resource_variables["Resource"] == "CPU")]
#     num_datacentres = len(network.data_centres)
#     num_racks=0
#     for data_centre in network.data_centres:
#         num_racks = len(data_centre.racks)
#         num_servers = len(data_centre.racks[0].servers)
#     fig, axes = plt.subplots(num_datacentres, num_racks*num_servers, sharex=True, sharey=True)
#
#     functions = resource_variables["Function"].unique()
#
#     # TODO: Build subplot for every DataCentre and Rack
#     axes=axes.ravel()
#     i=0
#     server_df = None
#     server = None
#     for data_centre in network.data_centres:
#         data_centre_df= resource_variables[(resource_variables["DataCentre"] == data_centre.name)]
#         for rack in data_centre.racks:
#             rack_df = data_centre_df[(data_centre_df["Rack"] == rack.name)]
#             for server in rack.servers:
#                 # Create subplot within subplot
#
#                 server_df = rack_df[(rack_df["Server"] == server.name)]
#                 #pal = itertools.cycle(sns.color_palette())
#
#                 # Create Mapping from function to resource
#
#                 # Plot stacked barplot of function resource in order. Bound on y axis should be server capacity
#
#                 server_name = "%s_%s_%s" % (data_centre.name, rack.name, server.name)
#                 stacked_barplot(server_df, slices, server.resources['CPU'], server_name, axes=axes[i])
#
#                 # g = sns.FacetGrid(rack_df,
#                 #                   col="Server",
#                 #                   hue="Function",
#                 #                   col_wrap=5,
#                 #                   palette=pal,
#                 #                   ax=axes[i]
#                 #                   )
#                 i+=1
#
#                 # Stacked bar plot
#                 # p1 = plt.bar(ind, menMeans, width, yerr=menStd)
#                 # p2 = plt.bar(ind, womenMeans, width,
#                 #              bottom=menMeans, yerr=womenStd)
#
#     stacked_barplot(server_df, slices, server.resources['CPU'], "",add_label=True, axes=axes[i-1])
#     plt.savefig('./test_nested.png', bbox_inches='tight', dpi=100)
#     return

def encode(allocation_problem: AllocationProblem,name: str = 'example'):
    """
    :param name : Identifier of problem for writing to file
    :param allocation_problem: CRAN allocation problem to be converted into JSON
    :return: LpProblem
    """

    slices = allocation_problem.slices
    traffic = allocation_problem.traffic
    network = allocation_problem.network
    options = allocation_problem.options

    # TODO: Check set of RRHs in network and traffic are identical
    # for data_centre in network.data_centres:
    #     if note set(data_centre.remote_radio_heads).equals(set(traffic))

    objective_function = options.objective_function
    min_cpu_function_per_server = options.min_cpu_per_server
    min_qos = options.min_qos

    max_cpu_difference = SliceGroup(slices=slices).max_difference(resource='CPU')
    max_cpu = network.max_server(resource='CPU')
    #print(max_cpu, min_cpu_function_per_server, max_cpu_difference)
    max_proportion_difference = (max_cpu / min_cpu_function_per_server) * max_cpu_difference
    #print('Max proportional difference in function resources within slice: %f' % max_proportion_difference)

    prob = LpProblem("5G_Cloud_RAN_Virtual_Function_Allocation_%s" % (name), LpMinimize)

    resource_variables, internet_gateway_variables = set_resource_variables(network, slices)

    servers_in_use, racks_in_use = set_server_rack_variables(network)

    slice_qos, slice_weights = set_slice_qos_vars(slices=slices)

    set_objective_function(prob, racks_in_use, slice_qos, slice_weights, obj=objective_function)

    RRH_Phyprocessor_connections = set_remote_radio_head_variables(network)

    inter_server_delays, inter_rack_delays, inter_datacentre_delays, max_function_to_function_delays, max_slice_delay = set_delay_variables(network, slices)

    '''
    **************Constraints***********

    '''

    print('Setting Constraints...')

    # This is commented: This ensure qos is equal across all slices
    #set_qos_proportionality_constraints(prob, slice_qos)

    set_remote_radio_head_phy_processor_constraints(prob, network, RRH_Phyprocessor_connections)

    setting_rack_variables_to_turn_on_with_contained_servers(prob, network, resource_variables, servers_in_use, racks_in_use)

    set_resource_proportionality_constraints(prob, network, slices, resource_variables)

    set_lower_bound_function_resource_per_server(prob, network, slices, resource_variables, min_cpu_function_per_server)

    set_server_capacity_constraints(prob, network, resource_variables)

    set_slice_qos_minimum_requirements(prob, slices, slice_qos, min_qos)

    set_traffic_demand_constraints(prob, slices, traffic.traffic, resource_variables, slice_qos)

    set_phy_processor_demand_constraints(prob, slices, network, traffic.traffic, resource_variables, RRH_Phyprocessor_connections)

    # TODO: We take worst case delays. We should refine this if server-server, rack-rack, or datacentre-datacentre delays vary across network
    delay_between_datacentres = network.max_data_centre_delay()
    delay_between_racks = network.max_rack_delay()
    delay_within_racks = network.max_server_delay()

    set_delay_constraints(prob, network, slices, resource_variables, inter_server_delays, inter_rack_delays, inter_datacentre_delays, max_function_to_function_delays, max_slice_delay, max_proportion_difference, delay_between_datacentres, delay_between_racks, delay_within_racks)

    rack_variables = []
    for data_centre in network.data_centres:
        for rack in data_centre.racks:
            rack_var = racks_in_use[(racks_in_use['DataCentre'] == data_centre.name) &
                                    (racks_in_use['Rack'] == rack.name)]['Variable']
            rack_variables.append(rack_var)

    return prob, rack_variables, slice_qos, resource_variables, RRH_Phyprocessor_connections, racks_in_use

def encode_topology_constraints(allocation_problem):
    """
     :param name : Identifier of problem for writing to file
     :param allocation_problem: CRAN allocation problem to be converted into JSON
     :return: LpProblem
     """

    # TODO: Make sure that PhyProcessors are subset of Servers in AllocationProblem

    slices = allocation_problem.slices  # get_slices()
    #traffic = allocation_problem.traffic  # get_traffic()
    network = allocation_problem.network  # get_network()
    options = allocation_problem.options  # get_options() # Some options for different encoding alternatives and constraints on the problem. Constraints not necessarily set in stone

    # TODO: Check set of RRHs in network and traffic are identical
    # for data_centre in network.data_centres:
    #     if note set(data_centre.remote_radio_heads).equals(set(traffic))

    objective_function = options.objective_function
    min_cpu_function_per_server = options.min_cpu_per_server
    min_qos = options.min_qos

    max_cpu_difference = SliceGroup(slices=slices).max_difference(resource='CPU')
    max_cpu = network.max_server(resource='CPU')
    max_proportion_difference = (max_cpu / min_cpu_function_per_server) * max_cpu_difference
    #print('Max proportional difference in function resources within slice: %f' % max_proportion_difference)

    prob = LpProblem("5G_Cloud_RAN_Virtual_Function_Allocation_%s" % ('exp'), LpMinimize)

    resource_variables, internet_gateway_variables = set_resource_variables(network, slices)

    servers_in_use, racks_in_use = set_server_rack_variables(network)

    slice_qos, slice_weights = set_slice_qos_vars(slices=slices)

    set_objective_function(prob, racks_in_use, slice_qos, slice_weights, obj=objective_function)

    RRH_Phyprocessor_connections = set_remote_radio_head_variables(network)

    inter_server_delays, inter_rack_delays, inter_datacentre_delays, max_function_to_function_delays, max_slice_delay = set_delay_variables(
        network, slices)

    '''
    **************Constraints***********

    '''

    print('Setting Constraints...')

    # This is commented: This ensure qos is equal across all slices
    # set_qos_proportionality_constraints(prob, slice_qos)

    set_remote_radio_head_phy_processor_constraints(prob, network, RRH_Phyprocessor_connections)

    setting_rack_variables_to_turn_on_with_contained_servers(prob, network, resource_variables, servers_in_use,
                                                             racks_in_use)

    set_resource_proportionality_constraints(prob, network, slices, resource_variables)

    set_lower_bound_function_resource_per_server(prob, network, slices, resource_variables, min_cpu_function_per_server)
    #
    set_server_capacity_constraints(prob, network, resource_variables)
    #
    set_slice_qos_minimum_requirements(prob, slices, slice_qos, min_qos)

    delay_between_datacentres = network.max_data_centre_delay()
    delay_between_racks = network.max_rack_delay()
    delay_within_racks = network.max_server_delay()


    set_delay_constraints(prob, network, slices, resource_variables, inter_server_delays, inter_rack_delays,
                          inter_datacentre_delays, max_function_to_function_delays, max_slice_delay,
                          max_proportion_difference, delay_between_datacentres, delay_between_racks, delay_within_racks)

    rack_variables = []
    for data_centre in network.data_centres:
        for rack in data_centre.racks:
            rack_var = racks_in_use[(racks_in_use['DataCentre'] == data_centre.name) &
                                    (racks_in_use['Rack'] == rack.name)]['Variable']
            rack_variables.append(rack_var)

    return prob, slices, network, RRH_Phyprocessor_connections, slice_qos, resource_variables

def add_traffic_dependent_constraints(prob=None, traffic=None, slices=None, network=None, resource_variables=None, RRH_Phyprocessor_connections=None, slice_qos=None):

    set_phy_processor_demand_constraints(prob, slices, network, traffic, resource_variables,
                                         RRH_Phyprocessor_connections)

    set_traffic_demand_constraints(prob, slices, traffic, resource_variables, slice_qos)

def save_mip_encoding(prob=None, filename='./'):

    prob.writeLP(filename)

def create_problem_from_traffic(problem=None,
                                root_problem=None,
                                slices=None,
                                network=None,
                                RRH_Phyprocessor_connections=None,
                                slice_qos=None,
                                resource_variables=None,
                                directory=None,
                                instance_counter=None,
                                output=None):
    print("Instance counter: %d" % instance_counter)
    traffic_problem = root_problem.deepcopy()

    json_problem = json.loads(json.load(open(problem, 'r')))
    CloudRAN_Allocation_Problem = AllocationProblem.from_json(json_problem)
    traffic = CloudRAN_Allocation_Problem.traffic.traffic
    add_traffic_dependent_constraints(prob=traffic_problem,
                                      traffic=traffic,
                                      slices=slices,
                                      network=network,
                                      resource_variables=resource_variables,
                                      RRH_Phyprocessor_connections=RRH_Phyprocessor_connections,
                                      slice_qos=slice_qos)
    filename = os.path.join(directory, '%d.lp' % instance_counter)
    save_mip_encoding(prob=traffic_problem, filename=filename)
    output.put(filename)

def encode_distribution(problems=None,directory='./'):# list(AllocationProblem), directory: str):

    json_problem = json.loads(json.load(open(problems[0], 'r')))
    CloudRAN_Allocation_Problem = AllocationProblem.from_json(json_problem)
    root_problem, slices, network, RRH_Phyprocessor_connections, slice_qos, resource_variables = encode_topology_constraints(CloudRAN_Allocation_Problem)
    num_problems = len(problems)

    num_cores = mp.cpu_count()
    print('Num cores: %f' % (num_cores))
    start_time = time.time()

    output = mp.Queue()
    processes = [mp.Process(target=create_problem_from_traffic, args=(problems[i],
                                                                      root_problem,
                                slices,
                                network,
                                RRH_Phyprocessor_connections,
                                slice_qos,
                                resource_variables,
                                directory,
                                i,
                                output)) for i in range(num_problems)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    filenames = [output.get() for p in processes]

    end_time = time.time()
    print("Total time to get traffic problems %f sec" % (end_time - start_time))

    return filenames


if __name__ == '__main__':

    print("Parsing Arguments...")

    parser = argparse.ArgumentParser(description='Arguments for MIP encoding for 5G resource allocation problems')

    ''' Main options '''
    parser.add_argument('--problem', type=str, help='Cloud RAN allocation problem in JSON format')
    parser.add_argument('--name', type=str, default='test', help='Name of problem')
    parser.add_argument('--seed',type=float, default=0.0, help='random seed for replication')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to have verbose logging')

    ''' Output options '''
    parser.add_argument('--write_lp', type=bool, default=True, help='Whether to write encoded problem to .lp file')
    parser.add_argument('--output_directory', type=str, default="./instances", help="Directory to save JSON generated files")

    ''' Solving options '''
    parser.add_argument('--solve', type=bool, default=False, help='Whether to solve problem after producing encoding')
    parser.add_argument('--solver_path', type=str, default="/ubc/cs/research/arrow/software/CPLEX_Studio126/cplex/bin/x86-64_linux/cplex", help='Path to CPLEX executable. Currently does not support alternative solvers.')
    parser.add_argument('--print_output', type=bool, default=False, help='If solver turned on, whether to print solver output to stdout')
    parser.add_argument('--keep_solver_output_files', type=bool, default=False, help='If solve turned on, whether to keep CPLEX solver output files')

    args = parser.parse_args()

    random.seed(args.seed)

    json_problem = json.loads(json.load(open(args.problem, 'r')))
    CloudRAN_Allocation_Problem = AllocationProblem.from_json(json_problem)

    encoding_start_time = time.time()
    prob, rack_variables, qos_variables, resource_variables, RRH_Phyprocessor_connections, racks_in_use = encode(allocation_problem=CloudRAN_Allocation_Problem)
    encoding_end_time = time.time()
    print("Total encoding time: %f" % (encoding_end_time - encoding_start_time))

    if args.write_lp:
        if args.verbose:
            # Print constraints and variables for mip problem
            for constraint in prob.constraints:
                print(constraint)
            for variable in prob._variables:
                print(variable)
                print(variable.name)
        output_filename = args.problem.replace("json", "lp")
        print('Writing the .lp to %s' % (output_filename))

        prob.writeLP(output_filename)

    if args.solve:
        runtime, status = solve_problem(prob=prob,
                      solver_path=args.solver_path,
                      print_output=args.print_output,
                      keep_solver_output_files=args.keep_solver_output_files)

        if analyze_solution:
            analyze_solution(rack_variables, qos_variables, prob=prob)

        # TODO: plotting is broken!
        # plot_solution = False
        # if plot_solution and status != LpStatusInfeasible:
        #     visualize_solution(CloudRAN_Allocation_Problem.network, CloudRAN_Allocation_Problem.slices, resource_variables)

