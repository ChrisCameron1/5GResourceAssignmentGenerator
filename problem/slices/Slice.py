__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import sys
import networkx as nx

from problem.slices.VirtualFunction import VirtualFunction
from problem.network.Resource import Resource


class Slice:

    def __init__(self,
                 function_graph: nx.DiGraph,
                 delay_constraint: float = None,
                 importance: float = 1.0,
                 name: str = None
                 ):

        self.function_graph = function_graph
        self.virtual_functions = function_graph.nodes()
        if delay_constraint:
            self.delay_constraint = delay_constraint if delay_constraint > 0.0 else None  # 0 delay constraint is representative of No delay constraint
        else:
            self.delay_constraint = delay_constraint
        self.importance = importance
        self.name = name

    def get_resource_requirements_rate(self, resource):

        resource_requirements = 0
        for function in self.virtual_functions:
            resource_requirements += function.get_quantity_per_rate(resource)
        return resource_requirements

    def get_functions(self):
        return self.virtual_functions

    def get_connected_functions(self):

        connected_functions = []
        for node in self.function_graph:
            for neighbor in self.function_graph.neighbors(node):
                connected_functions.append((node, neighbor))

        return connected_functions

    def get_paths(self):

        paths = []
        for node in self.function_graph:
            if self.function_graph.out_degree(node) == 0:  # it's a leaf
                paths.append(nx.shortest_path(self.function_graph, self.root, node))
        return None

    def get_max_resource_requirements_rate(self, resource=None):

        max_resources_rate = 0
        for function in self.virtual_functions:

            quantity_per_rate = function.get_quantity_per_rate(resource=resource)
            if quantity_per_rate > max_resources_rate:
                max_resources_rate = quantity_per_rate

        return max_resources_rate

    def get_min_resource_requirements_rate(self, resource=None):

        min_resources_rate = sys.maxsize
        for function in self.virtual_functions:
            quantity_per_rate = function.get_quantity_per_rate(resource=resource)
            if quantity_per_rate < min_resources_rate:
                min_resources_rate = quantity_per_rate

        return min_resources_rate

    def max_difference(self, resource='CPU'):

        return self.get_max_resource_requirements_rate(resource=resource) / self.get_min_resource_requirements_rate(resource=resource)

    def phy_traffic_load(self):

        phy_traffic_load = 0
        for function in self.virtual_functions:
            if function.phy_processor_constrained:
                phy_traffic_load += function.get_quantity_per_rate(resource='CPU')

        return phy_traffic_load


    def to_json(self):
        '''
        self.function_graph = function_graph
        self.virtual_functions = virtual_functions
        self.delay_constraint = delay_constraint
        self.importance = importance
        self.name = name
        '''
        # for function in self.function_graph.nodes():
        #     print(function.to_json())
        #exit()
        json_representation = {
            "function_graph": dict(
                nodes=[[n.to_json(), self.function_graph.node[n]] for n in self.function_graph.nodes()], #self.function_graph.node[n]
                edges=[[u.to_json(), v.to_json(), self.function_graph.edge[u][v]] for u, v in
                       self.function_graph.edges()]), # self.function_graph.edge[u][v]
        # TODO: Probably have to properly implement JSON serilaization for node represnetations
            "delay_constraint": self.delay_constraint if self.delay_constraint else 0.0,
            "name": self.name,
            "importance": self.importance
        }

        return json_representation


    @staticmethod
    def get_example_slice(name: str):

        function_graph = nx.DiGraph()


        default_function_resources = [Resource(name='CPU', quantity_per_load=0.5),
                                      Resource(name='MEM', quantity_per_load=0),
                                      Resource(name='IO', quantity_per_load=0.5)]

        '''
        Begin by defining control-plane functions that are disconnected in the graph. In this preliminary model,
        we assume all control-plane functions have identical resource requirements.
        '''
        amf_function = VirtualFunction(resources=default_function_resources, name='AMF',
                                                       data_centre_type='CORE')
        function_graph.add_node(amf_function)

        SMF_function = VirtualFunction(resources=default_function_resources, name='SMF',
                                                       data_centre_type='CORE')
        function_graph.add_node(SMF_function)

        PCF_function = VirtualFunction(resources=default_function_resources, name='PCF',
                                                       data_centre_type='CORE')
        function_graph.add_node(PCF_function)

        nef_function = VirtualFunction(resources=default_function_resources, name='NEF',
                                                       data_centre_type='CORE')
        function_graph.add_node(nef_function)

        nrf_function = VirtualFunction(resources=default_function_resources, name='NRF',
                                                       data_centre_type='CORE')
        function_graph.add_node(nrf_function)

        ausf_function = VirtualFunction(resources=default_function_resources, name='AUSF',
                                                        data_centre_type='CORE')
        function_graph.add_node(ausf_function)

        NIWF_function = VirtualFunction(resources=default_function_resources, name='N31W5',
                                                        data_centre_type='CORE')
        function_graph.add_node(NIWF_function)

        smsf_function = VirtualFunction(resources=default_function_resources, name='SMSF',
                                                        data_centre_type='CORE')
        function_graph.add_node(smsf_function)

        LMF_function = VirtualFunction(resources=default_function_resources, name='LMF',
                                                       data_centre_type='CORE')
        function_graph.add_node(LMF_function)

        UDR_function = VirtualFunction(resources=default_function_resources, name='UDR',
                                                       data_centre_type='CORE')
        function_graph.add_node(UDR_function)

        if name == 'mMTC_old':
            # Initial model of MMTC slice
            '''
            mmtc-prot: 10X CPU
                mmtc-agg: 2X CPU
                    mmtc-split:5X CPU
            '''
            ordered_functions = []

            mmtc_prot_function_resources = [Resource(name='CPU', quantity_per_load=10)]
            mmtc_prot_function = VirtualFunction(resources=mmtc_prot_function_resources,
                                                                 name='mmtc_prot')
            ordered_functions.append(mmtc_prot_function)

            mmtc_agg_function_resources = [Resource(name='CPU', quantity_per_load=2)]
            mmtc_agg_function = VirtualFunction(resources=mmtc_agg_function_resources, name='mmtc_agg')

            ordered_functions.append(mmtc_agg_function)

            mmtc_split_function_resources = [Resource(name='CPU', quantity_per_load=5)]
            mmtc_split_function = VirtualFunction(resources=mmtc_split_function_resources,
                                                                  name='mmtc_split')
            ordered_functions.append(mmtc_split_function)

            delay_constraint = None

            return Slice(name='mMTC',
                         function_graph=Slice.graph_from_order_functions(ordered_functions),
                         delay_constraint=delay_constraint,
                         importance=1.0)

        elif name == 'embb_old':
            # Initial model of eMBB slice
            '''
            nb : 5x
                0.97: embb-gw 1X
                    embb-content 20X
                0.02 embb-mme 0.5X
                0.01 embb-HSS 0.5X
            '''

            nb_function_resources = [Resource(name='CPU', quantity_per_load=5)]
            nb_function = VirtualFunction(resources=nb_function_resources, name='nb')

            embb_gw_function_resources = [Resource(name='CPU', quantity_per_load=1)]
            embb_gw_function = VirtualFunction(resources=embb_gw_function_resources, name='embb_gw')

            embb_content_function_resources = [Resource(name='CPU', quantity_per_load=20)]
            embb_content_function = VirtualFunction(resources=embb_content_function_resources,
                                                                    name='embb_content')

            embb_mme_function_resources = [Resource(name='CPU', quantity_per_load=0.5)]
            embb_mme_function = VirtualFunction(resources=embb_mme_function_resources, name='embb_mme')

            embb_hss_function_resources = [Resource(name='CPU', quantity_per_load=0.5)]
            embb_hss_function = VirtualFunction(resources=embb_hss_function_resources, name='embb_HSS')

            # Building Function Graph
            function_graph = nx.DiGraph()

            # Add nodes
            function_graph.add_node(nb_function)
            function_graph.add_node(embb_gw_function)
            function_graph.add_node(embb_content_function)
            function_graph.add_node(embb_mme_function)
            function_graph.add_node(embb_hss_function)

            # Add weighted directed edges
            weighted_edges = [(nb_function, embb_gw_function, 0.97),
                              (embb_gw_function, embb_content_function, 1.0),
                              (nb_function, embb_mme_function, 0.02),
                              (nb_function, embb_hss_function, 0.01)]

            function_graph.add_weighted_edges_from(weighted_edges)

            delay_constraint = None

            return Slice(name='CachedVideo',
                         function_graph=function_graph,
                         delay_constraint=delay_constraint,
                         importance=1.0)

        elif name == 'URLLC_old':
            # Initial model of URLLC slice
            urrlc_nb_function_resources = [Resource(name='CPU', quantity_per_load=2.0)]
            urrlc_nb_function = VirtualFunction(resources=urrlc_nb_function_resources, name='urrlc')
            delay_constraint = 30
            return Slice(name='URLLC',
                         function_graph=Slice.graph_from_order_functions([urrlc_nb_function]),
                         delay_constraint=delay_constraint,
                         importance=1.0)


        elif 'MMTC' in name or 'eMBB' in name:
            '''
            Both the MMTC and eMBB slice have the same slice graph
            '''
            mac_function_resources = [Resource(name='CPU', quantity_per_load=5.0),
                                      Resource(name='MEM', quantity_per_load=1.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            mac_function = VirtualFunction(resources=mac_function_resources, name='MAC',
                                                           phy_processor_constrained=True, data_centre_type='RAN')
            function_graph.add_node(mac_function)

            rlc_function_resources = [Resource(name='CPU', quantity_per_load=0.5),
                                      Resource(name='MEM', quantity_per_load=2.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            rlc_function = VirtualFunction(resources=rlc_function_resources, name='RLC',
                                                           data_centre_type='RAN')
            function_graph.add_node(mac_function)

            pdcp_function_resources = [Resource(name='CPU', quantity_per_load=1.0),
                                       Resource(name='MEM', quantity_per_load=4.0),
                                       Resource(name='IO', quantity_per_load=0.5)]
            pdcp_function = VirtualFunction(resources=pdcp_function_resources, name='PDCP',
                                                            data_centre_type='RAN')
            function_graph.add_node(pdcp_function)

            sgw_function_resources = [Resource(name='CPU', quantity_per_load=2.0),
                                      Resource(name='MEM', quantity_per_load=8.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            sgw_function = VirtualFunction(resources=sgw_function_resources, name='SGWVNF',
                                                           data_centre_type='RAN')
            function_graph.add_node(sgw_function)

            pgw_function_resources = [Resource(name='CPU', quantity_per_load=10.0),
                                      Resource(name='MEM', quantity_per_load=10.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            pgw_function = VirtualFunction(resources=pgw_function_resources, name='PGWVNF',
                                                           data_centre_type='RAN')
            function_graph.add_node(pgw_function)

            rrc_function_resources = [Resource(name='CPU', quantity_per_load=5.0),
                                      Resource(name='MEM', quantity_per_load=10.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            rrc_function = VirtualFunction(resources=rrc_function_resources, name='RRCVNF',
                                                           data_centre_type='RAN')
            function_graph.add_node(rrc_function)

            weighted_edges = [(mac_function, rlc_function, 1.0),
                              (rlc_function, pdcp_function, 1.0),
                              (pdcp_function, sgw_function, 0.5),
                              (pdcp_function, rrc_function, 0.5),
                              (sgw_function, pgw_function, 1.0)]

            function_graph.add_weighted_edges_from(weighted_edges)

            if 'delay' in name:
                delay_constraint = 41
            else:
                delay_constraint = None

            return Slice(name=name,
                         function_graph=function_graph,
                         delay_constraint=delay_constraint,
                         importance=1.0)

        elif 'URLLC' in name:

            '''
            In URLLC, we combine many of the functions in eMBB/MMTC together because they must be collocated, substantially shrinking delay constraints
            '''

            mac_rlc_pdcp_function_resources = [Resource(name='CPU', quantity_per_load=6.5),
                                               Resource(name='MEM', quantity_per_load=7.0),
                                               Resource(name='IO', quantity_per_load=1.5)]
            mac_rlc_pdcp_function = VirtualFunction(resources=mac_rlc_pdcp_function_resources,
                                                                    name='MAC-RLC-PDCP',
                                                                    phy_processor_constrained=True,
                                                                    data_centre_type='RAN')
            function_graph.add_node(mac_rlc_pdcp_function)

            sgw_pgw_function_resources = [Resource(name='CPU', quantity_per_load=12.0),
                                          Resource(name='MEM', quantity_per_load=18.0),
                                          Resource(name='IO', quantity_per_load=1.0)]
            sgw_pgw_function = VirtualFunction(resources=sgw_pgw_function_resources, name='SGW-PGW-VNF',
                                                               data_centre_type='RAN')
            function_graph.add_node(sgw_pgw_function)

            rrc_function_resources = [Resource(name='CPU', quantity_per_load=5.0),
                                      Resource(name='MEM', quantity_per_load=10.0),
                                      Resource(name='IO', quantity_per_load=0.5)]
            rrc_function = VirtualFunction(resources=rrc_function_resources, name='RRCVNF',
                                                           data_centre_type='RAN')
            function_graph.add_node(rrc_function)

            weighted_edges = [(mac_rlc_pdcp_function, sgw_pgw_function, 0.5),
                              (mac_rlc_pdcp_function, rrc_function, 0.5)]

            function_graph.add_weighted_edges_from(weighted_edges)

            # TODO: Make parameterizable
            delay_constraint = 20

            return Slice(name=name,
                         function_graph=function_graph,
                         delay_constraint=delay_constraint,
                         importance=5.0)
        else:
            raise Exception('No moodel for slice %s' % (name))

    @staticmethod
    def from_json(obj):

        function_graph = nx.DiGraph()
        virtual_functions = []
        for node in obj['function_graph']['nodes']:
            virtual_function = VirtualFunction.from_json(node[0])
            virtual_functions.append(virtual_function)
            function_graph.add_node(virtual_function)
        for node_l, node_r, weight in obj['function_graph']['edges']:
            if not weight:
                weight = {'weight': 1.0}
            # Need to get the real function created from above
            for function in virtual_functions:
                if function.name == node_l['name']:
                    left_function = function
                if function.name == node_r['name']:
                    right_function = function
            function_graph.add_edge(left_function,
                                    right_function,
                                    weight=float(weight['weight']))

        delay_constraint = float(obj['delay_constraint'])
        name = obj['name']
        importance = obj['importance']

        slice = Slice(function_graph=function_graph,
                      delay_constraint=delay_constraint,
                      name=name,
                      importance=importance)  # traffic_per_unit_of_load=traffic_per_unit_load,)



        return slice

    @staticmethod
    def graph_from_order_functions(functions):

        function_graph = nx.DiGraph()
        for function in functions:
            function_graph.add_node(function)
        for function_index in range(len(functions) - 1):
            function_graph.add_edge(functions[function_index], functions[function_index + 1])

        return function_graph
