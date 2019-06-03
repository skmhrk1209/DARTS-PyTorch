import torch
from torch import nn
from ops import *
import networkx as nx
import matplotlib.pyplot as plt


class DARTS(nn.Module):

    def __init__(self, operations, num_nodes, num_cells, reduction_cells, num_channels):

        super().__init__()

        self.operations = operations
        self.num_nodes = num_nodes
        self.num_cells = num_cells
        self.reduction_cells = reduction_cells
        self.num_channels = num_channels

        self.build_dag()
        self.build_network()

    def build_dag(self):
        self.dag = nx.DiGraph()
        for parent in range(self.num_nodes):
            for child in range(self.num_nodes):
                if parent < child:
                    self.dag.add_edge(parent, child, operations=self.operations)

    def build_network(self):

        self.network = nn.ModuleDict()

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=self.num_channels * 3,
            stride=1,
            kernel_size=3,
            padding=1,
            preactivation=False
        )

        self.network.cells = nn.ModuleList()
        # NOTE: why multiplier is 3 ?
        in_channels_1st = self.num_channels * 3
        in_channels_2nd = self.num_channels * 3
        out_channels = self.num_channels
        reduction_input = False

        for i in range(self.num_cells):

            cell = nn.ModuleDict()
            reduction_output = False
            if i in self.reduction_cells:
                reduction_output = True
                out_channels <<= 1

            # NOTE: should be factorized reduce ?
            cell.conv_1st = Conv2d(
                in_channels=in_channels_1st,
                out_channels=out_channels,
                stride=2 if reduction_input else 1,
                kernel_size=1,
                padding=0,
            )
            cell.conv_2nd = Conv2d(
                in_channels=in_channels_2nd,
                out_channels=out_channels,
                stride=1,
                kernel_size=1,
                padding=0
            )
            cell.edges = nn.ModuleDict({
                str((parent, child)): nn.ModuleList([
                    operation(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        stride=2 if reduction_output and parent in [0, 1] else 1
                    ) for operation in attribute['operations']
                ])
                for parent, child, attribute in self.dag.edges(data=True)
            })

            self.network.cells.append(cell)
            in_channels_1st = in_channels_2nd
            in_channels_2nd = out_channels * (self.num_nodes - 2)
            reduction_input = reduction_output

        self.architecture = nn.ParameterDict()

        self.architecture.normal = nn.ParameterDict({
            str((parent, child)): nn.Parameter(torch.zeros(len(attribute['operations'])))
            for parent, child, attribute in self.dag.edges(data=True)
        })

        self.architecture.reduction = nn.ParameterDict({
            str((parent, child)): nn.Parameter(torch.zeros(len(attribute['operations'])))
            for parent, child, attribute in self.dag.edges(data=True)
        })

    def forward_cell(self, cell, reduction, child, cell_outputs):
        architecture = self.architecture.reduction if reduction else self.architecture.normal
        if self.dag.predecessors(child):
            if child not in cell_outputs:
                cell_outputs[child] = sum(sum(
                    operation(self.forward_cell(cell, reduction, parent, cell_outputs)) * weight
                    for operation, weight in zip(cell.edges[str((parent, child))], nn.functional.softmax(architecture[str((parent, child))]))
                ) for parent in self.dag.predecessors(child))
        return cell_outputs[child]

    def forward(self, input):
        output = self.network.conv(input)
        outputs = [output, output]
        for i, cell in enumerate(self.network.cells):
            cell_outputs = {0: cell.conv_1st(outputs[-2]), 1: cell.conv_2nd(outputs[-1])}
            self.forward_cell(cell, i in self.reduction_cells, self.num_nodes - 1, cell_outputs)
            outputs.append(torch.cat(list(cell_outputs.values())[2:], dim=1))
        return outputs[-1]
