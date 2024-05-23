import math


class Graph:

    def __init__(self):
        self.graph = {}

    def create_graph(self, sentences):
        temp_graph = {}
        for sentence in sentences:
            sentence = sentence.split()
            for word_idx in range(len(sentence) - 1):

                current_word = sentence[word_idx]
                next_word = sentence[word_idx + 1]

                if (
                    current_word not in temp_graph
                ):  # Make sure all words are a key in the dictionary
                    temp_graph[current_word] = {}
                if next_word not in temp_graph:
                    temp_graph[next_word] = {}

                if current_word in temp_graph:
                    if sentence[word_idx + 1] in temp_graph[current_word]:
                        temp_graph[current_word][next_word] += 1
                    else:
                        temp_graph[current_word][next_word] = 1
                else:
                    temp_graph[current_word] = {next_word: 1}
        self.compute_undirect_graph(temp_graph)

    def compute_undirect_graph(self, temp_graph):

        # Como lidar com os nos que estao vazios no dicionario?
        visited_nodes = []
        for node in temp_graph:
            connected_nodes = temp_graph[node]
            for connected_node in connected_nodes:
                if (node, connected_node) not in visited_nodes:
                    if temp_graph[
                        connected_node
                    ]:  # Se o no conectado ao no principal nao eh vazio
                        if node in temp_graph[connected_node]:
                            edge_value = (
                                temp_graph[node][connected_node]
                                + temp_graph[connected_node][node]
                            )
                            # print(edge_value)
                            temp_graph[node][connected_node] = edge_value
                            temp_graph[connected_node][node] = edge_value
                        else:
                            edge_value = temp_graph[node][connected_node]
                            # print(edge_value)
                            temp_graph[node][connected_node] = edge_value
                            temp_graph[connected_node][node] = edge_value
                    else:  # Is leaf word: im a leaf
                        # print(temp_graph[node][connected_node])
                        temp_graph[connected_node][node] = temp_graph[node][
                            connected_node
                        ]
                    visited_nodes.append((node, connected_node))
                    visited_nodes.append((connected_node, node))
        self.graph = temp_graph
        # print(self.graph)
        self.compute_edges_prob()

    def compute_edges_prob(self):

        nodes_weight = {}
        for out_node in self.graph:  # reference node
            out_node_weight = 0
            for in_node in self.graph[
                out_node
            ]:  # all other nodes inside the reference node
                out_node_weight = out_node_weight + self.graph[out_node][in_node]
            nodes_weight[out_node] = out_node_weight

        for out_node in self.graph:
            for in_node in self.graph[out_node]:
                edge_weight = self.graph[out_node][in_node]
                edge_prob = edge_weight / (
                    nodes_weight[out_node] + nodes_weight[in_node] - edge_weight
                )
                edge_log_prob = -math.log(edge_prob)
                self.graph[out_node][in_node] = edge_log_prob
        # print(self.graph)
