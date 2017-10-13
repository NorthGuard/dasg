import codecs
from time import time
from typing import List, Hashable, Iterable, Sized

from dasg.dasg_module.DASG import _DirectedAcyclicSequenceGraphNode, _Check


# TODO: Test whether the checks are inserted correctly!
# TODO: minimize basically checks and edge, whether the parent can instead point to another node
# # TODO: (instead of the current child)

class DirectedAcyclicCompressedSequenceGraph:
    def __init__(self, sequences=None, verbose=False):
        """
        Encodes a list of sequences into a Directed Acyclic Graph.
        Can look up sequences with:
            query in dasg
        Can search for sequences related to query by editdistance:
            dasg.edit_distance_search(query, max_cost=2)
        :param list sequences: List of sequences.
        :param bool verbose: Want prints?
        """
        self._previous_sequence = None
        self._previous_path = None
        self._sequence_count = 0
        self._height = 0
        self._next_id = 0
        self._sequence_type = None

        # Initialize root node
        self._root = self._build_node()

        # List of future checks for merging nodes with identical subtrees.
        self._checks_to_do = []  # type: [_Check]

        # Nodes in DASG.
        self._nodes = {}

        # Initialize if sequences are given
        if sequences is not None:
            self._build(sequences, verbose=verbose)

    def _find_sequence_path(self, sequence, path=None):
        # Start at root
        if path is None:
            path = [("", self._root, 0)]

        # Get last node
        _, node, sequence_depth = path[-1]

        # Largest shared prefix edge
        max_prefix_edge = None

        # Go through children
        for edge, child in node.children.items():
            # Check is edge is iterable (may contain a single element)
            # TODO: This may be irrelevant in later implementations
            if not isinstance(edge, Iterable):
                edge = [edge]  # type: str

            # Shared prefix
            shared_prefix = common_prefix(edge, sequence)

            # Direct node-sequence match
            if shared_prefix == len(edge) == len(sequence):
                path.append((edge, child, sequence_depth + len(edge)))
                return path

            # Direct node-match
            if shared_prefix == len(edge):
                path.append((edge, child, sequence_depth + len(edge)))
                return self._find_sequence_path(sequence=sequence[len(edge):], path=path)

            # Direct edge match
            if shared_prefix == len(sequence):
                path.append((edge, edge, sequence_depth + len(sequence)))
                return path

            # Remember edge for incomplete match
            if max_prefix_edge is None or max_prefix_edge[1] < shared_prefix:
                max_prefix_edge = (edge, shared_prefix, sequence_depth + shared_prefix)

        # If match in incomplete, return marker
        if max_prefix_edge is not None and max_prefix_edge[1] > 0:
            path.append(max_prefix_edge)
        else:
            path.append((sequence, None, sequence_depth))
        return path

    def __contains__(self, sequence):
        next_node = self._root

        while next_node is not None:
            node = next_node
            next_node = None

            # Go through children
            for edge, child in node.children.items():

                # Shared prefix
                shared_prefix = common_prefix(edge, sequence)

                # Direct node-sequence match
                if shared_prefix == len(edge) == len(sequence):
                    return True

                # Direct node-match
                if shared_prefix == len(edge):
                    sequence = sequence[shared_prefix:]
                    next_node = child
                    break

                # Direct edge match
                if shared_prefix == len(sequence):
                    return False

        return False

    def __iter__(self):
        # Search queue
        search_nodes = [(0, edge, child) for edge, child in self._root.children.items()]
        # search_nodes.sort()
        search_nodes.reverse()

        # Current position
        current_sequence = []

        # Go through queue
        while search_nodes:
            node_sequence_length, edge, node = search_nodes.pop()

            # Check length of current sequence
            current_sequence = current_sequence[:node_sequence_length]  # type: list

            # Add new element
            current_sequence.append(edge)

            # Check if node is end of sequence
            if node.is_sequence_end:
                yield self._convert_output(current_sequence)

            # Add children
            additions = [(node_sequence_length + self._edge_len(edge), edge, child)
                         for edge, child in node.children.items()]
            # additions.sort()
            additions.reverse()
            search_nodes.extend(additions)

    @staticmethod
    def _edge_len(edge):
        if isinstance(edge, Sized):
            return len(edge)
        return 1

    @property
    def node_count(self):
        return len(self._nodes)

    @property
    def edge_count(self):
        count = 0
        for node in self._nodes:
            count += len(node.children)
        return count

    @property
    def sequence_count(self):
        return self._sequence_count

    @property
    def height(self):
        return self._height

    def _build_node(self):
        """
        Build a new node with a new ID.
        :return: _DirectedAcyclicSequenceGraphNode
        """
        self._next_id += 1
        return _DirectedAcyclicSequenceGraphNode(self._next_id - 1)

    def _build(self, sequences, verbose=False):
        """
        Build DASG with a list of sequences.
        :param list sequences:
        :param bool verbose: Want prints?
        """
        if verbose:
            print(f"Creating {type(self).__name__}...")

        # Type
        self._sequence_type = type(sequences[0])

        # Check if hashable
        if not isinstance(sequences[0], Hashable):
            sequences = [tuple(val) for val in sequences]

        # Make sure sequences are list-type and sorted
        sequences = sorted(list(set(sequences)))

        # Insert all sequences
        i = 0
        start = time()
        for sequence in sequences:
            i += 1
            self._insert(sequence)

        self._finish()
        if verbose:
            print(
                f"Creation of {type(self).__name__} with {self._sequence_count} sequences "
                f"took {time() - start:.3f}s.\n")

    def _insert(self, sequence):
        # Get sequence path
        # TODO: If sorted, this can be done faster
        # TODO: (the path is simply the ancestor, which can be remembered by height)
        # TODO: Also it MUST be sorted!!!
        path = self._find_sequence_path(sequence)

        if self._previous_path is not None:

            # Ancestor depth is the number of nodes in the path, minus the root
            ancestor_depth = common_path_nodes(path, self._previous_path) - 1
            # path_sequence = self._path2sequence(path)

        else:
            ancestor_depth = 0
            # path_sequence = ""

        # Shared sequence prefix
        prefix = common_prefix(self._previous_sequence, sequence)
        # TODO: Use shared prefix for binary search in path node-list to find common ancestor
        # TODO: Keep path to ancestor and iteratively update it.

        # Reduce by merging nodes with equal subtrees (for the sequences that were handled last)
        self._minimize(ancestor_depth)

        # Re-find sequence after minimization
        path = self._find_sequence_path(sequence)

        # # Remaining sequence
        # sequence_remain = sequence[len(path_sequence):]

        # Ancestor of sequence
        edge, node, _ = path[-1]

        # Ending node is already in graph
        if isinstance(node, _DirectedAcyclicSequenceGraphNode):
            node.is_sequence_end = True

        # Matched inside edge
        elif isinstance(node, str):
            # Get parent and child
            _, parent, handled_length = path[-2]  # type: _DirectedAcyclicSequenceGraphNode, int
            child = parent.children[edge]  # type: _DirectedAcyclicSequenceGraphNode

            # Remove edge
            del parent.children[edge]

            # Edge values
            pn_edge = sequence[handled_length:]
            nc_edge = edge[len(pn_edge):]

            # Make new node and mark as end
            new_node = self._build_node()
            new_node.is_sequence_end = True

            # Link parent to new node and new node to child
            parent.children[pn_edge] = new_node
            new_node.children[nc_edge] = child

            # Add unchecked node
            # self._checks_to_do.append(_Check(parent, pn_edge, new_node))
            self._checks_to_do.append(_Check(new_node, nc_edge, child))

            # # Move to next node
            last_node = new_node

        # Incompletely matched inside edge
        elif isinstance(node, int):
            # Get parent and child
            _, parent, handled_length = path[-2]  # type: _DirectedAcyclicSequenceGraphNode, int
            child = parent.children[edge]  # type: _DirectedAcyclicSequenceGraphNode

            # Remove edge
            del parent.children[edge]

            # Edge values
            pnp_edge = sequence[handled_length:handled_length + node]
            npc_edge = edge[len(pnp_edge):]
            npn_edge = sequence[handled_length + node:]

            # Make new nodes and mark one as end
            new_parent = self._build_node()
            new_node = self._build_node()
            new_node.is_sequence_end = True

            # Create edges at edge-split
            parent.children[pnp_edge] = new_parent
            new_parent.children[npc_edge] = child

            # Create edge to new node
            new_parent.children[npn_edge] = new_node

            # Add unchecked node
            # self._checks_to_do.append(_Check(parent, pnp_edge, new_parent))
            self._checks_to_do.append(_Check(new_parent, npc_edge, child))
            # self._checks_to_do.append(_Check(new_parent, npn_edge, new_node))

            # # Move to next node
            last_node = new_node

        # None of sequence matched anything
        elif node is None:
            # Get parent and child
            _, parent, handled_length = path[-2]  # type: _DirectedAcyclicSequenceGraphNode, int

            # Make new node and mark as end
            new_node = self._build_node()
            new_node.is_sequence_end = True

            # Edge value
            pn_edge = sequence[handled_length:]

            # Link parent to new node and new node to child
            parent.children[pn_edge] = new_node

            # Add unchecked node
            self._checks_to_do.append(_Check(parent, pn_edge, new_node))

            # # Move to next node
            last_node = new_node

        # Remember sequence for next insertion
        self._previous_path = path
        self._previous_sequence = sequence

        # Increment counter
        self._sequence_count += 1

        # Note longest sequence
        self._height = max(self._height, nodes_in_path(path))

    def _finish(self):
        # Minimize all
        self._minimize()

    def _minimize(self, up_to=0):
        """
        Check the unchecked_nodes for redundant nodes, proceeding from last one down to the common
        prefix size. Then truncate the list at that point.
        :param int up_to: Upper limit in height for minimization.
        """
        # TODO: Does this work correctly? Does it not find the same node as is it looking for?
        # Split checks into those below upper-limit and those below
        self._checks_to_do, checks_below = self._checks_to_do[:up_to], self._checks_to_do[up_to:]

        # Handle checks below from bottom and up
        for parent, element, child in reversed(checks_below):

            # Check if there exists a minimized node identical to child (identical subtree)
            if child in self._nodes:
                # Replace child with minimized node
                parent.children[element] = self._nodes[child]
            else:
                # Add chile to minimized nodes for later usage
                self._nodes[child] = child

    #########
    # Searching

    def _convert_output(self, current_elements):
        if self._sequence_type == str:
            return "".join(current_elements)
        elif self._sequence_type == list or self._sequence_type == tuple:
            list_current_elements = []
            for element in current_elements:
                list_current_elements += list(element)
            return self._sequence_type(list_current_elements)
        else:
            NotImplementedError(f"Conversion from output type {type(current_elements).__name__} to "
                                f"{self._sequence_type} not implemented.")

    def edit_distance_search(self, query_sequence, max_cost=1, sort=False, exclude_query=False,
                             exclusions=None):
        """
        Searches for all sequences that have an edit-distance of at most "max_cost" from "query_sequence".
        :param str | list | tuple query_sequence: The query sequence.
        :param int max_cost: Maximum accepted edit-cost for search.
        :param bool sort: If True, the output will be sorted with respect to the distance measure.
        :param bool exclude_query: Don't output query_sequence as result.
        :param set | list exclusions: Sequences to ignore.
        :return: list[str]
        """
        return edit_distance_search(graph=self,
                                    query_sequence=query_sequence,
                                    max_cost=max_cost,
                                    sort=sort,
                                    exclude_query=exclude_query,
                                    exclusions=exclusions)


def edit_distance_search_recursion(graph, node,
                                   parent_elements,
                                   current_elements,
                                   query_sequence,
                                   previous_row,
                                   max_cost):
    """
    Recursively searches for sequences that have an edit-distance of at most "max_cost" from "query_sequence".
    :param DirectedAcyclicCompressedSequenceGraph graph: The graph in question.
    :param _DirectedAcyclicSequenceGraphNode node: The current node in the search.
    :param list parent_elements: The elements of parent sequence.
    :param list current_elements: The elements of the edge leading to this node.
    :param str | list | tuple query_sequence: The query sequence.
    :param list[int] previous_row: Previous row in dynamic cost-table.
    :param int max_cost: Maximum accepted edit-cost for search.
    """
    # Number of columns
    columns = len(query_sequence) + 1

    # Go through elements in edge
    current_row = previous_row
    for element in current_elements:

        # Initialize current row
        current_row = [None] * columns  # type: List[int]
        current_row[0] = previous_row[0] + 1

        # Build one row for the element, with a column for each element in the target
        # sequence, plus one for the empty element at column 0
        for column in range(1, columns):
            # Current node can be reached from prefix by force-inserting the correct element
            # The cost after this insertion
            insert_cost = current_row[column - 1] + 1

            # Current node can be reached by deleting the next search element.
            # The cost after deletion is the current cost plus 1
            delete_cost = previous_row[column] + 1

            # Assuming neither deletion nor insertion, the current cost is
            replace_cost = previous_row[column - 1]
            if query_sequence[column - 1] != element:
                replace_cost += 1

            # Append the minimum value to the row
            current_row[column] = min(insert_cost, delete_cost, replace_cost)

        # On to next row
        previous_row = current_row

    # Current elements
    current_elements = parent_elements + [current_elements]

    # Check if cost is acceptable and if node marks end of sequence
    results = []
    if current_row[-1] <= max_cost and node.is_sequence_end:
        # Append result
        sequence = graph._convert_output(current_elements=current_elements)
        results.append((sequence, current_row[-1]))

    # Check if any cost in row is acceptable
    if min(current_row) <= max_cost:

        # Go through all children and search recursively
        for child_element in node.children:
            child_results = edit_distance_search_recursion(
                graph=graph,
                node=node.children[child_element],
                parent_elements=current_elements,
                current_elements=child_element,
                query_sequence=query_sequence,
                previous_row=current_row,
                max_cost=max_cost
            )

            # Extend results
            results.extend(child_results)

    # Return
    return results


def edit_distance_search(graph, query_sequence, max_cost=1, sort=False, exclude_query=False,
                         exclusions=None):
    """
    Searches for all sequences that have an edit-distance of at most "max_cost" from "query_sequence".
    :param DirectedAcyclicCompressedSequenceGraph graph: The graph in question.
    :param str | list | tuple query_sequence: The query sequence.
    :param int max_cost: Maximum accepted edit-cost for search.
    :param bool sort: If True, the output will be sorted with respect to the distance measure.
    :param bool exclude_query: Don't output query_sequence as result.
    :param set | list exclusions: Sequences to ignore.
    :return: list[str]
    """
    # Check for simple search
    if max_cost == 0:
        if query_sequence in graph:
            return [(query_sequence, 0)]
        else:
            return []

    # Root node
    root = graph._root

    # Build first row of dynamic cost-table
    current_row = list(range(len(query_sequence) + 1))

    # List of search-results
    results = []

    # Recursively search each branch of the trie
    for element in root.children.keys():
        child_results = edit_distance_search_recursion(
            graph=graph,
            node=root.children[element],
            parent_elements=[],
            current_elements=element,
            query_sequence=query_sequence,
            previous_row=current_row,
            max_cost=max_cost
        )

        # Extend results
        results.extend(child_results)

    # Sort if needed
    if sort:
        results = sorted(results, key=lambda x: x[1])

    # Attempt using set for quicker lookup
    if exclusions is not None:
        try:
            exclusions = set(exclusions)
        except TypeError:
            pass

    # Add query sequence to exclusions if wanted
    if exclude_query:
        if isinstance(exclusions, set):
            exclusions.update(query_sequence)
        elif isinstance(exclusions, list):
            exclusions.append(query_sequence)
        elif exclusions is None:
            exclusions = [query_sequence]
        else:
            raise TypeError("Filtering of query-sequence not possible. ")

    # Filter sequences
    if exclusions is not None:
        results = [sequence for sequence in results if sequence[0] not in exclusions]

    # Return
    return results


def nodes_in_path(path):
    _, last_node, _ = path[-1]
    if isinstance(last_node, _DirectedAcyclicSequenceGraphNode):
        return len(path)
    return len(path) - 1


def common_path_nodes(path1, path2):
    return nodes_in_path(
        [element1 for element1, element2 in zip(path1, path2)
         if element1[0] == element2[0]]
    )


def path2sequence(path):
    return "".join([edge for edge, node, _ in path
                    if isinstance(node, _DirectedAcyclicSequenceGraphNode)])


def common_prefix(sequence1, sequence2):
    # TODO: Cython this method
    """
    Reports the length of the longest common prefix of two sequences.
    :return: int
    """
    i = 0
    for elem1, elem2 in zip(sequence1, sequence2):
        if elem1 != elem2:
            return i
        i += 1

    # Return length of sequence if sequences are identical
    return min(len(sequence1), len(sequence2))


if __name__ == "__main__":
    from pathlib import Path
    import os

    if Path.cwd().name != "dasg":
        os.chdir("Repositories/dasg")

    line = "-" * 75

    #############################
    # Non-string sequences

    print("\n\n" + line)
    print("Testing sequences of integers. ")

    the_sequences = [list(range(idx, idx + 5)) for idx in range(10)]
    the_dasg = DirectedAcyclicCompressedSequenceGraph(the_sequences)

    query = list(range(6, 6 + 5))
    if query in the_dasg:
        print(f"\n'{query}' is in the DASG.")
    else:
        print(f"\n'{query}' not in DASG.")

    print(f"\nSequences closest to '{query}':")
    print(the_dasg.edit_distance_search(query_sequence=query, max_cost=2, sort=True, exclude_query=True))

    print("\nPrint out of all sequences in DASG:")
    for c_sequence in the_dasg:
        print("\t", c_sequence)

    #############################
    # Text-file

    print("\n\n" + line)
    print("Analysing text-file.\n")

    # Get test-text
    with codecs.open(str("cats.txt"), encoding="utf-8", mode="r") as file:
        text = "\n".join([line for line in file])

    # Words and query
    all_words = ["family,"] + text.split()
    query = "pass"

    # Make DASG
    the_dasg = DirectedAcyclicCompressedSequenceGraph(all_words, verbose=True)

    for word in all_words:
        if word not in the_dasg:
            print(f"'{word}' not in DASG")

    # Print success
    print(f"Read {the_dasg.sequence_count} words into {the_dasg.node_count} nodes and {the_dasg.edge_count} edges")

    # Check if is in DASG
    if query in the_dasg:
        print(f"\n'{query}' is in the DASG.")
    else:
        print(f"\n'{query}' not in DASG.")

    query = "chill"
    print(f"\nWords closest to '{query}':")
    print(the_dasg.edit_distance_search(query_sequence=query, max_cost=2, sort=True))
