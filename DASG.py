from pathlib import Path
from time import time
from typing import List, Hashable


class _DirectedAcyclicSequenceGraphNode:
    def __init__(self, node_id):
        """
        Represents a node in a DASG.
        Contains an ID, a flag indicating whether the node is a leaf and a set of children.
        :param int node_id: ID of node.
        """
        # Assign ID and update class-counter
        self.id = node_id

        # Initialize
        self.is_sequence_end = False
        self.children = {}

    def subtree_identity(self):
        """
        Makes an immutable object representing the node, which can be used for comparing two
        nodes for equal subtrees (all children are identical).
        :return: str, str, str
        """
        if self.children:
            temp = [(str(edge), str(node.id), "1" if node.is_sequence_end else "0")
                    for (edge, node) in self.children.items()]
            edge_string, id_string, status_string = \
                zip(*temp)
        else:
            edge_string = id_string = status_string = ""

        # Join with underscores
        edge_string = "r_" + "_".join(edge_string)
        id_string = "r_" + "_".join(id_string)
        status_string = ("1_" if self.is_sequence_end else "0_") + "_".join(status_string)

        return edge_string, id_string, status_string

    def __hash__(self):
        return self.subtree_identity().__hash__()

    def __eq__(self, other):
        return self.subtree_identity() == other.subtree_identity()


class _Check:
    def __init__(self, node, element, next_node):
        self.next_node = next_node
        self.element = element
        self.node = node

    def __iter__(self):
        return (item for item in (self.node, self.element, self.next_node))


class DirectedAcyclicSequenceGraph:
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

    def __contains__(self, sequence):
        # Start at root
        node = self._root

        # Go through elements
        for element in sequence:
            # If element is not found after current prefix return false
            if element not in node.children:
                return False

            # Next element
            node = node.children[element]

        # Return sequence-end-status of node
        return node.is_sequence_end

    def __iter__(self):
        # Search queue
        search_nodes = [(0, edge, child) for edge, child in self._root.children.items()]
        # search_nodes.sort()
        search_nodes.reverse()

        # Current position
        current_sequence = []

        # Go through queue
        while search_nodes:
            node_height, edge, node = search_nodes.pop()

            # Check length of current sequence
            current_sequence = current_sequence[:node_height]  # type: list

            # Add new element
            current_sequence.append(edge)

            # Check if node is end of sequence
            if node.is_sequence_end:
                yield self._convert_output(current_sequence)

            # Add children
            additions = [(node_height + 1, edge, child) for edge, child in node.children.items()]
            # additions.sort()
            additions.reverse()
            search_nodes.extend(additions)

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

    @staticmethod
    def _common_prefix(sequence1, sequence2):
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

        # Check if hashable
        if not isinstance(sequences[0], Hashable):
            sequences = [tuple(val) for val in sequences]

        # Make sure sequences are list-type and sorted
        sequences = sorted(list(set(sequences)))

        # Type
        self._sequence_type = type(sequences[0])

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
        if self._previous_sequence is not None:

            # Check sequence-order consistency
            if sequence < self._previous_sequence:
                raise ValueError(f"{type(self).__name__}: Sequences must be ordered (comparable). "
                                 f"{type(sequence).__name__} and {type(self._previous_sequence).__name__} are not")

            # Find common prefix between sequence and previous sequence
            common_prefix = self._common_prefix(sequence, self._previous_sequence)

        else:
            common_prefix = 0

        # Reduce by merging nodes with equal subtrees
        self._minimize(common_prefix)

        # Add the suffix, starting from the correct node mid-way through the graph
        if len(self._checks_to_do) == 0:
            node = self._root
        else:
            node = self._checks_to_do[-1].next_node

        # Go through elements after the common prefix
        for element in sequence[common_prefix:]:
            # Make new node
            next_node = self._build_node()

            # Create edge between nodes with element as identifier
            node.children[element] = next_node

            # Add unchecked node
            self._checks_to_do.append(_Check(node, element, next_node))

            # Move to next node
            node = next_node

        # Note that this node represents the end of a sequence
        node.is_sequence_end = True

        # Remember sequence for next insertion
        self._previous_sequence = sequence

        # Increment counter
        self._sequence_count += 1

        # Note longest sequence
        self._height = max(self._height, len(sequence))

    def _finish(self):
        # Minimize all
        self._minimize()

    def _minimize(self, up_to=0):
        """
        Check the unchecked_nodes for redundant nodes, proceeding from last one down to the common
        prefix size. Then truncate the list at that point.
        :param int up_to: Upper limit in height for minimization.
        """
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
            return self._sequence_type(current_elements)
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
        # Check for simple search
        if max_cost == 0:
            if query_sequence in self:
                return [(query_sequence, 0)]
            else:
                return []

        # Root node
        root = self._root

        # Build first row of dynamic cost-table
        current_row = list(range(len(query_sequence) + 1))

        # List of search-results
        results = []

        # Recursively search each branch of the trie
        for element in root.children.keys():
            child_results = self._edit_distance_search_recursion(node=root.children[element],
                                                                 parent_elements=[],
                                                                 current_element=element,
                                                                 query_sequence=query_sequence,
                                                                 previous_row=current_row,
                                                                 max_cost=max_cost)

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

    def _edit_distance_search_recursion(self, node, parent_elements, current_element, query_sequence,
                                        previous_row, max_cost):
        """
        Recursively searches for sequences that have an edit-distance of at most "max_cost" from "query_sequence".
        :param _DirectedAcyclicSequenceGraphNode node: The current node in the search.
        :param list parent_elements: The elements of parent sequence.
        :param current_element: The element leading to this node.
        :param str | list | tuple query_sequence: The query sequence.
        :param list[int] previous_row: Previous row in dynamic cost-table.
        :param int max_cost: Maximum accepted edit-cost for search.
        """
        # Number of columns
        columns = len(query_sequence) + 1

        # Initialize current row
        current_row = [None] * columns  # type: List[int]
        current_row[0] = previous_row[0] + 1

        # Current elements
        current_elements = parent_elements + [current_element]

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
            if query_sequence[column - 1] != current_element:
                replace_cost += 1

            # Append the minimum value to the row
            current_row[column] = min(insert_cost, delete_cost, replace_cost)

        # Check if cost is acceptable and if node marks end of sequence
        results = []
        if current_row[-1] <= max_cost and node.is_sequence_end:
            # Append result
            sequence = self._convert_output(current_elements=current_elements)
            results.append((sequence, current_row[-1]))

        # Check if any cost in row is acceptable
        if min(current_row) <= max_cost:

            # Go through all children and search recursively
            for child_element in node.children:
                child_results = self._edit_distance_search_recursion(node=node.children[child_element],
                                                                     parent_elements=current_elements,
                                                                     current_element=child_element,
                                                                     query_sequence=query_sequence,
                                                                     previous_row=current_row,
                                                                     max_cost=max_cost)

                # Extend results
                results.extend(child_results)

        # Return
        return results
