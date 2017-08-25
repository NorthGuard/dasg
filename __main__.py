from pathlib import Path

from dasg import DirectedAcyclicSequenceGraph


line = "-" * 75

#############################
# Non-string sequences

print("\n\n" + line)
print("Testing sequences of integers. ")

the_sequences = [list(range(idx, idx + 5)) for idx in range(10)]
the_dasg = DirectedAcyclicSequenceGraph(the_sequences)

query = list(range(6, 6 + 5))
if query in the_dasg:
    print(f"\n'{query}' is in the DASG.")
else:
    print(f"\n'{query}' not in DASG.")

print(f"\nSequences closest to '{query}':")
print(the_dasg.edit_distance_search(query_sequence=query, max_cost=2, sort=True, exclude_query=True))

print("\nPrint out of all sequences in DASG:")
for sequence in the_dasg:
    print("\t", sequence)

#############################
# Text-file

print("\n\n" + line)
print("Analysing text-file.\n")

# Get test-text
text = "\n".join([line for line in Path("text.txt").open("r")])

# Words and query
all_words = text.split()
query = "pass"

# Make DASG
the_dasg = DirectedAcyclicSequenceGraph(all_words, verbose=True)

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
