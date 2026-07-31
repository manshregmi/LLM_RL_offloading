import pickle
import numpy as np


def to_python(value):
    """Convert NumPy scalar values to standard Python values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return tuple(to_python(x) for x in value)

    if isinstance(value, list):
        return [to_python(x) for x in value]

    return value


def print_value_table(value_table):
    print("\n=== CRITIC / VALUE TABLE ===")
    print(f"Number of states: {len(value_table)}\n")

    for index, (state, value) in enumerate(value_table.items(), start=1):
        bandwidth, contention, segment, previous_assignment = state

        print(f"State {index}")
        print(f"  Bandwidth bin:       {to_python(bandwidth)}")
        print(f"  Contention bin:      {to_python(contention)}")
        print(f"  Segment:             {to_python(segment)}")
        print(f"  Previous assignment: {to_python(previous_assignment)}")
        print(f"  Critic value:        {to_python(value)}")
        print()


def print_policy_table(policy_table):
    print("\n=== ACTOR / POLICY TABLE ===")
    print(f"Number of entries: {len(policy_table)}\n")

    for index, (key, value) in enumerate(policy_table.items(), start=1):
        print(f"Policy entry {index}")
        print(f"  Key:   {to_python(key)}")
        print(f"  Value: {to_python(value)}")
        print()


file_path = "a2c_tables.pkl"

with open(file_path, "rb") as file:
    data = pickle.load(file)

print(f"Loaded object type: {type(data).__name__}")

if isinstance(data, (tuple, list)) and len(data) >= 2:
    policy_table = data[0]
    value_table = data[1]

    if isinstance(policy_table, dict):
        print_policy_table(policy_table)
    else:
        print(f"data[0] is not a dictionary: {type(policy_table)}")

    if isinstance(value_table, dict):
        print_value_table(value_table)
    else:
        print(f"data[1] is not a dictionary: {type(value_table)}")

else:
    print("Unexpected pickle structure.")