import random

class DfaState:
    def __init__(self, alphabet: set[str], name="Unnamed State") -> None:
        self.alphabet = alphabet
        self.transitions = {}
        self.name = name
        for character in alphabet:
            self.transitions[character] = self

def iterate(string: str, start: DfaState):
    current_state = start
    for i in range(len(string)):
        character = string[i]
        print(current_state.name, f"-{character}->", end=" ")
        current_state = current_state.transitions[character]
    print(current_state.name)
    return current_state

def precomputeSuffixCounts(machine: set[DfaState], accepting: set[DfaState], limit: int):
    # `state.suffixes[i] == j` means:
    # if you are in state `state`, there are `j` distinct suffixes of length `i`
    # that lead to an accepting state
    for state in machine:
        if state in accepting:
            state.suffixes = [1]
        else:
            state.suffixes = [0]
    for i in range(limit):
        # suffixes[i] is already calculated
        for state in machine:
            current_sum = 0
            for character in state.alphabet:
                current_sum += state.transitions[character].suffixes[i]
            state.suffixes.append(current_sum)

def sampleRandom(machine: set[DfaState], start: DfaState, length: int):
    current_string = ""
    current_state = start
    for i in range(length):
        value = random.randint(0, current_state.suffixes[length - i] - 1)
        random_character = None
        for character in current_state.alphabet:
            value -= current_state.transitions[character].suffixes[length - i - 1]
            if value < 0:
                random_character = character
                current_state = current_state.transitions[character]
                break
        current_string = current_string + random_character
    return current_string

alphabet = {"a", "b"}
# a goes to A
# b goes to B

A = DfaState(alphabet, "A")
B = DfaState(alphabet, "B")

B.transitions["a"] = A
A.transitions["b"] = B

iterate("abababa", A)

precomputeSuffixCounts({A, B}, {B}, 8)
# print("A:", A.suffixes)
# print("B:", B.suffixes)
print(sampleRandom({A, B}, A, 8))
    