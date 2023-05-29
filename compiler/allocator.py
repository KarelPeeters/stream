import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt


class LinearAllocator:
    def __init__(self):
        self.next_offset = 0

    def alloc(self, size):
        assert int(size) == size
        size = int(size)

        offset = self.next_offset
        self.next_offset += size
        return offset

    def __len__(self):
        return self.next_offset


@dataclass
class Token:
    check: int = field(repr=False)

    index: int
    size: int

    time_start: float
    time_end: Optional[float]

    offset: Optional[int] = None


@dataclass
class AllocationHistory:
    size: int
    history: List[Tuple[float, List[Tuple[int, int]]]]

    def plot_history(self, path: Optional[str], block: bool):
        # plot patch
        ax = plt.figure(figsize=(16, 16)).gca()
        ax.set_xlim(0, self.history[-1][0])
        ax.set_ylim(0, self.size)

        for i, (time_start, free_segments) in enumerate(self.history):
            if i < len(self.history) - 1:
                time_delta = self.history[i + 1][0] - time_start
            else:
                time_delta = 1

            print(f"Plotting time {time_start}..{time_start + time_delta}: {free_segments}")

            prev_end = 0
            for (start, end) in free_segments:
                if start != prev_end:
                    ax.add_patch(plt.Rectangle((time_start, prev_end), time_delta, start - prev_end, color='r'))

                ax.add_patch(plt.Rectangle((time_start, start), time_delta, end - start, color='g'))
                prev_end = end

            if prev_end != self.size:
                ax.add_patch(
                    plt.Rectangle((time_start, prev_end), time_delta, self.size - prev_end, color='r'))

        if path is not None:
            plt.savefig(path)
        plt.show(block=block)


class TimeAllocator:
    def __init__(self, start_time: float):
        self.check = random.randint(0, 2 ** 32)
        self.tokens = []
        self.allocated = False

        self.start_time = start_time

    def alloc(self, size: int, time: float) -> Token:
        token = Token(self.check, len(self.tokens), size, time, None)
        self.tokens.append(token)
        return token

    def free(self, token: Token, time: float):
        assert self.check == token.check
        assert token.time_end is None
        token.time_end = time

    # TODO use a better memory allocation algorithm for this
    def run_allocation(self, size: int, final_time: float) -> AllocationHistory:
        assert not self.allocated, "Allocation has already happened"
        self.allocated_size = size

        # free all remaining tokens
        for token in self.tokens:
            if token.time_end is None:
                self.free(token, final_time)

        free_segments = [(0, size)]
        history = []

        # find all distinct time points
        times_set = set(t for token in self.tokens for t in (token.time_start, token.time_end) if t is not None)
        times_set.add(self.start_time)
        times_set.add(final_time)
        times = sorted(times_set)

        # loop over potential event times
        history.append((self.start_time, list(free_segments)))

        for t in times:
            print(f"T={t}, free_segments={free_segments}")

            # allocate new tensors in first fitting range
            for token in self.tokens:
                if token.time_start == t:
                    print(f"Allocating token {token.index}")
                    for (seg_index, (seg_start, seg_end)) in enumerate(free_segments):
                        if (seg_end - seg_start) >= token.size:
                            token.offset = seg_start

                            if (seg_end - seg_start) == token.size:
                                free_segments.pop(seg_index)
                            else:
                                free_segments[seg_index] = (seg_start + token.size, seg_end)

                            break

            # free old tensors (after allocating new ones, just be extre safe there's no overlap)
            for token in self.tokens:
                if token.time_end == t:
                    print(f"Freeing token {token.index}")
                    new_segment = (token.offset, token.offset + token.size)

                    # insert new segment at the right spot
                    for (seg_index, (seg_start, seg_end)) in enumerate(free_segments):
                        if seg_start > new_segment[0]:
                            free_segments.insert(seg_index, new_segment)
                            break
                    else:
                        free_segments.append(new_segment)

            # merge ranges
            seg_index = 0
            while seg_index < len(free_segments) - 1:
                if free_segments[seg_index][1] == free_segments[seg_index + 1][0]:
                    free_segments[seg_index] = (free_segments[seg_index][0], free_segments[seg_index + 1][1])
                    free_segments.pop(seg_index + 1)
                else:
                    seg_index += 1

            history.append((t, list(free_segments)))

        return AllocationHistory(size, history)


def main():
    alloc = TimeAllocator(0)

    a = alloc.alloc(128, 1)
    b = alloc.alloc(256, 4)
    alloc.free(a, 6)
    # c = alloc.alloc(3)
    alloc.free(b, 8)
    # alloc.free(c)
    d = alloc.alloc(512, 20)

    alloc.run_allocation(1024, 30).plot_history(None, True)

    for token in alloc.tokens:
        print(token)


if __name__ == '__main__':
    main()
