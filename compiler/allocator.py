import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt

from stream.visualization.memory_usage import humanbytes, BIGGER_SIZE


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
    check: int

    index: int
    size: int

    time_start: float
    time_end: Optional[float]

    offset: Optional[int] = None


def plot_mem_line(ax, timestamps, y, y_size, style: str, va: str):
    ax.axhline(
        y=y,
        xmin=min(timestamps),
        xmax=max(timestamps),
        color="r",
        linestyle=style,
    )

    if y_size is None:
        mem_text = humanbytes(y)
    else:
        mem_text = f"{humanbytes(y)} / {humanbytes(y_size)}"

    ax.text(
        max(timestamps) - 1,
        y,
        mem_text,
        color="r",
        va=va,
        ha="right",
        fontsize=BIGGER_SIZE,
    )


@dataclass
class AllocationHistory:
    size: Optional[int]
    size_used: int

    size_used_dense: int
    history: List[Tuple[float, List[Tuple[int, int]]]]

    def plot_history(self, block_path: Optional[str], line_path: Optional[str]):
        plot_size = self.size if self.size is not None else self.size_used

        # plot patch
        ax = plt.figure(figsize=(16, 16)).gca()
        ax.set_xlim(0, self.history[-1][0])
        ax.set_ylim(0, plot_size * 1.2)

        timestamps = []
        mem_used_dense = []

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

                if end is None:
                    end = plot_size

                ax.add_patch(plt.Rectangle((time_start, start), time_delta, end - start, color='g'))
                prev_end = end

            if prev_end != plot_size:
                ax.add_patch(
                    plt.Rectangle((time_start, prev_end), time_delta, self.size - prev_end, color='r'))

            timestamps.append(time_start)
            mem_free = sum((end if end is not None else plot_size) - start for start, end in free_segments)
            mem_used_dense.append(plot_size - mem_free)

        if not (any(x > 0 for x in mem_used_dense)):
            plt.close()
            return

        if block_path is None:
            plt.show()
        else:
            plt.savefig(block_path)
        plt.close()

        ax = plt.figure().gca()
        plt.plot(timestamps, mem_used_dense, drawstyle="steps-post")
        plot_mem_line(ax, timestamps, max(mem_used_dense), self.size, "dotted", "top")
        plot_mem_line(ax, timestamps, self.size_used, self.size, "dashed", "bottom")

        if line_path is None:
            plt.show()
        else:
            plt.savefig(line_path)
        plt.close()


class TimeAllocator:
    def __init__(self, start_time: float):
        self.check = random.randint(0, 2 ** 32)
        self.tokens = []

        self.start_time = start_time

        self.allocated = False
        self.allocated_size = None
        self.allocated_size_used = None

    def alloc(self, size: int, time: float) -> Token:
        assert int(size) == size
        size = int(size)

        token = Token(self.check, len(self.tokens), size, time, None)
        self.tokens.append(token)
        return token

    def free(self, token: Token, time: float):
        assert self.check == token.check
        assert token.time_end is None
        token.time_end = time

    # TODO use a better memory allocation algorithm for this
    def run_allocation(self, size: Optional[int], final_time: float) -> AllocationHistory:
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
        size_used = 0
        size_used_dense = 0
        curr_size_used_dense = 0

        for t in times:
            print(f"T={t}, free_segments={free_segments}")

            # allocate new tensors in first fitting range
            for token in self.tokens:
                if token.time_start == t:
                    print(f"Allocating token {token.index}")
                    for (seg_index, (seg_start, seg_end)) in enumerate(free_segments):
                        if seg_end is None or (seg_end - seg_start) >= token.size:
                            token.offset = seg_start
                            size_used = max(size_used, seg_start + token.size)
                            curr_size_used_dense += token.size

                            if seg_end is not None and (seg_end - seg_start) == token.size:
                                free_segments.pop(seg_index)
                            else:
                                free_segments[seg_index] = (seg_start + token.size, seg_end)

                            break
                    else:
                        raise ValueError(f"Failed to allocate {token}")

            size_used_dense = max(size_used_dense, curr_size_used_dense)

            # free old tensors (after allocating new ones, just be extra safe there's no overlap)
            for token in self.tokens:
                if token.time_end == t:
                    print(f"Freeing token {token.index}")
                    new_segment = (token.offset, token.offset + token.size)
                    curr_size_used_dense -= token.size

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

        self.allocated_size_used = size_used
        return AllocationHistory(size=size, size_used=size_used, size_used_dense=size_used_dense, history=history)


def main():
    alloc = TimeAllocator(0)

    a = alloc.alloc(128, 1)
    b = alloc.alloc(256, 4)
    alloc.free(a, 6)
    # c = alloc.alloc(3)
    alloc.free(b, 8)
    # alloc.free(c)
    d = alloc.alloc(512, 20)

    alloc.run_allocation(None, 30).plot_history(None, None)

    for token in alloc.tokens:
        print(token)


if __name__ == '__main__':
    main()
