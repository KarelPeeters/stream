import random
from dataclasses import dataclass
from typing import Optional, List

import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp

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


class AllocationProblem:
    def __init__(self, start_time: float):
        self.check = random.randint(0, 2 ** 32)
        self.tokens = []

        self.start_time = start_time
        self.final_time = None
        self.available_size = None

        self.allocated = False
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

    def set_final_time(self, time: float):
        assert self.final_time is None, "Final time has already been set"
        self.final_time = time

        # free all remaining tokens
        for token in self.tokens:
            assert time >= token.time_start and (token.time_end is None or time >= token.time_end)
            if token.time_end is None:
                self.free(token, self.final_time)

    def set_available_size(self, size: int):
        assert self.available_size is None, "Available size has already been set"
        self.available_size = size

    def all_times(self) -> List[float]:
        result = {self.start_time, self.final_time}
        for token in self.tokens:
            result.add(token.time_start)
            result.add(token.time_end)
        if None in result:
            result.remove(None)
        return sorted(result)

    def plot_solved(self, block_path: Optional[str], line_path: Optional[str]):
        assert self.allocated

        # Patched plot
        plot_size = self.available_size if self.available_size is not None else self.allocated_size_used

        ax = plt.figure(figsize=(16, 16)).gca()
        ax.set_xlim(0, self.final_time)
        ax.set_ylim(0, plot_size * 1.2)

        ax.add_patch(plt.Rectangle((self.start_time, 0), self.final_time - self.start_time, plot_size, color='g'))

        for token in self.tokens:
            ax.add_patch(plt.Rectangle(
                (token.time_start, token.offset),
                token.time_end - token.time_start,
                token.size,
                color='r'
            ))

        if block_path is None:
            plt.show(block=False)
        else:
            plt.savefig(block_path)
            plt.close()

        # Line plot
        ax = plt.figure().gca()

        timestamps = self.all_times()
        mem_used_dense = [
            sum(token.size for token in self.tokens if token.time_start <= time < token.time_end)
            for time in timestamps
        ]

        plt.plot(timestamps, mem_used_dense, drawstyle="steps-post")
        plot_mem_line(ax, timestamps, max(mem_used_dense), self.available_size, "dotted", "top")
        plot_mem_line(ax, timestamps, self.allocated_size_used, self.available_size, "dashed", "bottom")

        if line_path is None:
            plt.show()
        else:
            plt.savefig(line_path)
            plt.close()

    # TODO use a better memory allocation algorithm for this
    def run_allocation(self) -> AllocationHistory:
        assert not self.allocated, "Allocation has already happened"
        assert self.final_time is not None, "Final time has not been set"

        free_segments = [(0, self.available_size)]
        history = []

        # find all distinct time points
        times_set = set(t for token in self.tokens for t in (token.time_start, token.time_end) if t is not None)
        times_set.add(self.start_time)
        times_set.add(self.final_time)
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
        self.allocated = True

        return AllocationHistory(size=self.available_size, size_used=size_used, size_used_dense=size_used_dense,
                                 history=history)

    def solve_allocation_perfect(self):
        EPSILON = 1e-10
        solver = pywraplp.Solver.CreateSolver("SAT")

        # create all variables
        token_var_pairs = []

        # todo only use overlapping tokens for this
        worst_case_size = sum(token.size for token in self.tokens)

        var_mem_used = solver.IntVar(0, worst_case_size, "mem_used")

        for ai, token in enumerate(self.tokens):
            if token.size == 0:
                token.offset = 0
                continue

            var_offset = solver.IntVar(0, worst_case_size - token.size, f"offset_{ai}")
            token_var_pairs.append((token, var_offset))

            # constraint: below peak mem usage
            solver.Add(var_offset + token.size <= var_mem_used)

        # visit overlapping tokens
        for ai, (token_a, var_a) in enumerate(token_var_pairs):
            for bi, (token_b, var_b) in enumerate(token_var_pairs[:ai]):
                # skip if no overlap
                if token_a.time_start >= token_b.time_end or token_b.time_start >= token_a.time_end:
                    continue

                a_above_b = solver.IntVar(0, 1, f"Bp_{ai}_{bi}")
                b_above_a = solver.IntVar(0, 1, f"Bn_{ai}_{bi}")

                # constraint: below xor above
                solver.Add(a_above_b + b_above_a == 1)

                # constraint: no overlap
                solver.Add(var_a + token_a.size - worst_case_size * a_above_b <= var_b - 1)
                solver.Add(var_b + token_b.size - worst_case_size * b_above_a <= var_a - 1)

        solver.Minimize(var_mem_used)
        status = solver.Solve()
        assert (status == pywraplp.Solver.OPTIMAL)

        def to_int(x):
            assert abs(int(round(x)) - x) < EPSILON
            return int(round(x))

        for (token, var) in token_var_pairs:
            var_value = var.solution_value()
            token.offset = to_int(var_value)

        self.allocated_size_used = to_int(var_mem_used.solution_value())
        self.allocated = True


def problem_simple() -> AllocationProblem:
    alloc = AllocationProblem(0)

    a = alloc.alloc(128, 1)
    b = alloc.alloc(256, 4)
    alloc.free(a, 6)
    # c = alloc.alloc(3)
    alloc.free(b, 8)
    # alloc.free(c)
    _ = alloc.alloc(512, 20)

    alloc.set_final_time(30)

    return alloc


def problem_frag() -> AllocationProblem:
    alloc = AllocationProblem(0)
    a = alloc.alloc(128, 1)
    b = alloc.alloc(256, 2)
    alloc.free(a, 3)
    _obstacle = alloc.alloc(64, 4)
    _ = alloc.alloc(128, 4)
    alloc.free(b, 5)
    alloc.set_final_time(7)
    return alloc


def main():
    # problem = problem_simple()
    problem = problem_frag()

    problem.run_allocation().plot_history(None, None)
    # problem.solve_allocation_perfect()
    problem.plot_solved(None, None)

    for token in problem.tokens:
        print(token)


if __name__ == '__main__':
    main()
