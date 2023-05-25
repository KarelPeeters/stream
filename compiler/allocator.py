import random
from dataclasses import dataclass, field
from typing import Optional


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

    start: int
    end: Optional[int]

    offset: Optional[int] = None


class TimeAllocator:
    def __init__(self):
        self.check = random.randint(0, 2 ** 32)
        self.tokens = []
        self.steps = 0
        self.allocated = False

    def alloc(self, size: int) -> Token:
        token = Token(self.check, len(self.tokens), size, self.steps, None)

        self.steps += 1
        self.tokens.append(token)

        return token

    def free(self, token: Token):
        assert self.check == token.check
        assert token.end is None

        token.end = self.steps
        self.steps += 1

    def allocate(self, size: int):
        assert not self.allocated
        self.allocated = True

        # TODO use a better memory allocation algorithm for this
        free_segments = [(0, size)]

        for t in range(self.steps):
            print(f"T={t}, free_segments={free_segments}")

            # allocate new tensors in first fitting range
            for token in self.tokens:
                if token.start == t:
                    print(f"Allocating token {token.index}")
                    for (seg_index, (seg_start, seg_end)) in enumerate(free_segments):
                        if (seg_end - seg_start) >= token.size:
                            token.offset = seg_start

                            if (seg_end - seg_start) == token.size:
                                free_segments.pop(seg_index)
                            else:
                                free_segments[seg_index] = (seg_start + token.size, seg_end)

                            break

            # free old tensors
            for token in self.tokens:
                if token.end == t:
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


def main():
    alloc = TimeAllocator()

    a = alloc.alloc(1)
    b = alloc.alloc(2)
    alloc.free(a)
    # c = alloc.alloc(3)
    alloc.free(b)
    # alloc.free(c)
    d = alloc.alloc(4)

    alloc.allocate(1024)

    for token in alloc.tokens:
        print(token)


if __name__ == '__main__':
    main()
