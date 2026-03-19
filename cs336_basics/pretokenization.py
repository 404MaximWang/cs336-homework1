import os
import psutil

def find_chunk_boundaries(
    file_path: str,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    # 计算合适的块数量
    file_size = os.path.getsize(file_path)
    free_mem: float = psutil.virtual_memory().available # 单位是B
    cpu_core_num: int = psutil.cpu_count()
    safety_margin: float = 0.3
    num_chunks : int = int(file_size / (free_mem * safety_margin / cpu_core_num))
    if num_chunks < cpu_core_num:
        num_chunks = cpu_core_num
    print("expected num of chunks:" + str(num_chunks))

    with open(file_path, 'rb') as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file.seek(0)

        chunk_size = file_size // num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))