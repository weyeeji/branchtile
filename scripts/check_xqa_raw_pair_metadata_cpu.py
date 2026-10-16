"""CPU-only checks for pair register-row mapping and exact split continuation."""
import json


def main():
    tile_checks = 0
    for splits in (1, 2, 3, 10, 21):
        for total in range(1, 60):
            for cut in range(total + 1):
                for split in range(splits):
                    start = split + max(0, (cut - split + splits - 1) // splits) * splits
                    original = list(range(split, total, splits))
                    resumed = list(range(split, min(cut, total), splits)) + list(range(start, total, splits))
                    assert resumed == original
                    tile_checks += 1
    registers = 0
    for member in (0, 1):
        for lane in range(32):
            for n in range(8):
                for j in range(2):
                    native_register = 4 * n + j
                    packed_register = native_register + 2 * member
                    native_row = lane // 4
                    packed_row = lane // 4 + 8 * ((packed_register % 4) // 2)
                    native_col = 8 * n + 2 * (lane % 4) + j
                    packed_col = 8 * (packed_register // 4) + 2 * (lane % 4) + packed_register % 2
                    assert packed_row == native_row + 8 * member
                    assert packed_col == native_col
                    registers += 1
        for row in range(8):
            assert 0 <= row + 8 * member < 16
    print(json.dumps({"status": "passed", "tile_partition_checks": tile_checks,
                      "active_accumulator_mapping_checks": registers, "gpu_started": False}))


if __name__ == "__main__":
    main()
