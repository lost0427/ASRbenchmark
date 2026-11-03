"""Convert a GGUF's F16 tensors to BF16 while keeping F32 tensors unchanged."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--file-type", type=int, default=24,
                        help="general.file_type (24 is GGML_FTYPE_MOSTLY_BF16)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output} (pass --force)")
    try:
        import ml_dtypes
    except ImportError as error:
        raise SystemExit("Install ml-dtypes from tools/model-requirements.txt") from error

    reader = GGUFReader(str(args.input))
    arch = reader.fields["general.architecture"].contents()
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(temporary), arch)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value_type = field.types[0]
        sub_type = field.types[1] if value_type == GGUFValueType.ARRAY else None
        value = field.contents()
        if key == "general.file_type":
            value = args.file_type
        writer.add_key_value(key, value, value_type, sub_type)

    counts = {"f16": 0, "f32": 0}
    for tensor in reader.tensors:
        if tensor.tensor_type == GGMLQuantizationType.F16:
            bits = (np.asarray(tensor.data, dtype=np.float32)
                    .astype(ml_dtypes.bfloat16).view(np.uint16))
            writer.add_tensor(tensor.name, bits, raw_dtype=GGMLQuantizationType.BF16)
            counts["f16"] += 1
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            writer.add_tensor(tensor.name, np.asarray(tensor.data, dtype=np.float32),
                              raw_dtype=GGMLQuantizationType.F32)
            counts["f32"] += 1
        else:
            raise SystemExit(f"unsupported tensor type {tensor.tensor_type} for {tensor.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.replace(temporary, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output),
                      "bytes": args.output.stat().st_size, **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
