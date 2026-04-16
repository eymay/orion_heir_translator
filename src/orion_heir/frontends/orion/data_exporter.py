"""
Data exporter for Orion compiled models.

Exports model weights (diagonals), biases, and inputs as raw little-endian
float64 binary files for consumption by HEIR-generated Go code.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List

import numpy as np

from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters


@dataclass
class ExportedFile:
    """Metadata about a single exported binary file."""

    name: str  # e.g., "fc1_weights"
    file: str  # relative path from output_dir, e.g., "data/fc1_weights.bin"
    shape: List[int]  # shape of the exported array
    role: str  # "weights", "bias", or "input"


@dataclass
class ExportManifest:
    """Complete manifest of exported data."""

    func_name: str
    slots: int
    args: List[ExportedFile]
    input: dict
    crypto_params: dict

    def write(self, path: Path):
        data = {
            "func_name": self.func_name,
            "slots": self.slots,
            "args": [asdict(f) for f in self.args],
            "input": self.input,
            "crypto_params": self.crypto_params,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")


def _write_f64_bin(path: Path, data: np.ndarray):
    """Write a flat float64 array as raw little-endian binary."""
    data = np.ascontiguousarray(data, dtype=np.float64)
    path.write_bytes(data.tobytes())


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """Pad a 1-D float64 array with zeros to the given length."""
    if len(arr) >= length:
        return arr[:length]
    padded = np.zeros(length, dtype=np.float64)
    padded[: len(arr)] = arr
    return padded


class OrionDataExporter:
    """Exports compiled Orion model data as binary files for HEIR Go harnesses."""

    def __init__(self, scheme_params: OrionSchemeParameters):
        self.scheme_params = scheme_params
        self.slots = scheme_params.slots

    def export_model_data(
        self,
        model: Any,
        input_tensor: Any,
        output_dir: Path,
        func_name: str = "mlp",
    ) -> ExportManifest:
        """
        Export all model data (weights, biases, input) as raw float64 binary files.

        Walks model layers in registration order (same as OrionFrontend.extract_operations),
        filtering to layers with precomputed diagonals (Linear, Conv2d).

        Returns an ExportManifest describing all exported files.
        """
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Export input
        input_flat = input_tensor.detach().numpy().flatten().astype(np.float64)
        input_padded = _pad_to(input_flat, self.slots)
        _write_f64_bin(data_dir / "input.bin", input_padded)

        # Walk layers in the same order as extract_operations
        args: List[ExportedFile] = []
        for name, layer in model.named_modules():
            if not name:
                continue
            if not (hasattr(layer, "diagonals") and layer.diagonals):
                continue

            args.extend(self._export_layer(layer, name, data_dir))

        manifest = ExportManifest(
            func_name=func_name,
            slots=self.slots,
            args=args,
            input={"file": "data/input.bin", "len": self.slots},
            crypto_params=self._crypto_params_dict(),
        )
        manifest.write(output_dir / "manifest.json")

        print(f"  Exported model data to {data_dir}/")
        return manifest

    def _export_layer(self, layer: Any, name: str, data_dir: Path) -> List[ExportedFile]:
        """Export diagonals and bias for a single layer."""
        files = []

        # -- Diagonals (weights) — one file per block, in key order --
        for block_key in layer.diagonals.keys():
            block_diags = layer.diagonals[block_key]
            sorted_indices = sorted(block_diags.keys())
            flat_diags = np.zeros(len(sorted_indices) * self.slots, dtype=np.float64)
            for i, diag_idx in enumerate(sorted_indices):
                d = block_diags[diag_idx]
                if hasattr(d, "numpy"):
                    d = d.detach().cpu().numpy()
                d = np.asarray(d, dtype=np.float64)
                flat_diags[i * self.slots : (i + 1) * self.slots] = _pad_to(d, self.slots)

            block_suffix = (
                f"_block_{block_key[0]}_{block_key[1]}" if len(layer.diagonals) > 1 else ""
            )
            fname = f"{name}_weights{block_suffix}.bin"
            _write_f64_bin(data_dir / fname, flat_diags)
            files.append(
                ExportedFile(
                    name=f"{name}_weights{block_suffix}",
                    file=f"data/{fname}",
                    shape=[len(sorted_indices), self.slots],
                    role="weights",
                )
            )

        # -- Bias — use Orion's packing formulas to match the FHE slot layout --
        # Only export bias when the layer actually has one; the translator's
        # _get_linear_operations likewise only emits encode+add_plain for bias
        # when layer.bias is not None, so the counts must agree.
        if not (hasattr(layer, "bias") and layer.bias is not None):
            return files

        # The bias vector may span multiple ciphertext IDs (chunks of `slots` elements).
        # Export each chunk as a separate file so the translator can apply them correctly
        # to each row-block ciphertext in the pipelined BSGS structure.
        bias_full = self._pack_bias_full(layer)
        num_chunks = max(1, math.ceil(len(bias_full) / self.slots))

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.slots
            chunk = bias_full[start : start + self.slots]
            chunk = _pad_to(chunk, self.slots)

            chunk_suffix = f"_chunk{chunk_idx}" if num_chunks > 1 else ""
            fname = f"{name}_bias{chunk_suffix}.bin"
            _write_f64_bin(data_dir / fname, chunk)
            files.append(
                ExportedFile(
                    name=f"{name}_bias{chunk_suffix}",
                    file=f"data/{fname}",
                    shape=[self.slots],
                    role="bias",
                )
            )

        return files

    def _pack_bias_full(self, layer: Any) -> np.ndarray:
        """Return the full FHE-packed bias vector (may exceed slots for multi-ID layers)."""
        from orion.core import packing as orion_packing
        from orion.nn.linear import Linear as OrionLinear, Conv2d as OrionConv2d

        if isinstance(layer, OrionLinear):
            bias_torch = orion_packing.construct_linear_bias(layer)
        elif isinstance(layer, OrionConv2d):
            bias_torch = orion_packing.construct_conv2d_bias(layer)
        else:
            if layer.bias is not None:
                return layer.bias.detach().cpu().numpy().astype(np.float64)
            return np.zeros(self.slots, dtype=np.float64)

        return bias_torch.detach().cpu().numpy().astype(np.float64)

    def _crypto_params_dict(self) -> dict:
        """Build the crypto_params section of the manifest."""
        return {
            "logN": self.scheme_params.logN,
            "Q": list(self.scheme_params.ciphertext_modulus_chain),
            "P": list(self.scheme_params.auxiliary_modulus_chain),
            "logScale": self.scheme_params.logScale,
        }


def generate_go_wrapper(
    manifest: ExportManifest,
    output_dir: Path,
    has_bootstrapping: bool = False,
) -> None:
    """Generate model-specific Go wrapper ({func_name}_run.go) and main.go.

    All model-specific details (file paths, function name, argument arity,
    bootstrapping support) live only in the generated files so that main.go
    stays static across all models.
    """
    func_name = manifest.func_name
    arg_files = [(arg.file, arg.name) for arg in manifest.args]

    # ── Build {func_name}_run.go ─────────────────────────────────────
    imports = [
        '"encoding/binary"',
        '"fmt"',
        '"math"',
        '"os"',
        "",
        '"github.com/tuneinsight/lattigo/v6/core/rlwe"',
        '"github.com/tuneinsight/lattigo/v6/schemes/ckks"',
    ]
    if has_bootstrapping:
        # Insert before the rlwe import
        imports.insert(5, '"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"')

    imports_str = "\n".join(f"\t{imp}" for imp in imports)
    arg_files_str = "\n".join(f'\t"{file}",  // {name}' for file, name in arg_files)
    arg_call = ", ".join(f"args[{i}]" for i in range(len(arg_files)))

    if has_bootstrapping:
        configure_ret = (
            "(*bootstrapping.Evaluator, *ckks.Evaluator, ckks.Parameters, "
            "*ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)"
        )
        configure_vars = "bootstrappingEval, evaluator, params, encoder, encryptor, decryptor"
        run_sig = (
            "bootstrappingEval *bootstrapping.Evaluator, "
            "evaluator *ckks.Evaluator, params ckks.Parameters, "
            "encoder *ckks.Encoder,\n"
            "\tencryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor"
        )
        call_prefix = "bootstrappingEval, evaluator, params, encoder"
    else:
        configure_ret = (
            "(*ckks.Evaluator, ckks.Parameters, " "*ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)"
        )
        configure_vars = "evaluator, params, encoder, encryptor, decryptor"
        run_sig = (
            "evaluator *ckks.Evaluator, params ckks.Parameters, "
            "encoder *ckks.Encoder,\n"
            "\tencryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor"
        )
        call_prefix = "evaluator, params, encoder"

    run_go = (
        f"// {func_name}_run.go — AUTO-GENERATED by orion_heir. Do not edit by hand.\n"
        f"// Provides loadF64, configure(), and run() for the model-specific harness.\n"
        f"package main\n"
        f"\n"
        f"import (\n"
        f"{imports_str}\n"
        f")\n"
        f"\n"
        f'const inputFile = "{manifest.input["file"]}"\n'
        f"\n"
        f"var argFiles = []string{{\n"
        f"{arg_files_str}\n"
        f"}}\n"
        f"\n"
        f"func loadF64(path string) []float64 {{\n"
        f"\tdata, err := os.ReadFile(path)\n"
        f"\tif err != nil {{\n"
        f'\t\tpanic(fmt.Sprintf("failed to read %s: %v", path, err))\n'
        f"\t}}\n"
        f"\tn := len(data) / 8\n"
        f"\tresult := make([]float64, n)\n"
        f"\tfor i := range n {{\n"
        f"\t\tresult[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8 : (i+1)*8]))\n"
        f"\t}}\n"
        f"\treturn result\n"
        f"}}\n"
        f"\n"
        f"func configure() {configure_ret} {{\n"
        f"\treturn {func_name}__configure()\n"
        f"}}\n"
        f"\n"
        f"func run({run_sig}) []float64 {{\n"
        f"\tinputVec := loadF64(inputFile)\n"
        f"\targs := make([][]float64, len(argFiles))\n"
        f"\tfor i, f := range argFiles {{\n"
        f"\t\targs[i] = loadF64(f)\n"
        f"\t}}\n"
        f"\n"
        f"\tpt := ckks.NewPlaintext(params, params.MaxLevel())\n"
        f"\tpt.Scale = params.DefaultScale()\n"
        f"\tif err := encoder.Encode(inputVec, pt); err != nil {{\n"
        f"\t\tpanic(err)\n"
        f"\t}}\n"
        f"\tct, err := encryptor.EncryptNew(pt)\n"
        f"\tif err != nil {{\n"
        f"\t\tpanic(err)\n"
        f"\t}}\n"
        f"\n"
        f"\tresultCt := {func_name}({call_prefix}, ct, {arg_call})\n"
        f"\n"
        f"\tresultPt := decryptor.DecryptNew(resultCt)\n"
        f"\tresult := make([]float64, params.MaxSlots())\n"
        f"\tif err := encoder.Decode(resultPt, result); err != nil {{\n"
        f"\t\tpanic(err)\n"
        f"\t}}\n"
        f"\treturn result\n"
        f"}}\n"
    )

    out_path = output_dir / f"{func_name}_run.go"
    out_path.write_text(run_go)
    print(f"  Wrote {out_path}")

    # ── Build main.go ────────────────────────────────────────────────
    main_go = (
        "package main\n"
        "\n"
        "import (\n"
        '\t"encoding/json"\n'
        '\t"fmt"\n'
        '\t"time"\n'
        ")\n"
        "\n"
        "func main() {\n"
        f"\t{configure_vars} := configure()\n"
        "\tstart := time.Now()\n"
        f"\tresult := run({configure_vars})\n"
        "\telapsed := time.Since(start)\n"
        "\ttype Result struct {\n"
        '\t\tResult      []float64 `json:"result"`\n'
        '\t\tInferenceMs float64   `json:"inference_ms"`\n'
        "\t}\n"
        "\tenc, _ := json.Marshal(Result{Result: result, InferenceMs: float64(elapsed.Milliseconds())})\n"
        "\tfmt.Println(string(enc))\n"
        "}\n"
    )

    main_path = output_dir / "main.go"
    main_path.write_text(main_go)
    print(f"  Wrote {main_path}")
