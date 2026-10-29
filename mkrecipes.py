import pathlib, subprocess

base = subprocess.run(
    ["git", "show", "e2b15d8:recipe.yaml"], capture_output=True, text=True, check=True
).stdout

trace = base.replace('  patches: ""', '  patches: mbx-warmup-trace')
assert trace != base
pathlib.Path("recipe.yaml.bisect-v54-trace").write_text(trace)

noskinny = base.replace('  MBX_SKINNY_GEMM_SM12X: "1"', '  MBX_SKINNY_GEMM_SM12X: "0"')
assert noskinny != base
pathlib.Path("recipe.yaml.bisect-v54-noskinny").write_text(noskinny)

print("written")
