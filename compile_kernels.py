import subprocess
from pathlib import Path

def compile_metal_library():
    """Compiles .metal source files into a .metallib library."""
    
    source_dir = Path("src/dpf/metal/kernels")
    build_dir = Path("src/dpf/metal/build")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Sources
    sources = [
        "common.metal",
        "plm_reconstruct_x.metal",
        "hll_flux.metal"
    ]
    
    # Check if xcrun is available
    try:
        subprocess.run(["xcrun", "--show-sdk-path"], check=True, stdout=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Xcode Command Line Tools (xcrun) not found. Cannot compile Metal.")
        return

    # Compile each .metal to .air
    air_files = []
    for src in sources:
        src_path = source_dir / src
        air_path = build_dir / (src_path.stem + ".air")
        
        cmd = [
            "xcrun", "-sdk", "macosx", "metal",
            "-c", str(src_path),
            "-o", str(air_path)
        ]
        
        print(f"Compiling {src}...")
        try:
            subprocess.run(cmd, check=True)
            air_files.append(str(air_path))
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed for {src}: {e}")
            return

    # Link .air files into .metallib
    lib_path = build_dir / "default.metallib"
    link_cmd = [
        "xcrun", "-sdk", "macosx", "metallib",
        "-o", str(lib_path)
    ] + air_files
    
    print("Linking .metallib...")
    try:
        subprocess.run(link_cmd, check=True)
        print(f"Success! Library created at: {lib_path}")
    except subprocess.CalledProcessError as e:
        print(f"Linking failed: {e}")

if __name__ == "__main__":
    compile_metal_library()
