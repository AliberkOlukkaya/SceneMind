"""Create a tiny synthetic video fixture; no user media or datasets required."""
import subprocess
from pathlib import Path

import imageio_ffmpeg

path = Path(__file__).resolve().parents[1] / "data" / "e2e-fixture.mp4"
path.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi",
                "-i", "color=red:s=640x360:r=10:d=5", "-f", "lavfi", "-i",
                "color=blue:s=640x360:r=10:d=5", "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264", str(path)], check=True)
