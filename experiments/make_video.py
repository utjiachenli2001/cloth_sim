import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from experiments.utils.ffmpeg import make_video


def main():
    parser = argparse.ArgumentParser(description='Create mp4 video from a demo output folder.')
    parser.add_argument('folder', type=Path, help='Path to output folder, e.g. log/experiments/output_demo/20260305-212500')
    parser.add_argument('--fps', type=int, default=30, help='Output video frame rate (default: 30)')
    parser.add_argument('--output', type=Path, default=None, help='Output video path (default: <folder>/video.mp4)')
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.exists():
        print(f'Error: folder not found: {folder}')
        sys.exit(1)

    video_path = args.output if args.output else folder / 'video.mp4'

    make_video(folder, video_path, image_pattern='%06d.png', frame_rate=args.fps)
    print(f'Saved video to {video_path}')


if __name__ == '__main__':
    main()
