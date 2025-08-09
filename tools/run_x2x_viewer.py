from argparse import ArgumentParser
from pathlib import Path

from x2x.viewer.x2x_directory_server import create_app

if __name__ == "__main__":
    parser = ArgumentParser(usage="%(prog)s [options] [SLIDE-DIRECTORY]")
    parser.add_argument(
        "-B",
        "--ignore-bounds",
        dest="DEEPZOOM_LIMIT_BOUNDS",
        default=True,
        action="store_false",
        help="display entire scan area",
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE", type=Path, dest="config", help="config file"
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="DEBUG",
        action="store_true",
        help="run in debugging mode (insecure)",
    )
    parser.add_argument(
        "-e",
        "--overlap",
        metavar="PIXELS",
        dest="DEEPZOOM_OVERLAP",
        type=int,
        help="overlap of adjacent tiles [1]",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="DEEPZOOM_FORMAT",
        choices=["jpeg", "png"],
        help="image format for tiles [jpeg]",
    )
    parser.add_argument(
        "-l",
        "--listen",
        metavar="ADDRESS",
        dest="host",
        default="127.0.0.1",
        help="address to listen on [127.0.0.1]",
    )
    parser.add_argument(
        "-p",
        "--port",
        metavar="PORT",
        dest="port",
        type=int,
        default=5000,
        help="port to listen on [5000]",
    )
    parser.add_argument(
        "-Q",
        "--quality",
        metavar="QUALITY",
        dest="DEEPZOOM_TILE_QUALITY",
        type=int,
        help="JPEG compression quality [75]",
    )
    parser.add_argument(
        "-s",
        "--size",
        metavar="PIXELS",
        dest="DEEPZOOM_TILE_SIZE",
        type=int,
        help="tile size [254]",
    )
    parser.add_argument(
        "SLIDE_DIR",
        metavar="SLIDE-DIRECTORY",
        type=Path,
        nargs="?",
        help="slide directory",
    )

    args = parser.parse_args()
    config = {}
    config_file = args.config

    # Set only those settings specified on the command line
    for k in dir(args):
        v = getattr(args, k)
        if not k.startswith("_") and v is not None:
            config[k] = v
    app = create_app(config, config_file)

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.DEBUG,
        use_reloader=True,
    )
