#!/usr/bin/env python3
import sys
import argparse
from amuse.ext.ekster import ekster_settings
from amuse.ext.ekster.run import main


def new_argument_parser(settings):
    "Parse command line arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        dest='settingfilename',
        default='settings.ini',
        help='settings file [settings.ini]',
    )
    parser.add_argument(
        '--setup',
        dest='setup',
        default="default",
        help='configuration setup [default]',
    )
    parser.add_argument(
        '--writesetup',
        dest='writesetup',
        default=False,
        action="store_true",
        help='write default settings file and exit',
    )
    return parser.parse_args()


if __name__ == "__main__":
    settings = ekster_settings.Settings()
    args = new_argument_parser(settings)
    if args.writesetup:
        ekster_settings.write_config(
            settings, args.settingfilename, args.setup
        )
        sys.exit()
    model = main(args, settings=settings)
