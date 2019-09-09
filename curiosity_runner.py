import argparse
from curiosity_experiment_config import CuriosityExperimentConfig
import run as large_scale_baseline


class CuriosityRunner(object):
    def __init__(self, config):
        self._config = config

    def run(self):
        # Parse the command line arguments
        _, parser = parse_args()  # Pull out the args curiosity_runner uses. We'll allow the large scale baseline to parse the rest.
        large_scale_baseline.add_all_default_args(parser)
        baseline_args = parser.parse_args()

        # Overwrite the default args with the one from the config file
        self._config.output_dir = self._config.experiment_output_dir  # To properly overwrite in the next step
        baseline_args.__dict__.update(self._config.__dict__)  # This is kind of hacky - does not guarantee overlaps happen correctly.

        # Run the experiment with the combined args
        # Technically self._config and baseline_args contain many of the same things.
        # The difference here is that baseline_args are the command line arguments, that we can optionally overwrite in the config.
        # self._config is intended to be used for things that are not currently configurable via command line.
        # This generally represents a pivot from command line to config file, as the second is more easily reproducible.
        large_scale_baseline.start_experiment(config_file_settings=self._config, **baseline_args.__dict__)


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment configuration')
    parser.add_argument('--experiment-config', required=True,
                        help='Path to the json file containing a list of experiments.')
    parser.add_argument('--output-dir', required=True,
                        help='Path to the directory to put experiments in. Experiments in this folder won''t be rerun.')

    args = parser.parse_args()
    return args, parser


if __name__ == "__main__":
    args, _ = parse_args()

    next_config = CuriosityExperimentConfig(args.experiment_config,
                                            args.output_dir).load_next_experiment()

    while next_config is not None:

        try:
            runner = CuriosityRunner(next_config)
            runner.run()
        except Exception as e:
            print("Exception: {}".format(e))
            raise e

        next_config = CuriosityExperimentConfig(args.experiment_config,
                                                args.output_dir).load_next_experiment()
