from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import yaml
from model.pytorch.supervisor_ssl import GTSSupervisor
from lib.utils import load_graph_data


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config["mr"] = args.mr
        supervisor_config["mode"] = args.mode
        save_adj_name = args.config_filename[11:-5]
        supervisor = GTSSupervisor(save_adj_name, temperature=args.temperature, **supervisor_config)
        result = supervisor.train()
        try:
            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
            best_checkpoint_dir = best_trial.checkpoint.value
        except Exception:
            pass
        try:
            from google.colab import files
            files.download(os.path.join(best_checkpoint_dir, "checkpoint"))
        except Exception:
            print(f'Best checkpoint={str(os.path.join(best_checkpoint_dir, "checkpoint"))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/para_la.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
    parser.add_argument('--mr', default=0.3, type=float, help='mask ratio')
    parser.add_argument('--mode', default="spatial", type=float, help='mask method')
    args = parser.parse_args()
    main(args)
