"""
File for testing the algorithms on various environments, such as
the Furuta pendulum swing-up ones.

NOTE: The word epoch is commonly used to refer to the number of
parameter updates performed throughout this code base.

Last edit: 2022-06-21
By: dansah
"""

from deps.spinningup.dansah_custom import test_policy
from deps.spinningup.dansah_custom import plot

import pathlib
from result_maker import make_html

import torch.nn as nn

import numpy as np
import os
import pickle

BASE_DIR = os.path.join('.', 'out%s' % os.sep)  # The base directory for storing the output of the algorithms.
WORK_DIR = pathlib.Path().resolve()


def make_env_pbrs3():
    """
    Creates a Furuta Pendulum environment (swing-up) similar to PBRS V2, 
    but with a big neative reward on early termination.
    """
    from custom_envs.furuta_swing_up_pbrs_v3 import FurutaPendulumEnvPBRS_V3
    return FurutaPendulumEnvPBRS_V3()


#########################
# Misc helper functions #
#########################
def annonuce_message(message):
    """
    Prints a message in a fancy way.
    """
    print("------------------------------------")
    print("----> %s <----" % (message))
    print("------------------------------------")

def heads_up_message(message):
    """
    Prints a message in a fancy way.
    """
    print("----> %s <----" % (message))

def get_output_dir(alg_name, arch_name, env_name, seed):
    """
    Returns the approriate output directory name relative to the
    project root for the given experiment.
    """
    return os.path.join(BASE_DIR + env_name, alg_name, arch_name, "seed" + str(seed) + os.sep)

def get_diagram_filepath(env_name, arch_name):
    """
    Returns the full filepath that the diagram should use given
    the provided arguments.
    """
    return os.path.join(WORK_DIR, "out", "res", env_name, arch_name + ".svg")

def get_res_filepath():
    """
    Returns an appropriate full filepath for the html-file
    with the results.
    """
    return os.path.join(WORK_DIR, "out", "res.html")


def get_table_data_filepath():
    """
    Returns the relative filepath at which the
    table containing evaluation data for Furuta Pendulum
    environments should be stored.
    """
    return os.path.join(BASE_DIR, "eval_table")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices={"train", "eval"})
    argparser.add_argument("-s", "--seed", type=int, help="The seed to use", default=0)
    argparser.add_argument("-i", "--inter", type=int, help="The number of interactions to target", default=500_000)
    argparser.add_argument("-p", "--plot", action="store_true", help="Produce plots")
    args = argparser.parse_args()

    # env = "furuta_pbrs3"
    env_fn = make_env_pbrs3
    max_ep_len = 501

    output_dir = get_output_dir("ddpg", "256_128_relu", "furuta_pbrs3", args.seed)
    if args.mode == "train":
        from deps.spinningup.dansah_custom.ddpg import ddpg

        ac_kwargs = {
            "hidden_sizes": [256, 128],
            "activation": nn.ReLU
        }
        logger_kwargs = {
            "output_dir": output_dir,
            "exp_name": "experiment_test0_" + output_dir,
            "log_frequency": 2000
        }

        ddpg(env_fn=env_fn,
             ac_kwargs=ac_kwargs,
             max_ep_len=max_ep_len,
             steps_per_epoch=256, 
             min_env_interactions=args.inter,
             logger_kwargs=logger_kwargs,
             seed=args.seed,
             start_steps=10000,
             perform_eval=False,
             pi_lr=5e-4,
             q_lr=5e-4)

        if args.plot:
            # plot
            import webbrowser
            fname = get_diagram_filepath("furuta_pbrs3", "256_128_relu")
            res_maker_dict = {
                "furuta_pbrs3": {
                    "256_128_relu": {
                        "diagrams": {"Performance": fname}
                    }
                }
            }
            annonuce_message("Analyzing metrics for environment furuta_pbrs3")
            plot.make_plots([output_dir], legend=["ddpg"], xaxis='TotalEnvInteracts', values=["Performance"], count=False, 
                            smooth=1, select=None, exclude=None, estimator='mean', fname=fname)
            eval_table = None
            try:
                with open(get_table_data_filepath(), "rb") as f:
                    eval_table = pickle.load(f)
                    heads_up_message("Loaded eval_table: %s" % eval_table['_debug'])
            except:
                print("NOTE: Could not load evaluation table data. Run an evaluation on the Furuta environments to generate it.")
            # Process eval. table data.
            if eval_table is not None:
                for env_name in eval_table:
                    if env_name.startswith('_'):
                        continue
                    for arch_name in eval_table[env_name]:
                        eval_2d_table_data = [["Alg", "Mean", "Std"]]
                        for alg_name in eval_table[env_name][arch_name]:
                            eval_data = eval_table[env_name][arch_name][alg_name]
                            eval_2d_table_data.append([alg_name, np.mean(eval_data), np.std(eval_data)])
                        try:
                            res_maker_dict[env_name][arch_name]["tables"] = {"Independent evaluation data": eval_2d_table_data}
                        except:
                            print("ERROR: Failed to add evaluation table for %s %s: Missing entry in result dict." % 
                                (env_name, arch_name))
            res_file = get_res_filepath()
            make_html(res_file, res_maker_dict)
            print(res_file)
            assert webbrowser.get().open("file://" + res_file)
    else:
        # eval
        from custom_envs.furuta_swing_up_eval import FurutaPendulumEnvEvalWrapper
        from deps.visualizer.visualizer import plot_animated

        env, get_action = test_policy.load_policy_and_env(output_dir,
                                                          'last',
                                                          deterministic=False,
                                                          force_disc=False)
        env = FurutaPendulumEnvEvalWrapper(env=env, seed=args.seed, qube2=False)
        env.collect_data()

        test_policy.run_policy(env, get_action, max_ep_len=max_ep_len, num_episodes=5, render=False)
        collected_data = env.get_data()
        env.close()
        independent_furuta_data = env.get_internal_rewards()
    
        assert collected_data is not None, "No data was collected for rendering!"
        name = "%s - %s" % ("ddpg", "256_128_relu")
        best_episode_idx = np.argmax(independent_furuta_data) # Visualize the best episode
        plot_data = collected_data[best_episode_idx]
        plot_animated(phis=plot_data["phis"], thetas=plot_data["thetas"], l_arm=1.0, l_pendulum=1.0, 
                      frame_rate=50, name=name, save_as=None)

        print("Independently defined evaluation data:", independent_furuta_data)
        eval_table = {"furuta_pbrs3": {"256_128_relu": {"ddpg": independent_furuta_data}}}
        import datetime
        eval_table['_debug'] = "eval_table created at %s, using seeds %s." % (datetime.datetime.now(), [args.seed])
        with open(get_table_data_filepath(), "wb") as f:
            pickle.dump(eval_table, f)
