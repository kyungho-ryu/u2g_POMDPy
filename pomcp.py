#!/usr/bin/env python
from __future__ import print_function
from pomdpy import Agent
from pomdpy.solvers import POMCP, POMCPWITHNN, POMCPWITHPPO
from pomdpy.log import init_logger
from mobility import SLModel
from examples.u2g import U2GModel
from pomdpy.solvers.structure import SolverType
from pomdpy.action_selection.structure import ActionType
from DRL.structure import DRLType
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', default="U2GModel", type=str, help='Specify the env to solve')
    parser.add_argument('--solver', default='POMCP_DPW_WITH_NN', type=str,
                        help='Specify the solver to use {POMCP}')   # POMCP_DPW, POMCP_DPW_WITH_NN
    parser.add_argument('--seed', default=1993, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--use_tf', dest='use_tf', action='store_true', help='Set if using TensorFlow')
    parser.add_argument('--discount', default=0.99, type=float, help='Specify the discount factor (default=0.95)')
    parser.add_argument('--n_epochs', default=1000000, type=int, help='Num of epochs of the experiment to conduct')
    parser.add_argument('--max_steps', default=1024, type=int, help='Max num of steps per trial/episode/trajectory/epoch')
    parser.add_argument('--save', dest='save', action='store_true', help='Pickle the weights/alpha vectors')
    parser.add_argument('--test', default=10, type=int, help='Evaluate the agent every `test` epochs')
    parser.add_argument('--epsilon_start', default=1, type=float)
    parser.add_argument('--epsilon_minimum', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.95, type=float)
    parser.add_argument('--epsilon_decay_step', default=20, type=int)
    parser.add_argument('--n_sims', default=5000, type=int,
                        help='For POMCP, this is the num of MC sims to do at each belief node. '
                             'For SARSA, this is the number of rollouts to do per epoch')
    parser.add_argument('--timeout', default=3600, type=int, help='Max num of sec the experiment should run before '
                                                                  'timeout')
    parser.add_argument('--preferred_actions', dest='preferred_actions', action='store_true', help='For RockSample, '
                                                    'specify whether smart actions should be used')
    parser.add_argument('--ucb_coefficient', default=5, type=float, help='Coefficient for UCB algorithm used by MCTS')
    parser.add_argument('--n_start_states', default=100, type=int, help='Num of state particles to generate for root '
                        'belief node in MCTS')
    parser.add_argument('--min_particle_count', default=100, type=int, help='Lower bound on num of particles a belief '
                        'node can have in MCTS')
    parser.add_argument('--max_particle_count', default=1000, type=int, help='Upper bound on num of particles a belief '
                        'node can have in MCTS')
    parser.add_argument('--max_depth', default=500, type=int, help='Max depth for a DFS of the belief search tree in '
                        'MCTS')
    parser.add_argument('--max_rollout_depth', default=5, type=int, help='Max depth for a DFS of the belief search tree in '
                                                                   'MCTS')
    parser.add_argument('--action_selection_timeout', default=60, type=int, help='Max num of secs for action selection')

    # Using probabilistic trajectory prediction model
    parser.add_argument('--trajectory_prediction_type', default=1, type=int, help='')

    # Using NN
    parser.add_argument('--solver_type', type=int, help='')
    parser.add_argument('--ActionType', default=0, type=int, help='a method for action selection')
    parser.add_argument('--batch_for_NN', default=1, type=int, help='a method for action selection')
    parser.add_argument('--inner_batch_for_NN', default=32, type=int, help='a method for action selection')
    parser.add_argument('--learning_iteration_for_PPO', default=10, type=int, help='a method for action selection')
    parser.add_argument('--DRLType', default="IS_PPOModel", type=str, help='a method for action selection')
    parser.add_argument('--MaxNumIterationForBeliefNode', default=10000000, type=int, help='a method for action selection') # optimal : 30

    # Progressive Widening
    parser.add_argument('--pw_a_k', default=1, type=int, help='coefficient for progrssive widening in action')
    parser.add_argument('--pw_a_alpha', default=0.3, type=float, help='coefficient for progrssive widening in action')
    parser.add_argument('--pw_o_k', default=1, type=int, help='coefficient for progrssive widening in observation')
    parser.add_argument('--pw_o_alpha', default=0.3, type=float, help='coefficient for progrssive widening in observation')

    parser.add_argument('--grab_threshold', default=0, type=float, help='threshold for dissmilarity with nearest belief node')
    # Penalty
    parser.add_argument('--connection_penalty', default=False, type=bool, help='')
    parser.add_argument('--discovery_penalty', default=False, type=bool, help='')
    parser.add_argument('--discovery_penalty_threshold', default=0.2, type=float, help='')
    parser.add_argument('--penalty', default=-1, type=float, help='')

    parser.add_argument('--MaxActionPool', default=5, type=int, help='')


    parser.set_defaults(preferred_actions=False)
    parser.set_defaults(use_tf=False)
    parser.set_defaults(save=False)

    args = vars(parser.parse_args())

    init_logger()

    np.random.seed(args['seed'])

    if args['solver'] == "POMCP_DPW" :
        args["solver_type"] = SolverType.POMCP_DPW.value
        args["ActionType"] = ActionType.Random.value
    elif args['solver'] == "POMCP_POW" :
        args["solver_type"] = SolverType.POMCP_POW.value
        args["ActionType"] = ActionType.Random.value
    elif args['solver'] == "POMCP_DPW_WITH_NN" :
        args["solver_type"] = SolverType.POMCP_DPW_WITH_NN.value
        args["ActionType"] = ActionType.NN.value
    elif args['solver'] == "POMCP_POW_WITH_NN" :
        args["solver_type"] = SolverType.POMCP_POW_WITH_NN.value
        args["ActionType"] = ActionType.NN.value

    if args["ActionType"] == ActionType.Random.value :
        solver = POMCP
    elif args["ActionType"] == ActionType.NN.value :
        if args["DRLType"] == "IS_A2CModel" :
            solver = POMCPWITHNN
            args["DRLType"] = DRLType.IS_A2CModel.value
        elif args["DRLType"] == "IS_PPOModel" :
            solver = POMCPWITHPPO
            args["DRLType"] = DRLType.IS_PPOModel.value
        elif args["DRLType"] == "OS_A2CModel" :
            solver = POMCPWITHNN
            args["DRLType"] = DRLType.OS_A2CModel.value
            args["learning_iteration_for_PPO"] = 1
        elif args["DRLType"] == "OS_PPOModel" :
            solver = POMCPWITHPPO
            args["DRLType"] = DRLType.OS_PPOModel.value


    if args['env'] == 'U2GModel':
        env = U2GModel(args)
        agent = Agent(env, solver)
        agent.discounted_return()
    else:
        print('Unknown env {}'.format(args['env']))
