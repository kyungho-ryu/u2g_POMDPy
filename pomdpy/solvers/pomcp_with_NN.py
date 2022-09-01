from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import time, logging
import numpy as np
from pomdpy.util import console, summary
from pomdpy.action_selection import Max_Q_action, action_progWiden, structure
from pomdpy.solvers.structure import SolverType
from .belief_tree_solver import BeliefTreeSolver
from DRL.drl_model import DRLModel
from DRL.sample import Sample

module = "pomcp_with_nn"


class POMCPWITHNN(BeliefTreeSolver):

    """
    Monte-Carlo Tree Search implementation, from POMCP
    """

    # Dimensions for the fast-UCB table
    UCB_N = 10000   # total action visit count in mct
    UCB_n = 100     # each action vis count in mct

    def __init__(self, agent):
        """
        Initialize an instance of the POMCP solver
        :param agent:
        :param model:
        :return:
        """
        super(POMCPWITHNN, self).__init__(agent)
        self.logger = logging.getLogger('POMDPy.Simulation')
        self.logger.setLevel("INFO")
        self.fast_UCB = [[None for _ in range(POMCPWITHNN.UCB_n)] for _ in range(POMCPWITHNN.UCB_N)]

        state_space = self.model.get_state_space()
        state_dimension = self.model.get_state_dimension()
        action_space = self.model.get_action_space()
        action_dimension = self.model.get_action_dimension()

        self.A2CModel = DRLModel(state_dimension, state_space, action_dimension, action_space)
        self.A2CSample = Sample()

        for N in range(POMCPWITHNN.UCB_N):
            for n in range(POMCPWITHNN.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = agent.model.ucb_coefficient * np.sqrt(old_div(np.log(N + 1), n))
                    # 3.0 * root(log(n+1) / n)
                    # if the numer of visit is increase, value is decrease

    @staticmethod
    def reset(agent):
        """
        Generate a new POMCP solver

        :param agent:
        Implementation of abstract method
        """
        return POMCPWITHNN(agent)

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        """
        Look up and return the value in the UCB table corresponding to the params
        :param total_visit_count:
        :param action_map_entry_visit_count:
        :param log_n:
        :return:
        """
        assert self.fast_UCB is not None
        if total_visit_count < POMCPWITHNN.UCB_N and action_map_entry_visit_count < POMCPWITHNN.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.ucb_coefficient * np.sqrt(old_div(log_n, action_map_entry_visit_count))

    def select_eps_greedy_action(self,epoch,step, eps, start_time):
        """
        Starts off the Monte-Carlo Tree Search and returns the selected action. If the belief tree
                data structure is disabled, random rollout is used.
        """
        start = time.time()
        if self.disable_tree:   # False
            self.rollout_search(self.belief_tree_index)
        else:
            self.monte_carlo_approx(eps, start_time)

        action, best_ucb_value, best_q_value = Max_Q_action(self, self.belief_tree_index, True)
        action_selection_delay = time.time()
        self.logger.info("action selection delay : {}".format(action_selection_delay - start))
        self.model.reset_for_simulation()

        # summary.summary_simulationResult(self.model.writer, self.belief_mapping_index, step)

        return action, best_ucb_value, best_q_value

    def simulate(self, belief_node, eps, start_time):   # not use eps
        """
        :param belief_node:
        :return:
        """
        if self.model.solver_type == SolverType.POMCP_DPW_WITH_NN.value:
            return self.POCMP_DPW(belief_node, 0, start_time)
        elif self.model.solver_type == SolverType.POMCP_POW_WITH_NN.value:
            return self.POCMP_POW(belief_node, 0, start_time)
        else:
            return self.traverse(belief_node, 0, start_time)

    def POCMP_DPW(self, belief_node, tree_depth, start_time):
        delayed_reward = 0

        # choice random state from particles every simulation
        state = belief_node.sample_particle()

        if tree_depth == 0 :
            self.model.reset_for_simulation()

        self.logger.debug("depth : {} ================================================================".format(tree_depth))
        self.logger.debug("state : {}".format(state.to_string()))

        # Search horizon reached
        if tree_depth >= self.model.max_depth:  # default = 100
            console(4, module, "Search horizon reached")
            return 0

        # use UCB table
        temp_action = self.A2CModel.get_action(np.array(state.as_DRL_state()))

        action, C_A, N_A, actionStatus = action_progWiden(self, belief_node, temp_action, self.model.pw_a_k, self.model.pw_a_alpha)
        self.logger.debug("C,N: [{},{}] action: {}".format(C_A, N_A, action.to_string()))
        self.logger.debug(actionStatus)

        # update visit count of child belief node
        N_O = belief_node.get_visit_count_observation(action)
        C_O = belief_node.get_number_observation(action)

        if C_O <= self.model.pw_o_k * (N_O**self.model.pw_o_alpha) :
            reward, delayed_reward = self.create_new_step_for_dpw(
                belief_node, tree_depth, state, action, start_time, delayed_reward
            )
        else :
            reward, delayed_reward = self.select_existing_step_for_dpw(
                belief_node, tree_depth, state, action, start_time, delayed_reward
            )

        action_mapping_entry = belief_node.action_map.get_entry(action.UAV_deployment)

        td_target = reward + self.model.discount * delayed_reward
        NumSample = self.A2CSample.add_sample(state.as_DRL_state(), action.UAV_deployment, td_target)

        if NumSample == self.model.batch_for_NN :
            advantage, value, loss_entropy, loss, step = self.A2CModel.update(self.A2CSample)
            self.A2CSample.reset_sample()

            summary.summary_NNResult(self.model.writer, advantage, value, loss_entropy, loss, step)

        q_value = action_mapping_entry.mean_q_value
        q_value += td_target - q_value

        action_mapping_entry.update_visit_count(1)
        belief_node.update_visit_count_observation(action, 1)

        action_mapping_entry.update_q_value(q_value)

        self.logger.debug("update total count of observation : {}".format(belief_node.get_visit_count_observation(action)))

        return td_target

    def create_new_step_for_dpw(self, belief_node, tree_depth, state, action, start_time, delayed_reward):
        self.logger.debug("create new step")
        step_result, is_legal = self.model.generate_step(state, action)
        self.logger.debug("observation : {}/{}".format(len(step_result.observation.observed_gmu_status),
                                                      step_result.observation.observed_gmu_status))

        # child belief node = observation
        child_belief_node, ChildStatus = belief_node.child(action, step_result.observation)
        self.logger.debug("Status : {}".format(ChildStatus))

        added = False
        if child_belief_node is None and belief_node.action_map.total_visit_count >= 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)
            if step_result.reward == self.model.penalty :
                belief_node.penalty_count +=1

        child_belief_node.add_particle(step_result.next_state, 1)

        if not step_result.is_terminal or not is_legal:
            tree_depth += 1
            if added:
                delayed_reward = self.A2CModel.get_value(np.array(step_result.next_state.as_DRL_state()))
            else:
                delayed_reward = self.POCMP_DPW(child_belief_node, tree_depth, start_time)
            tree_depth -= 1
            console(4, module, "Reached terminal state.")

        belief_node.update_visit_count_specific_observation(action, step_result.observation, 1)

        self.logger.debug("update visit count of specific observation : {}".format(
            belief_node.get_visit_count_specific_observation(action, step_result.observation)
        ))

        return step_result.reward, delayed_reward

    def select_existing_step_for_dpw(self, belief_node, tree_depth, state, action, start_time, delayed_reward):
        self.logger.debug("select existing step")
        obsEntries = belief_node.get_child_obs_entries(action)

        max = 0
        selected_entry = 0
        for entry in obsEntries :
            visit_count = entry.get_visit_count()
            if max < visit_count :
                max = visit_count
                selected_entry = entry

        selected_obs = selected_entry.observation
        selected_next_state = selected_entry.child_node.sample_particle()

        reward = self.model.get_reward(state, action, selected_next_state)

        self.logger.debug("the number of particles : {}".format(selected_entry.child_node.get_num_total_particle()))
        self.logger.debug("selected observation : {}".format(selected_obs.observed_gmu_status))
        self.logger.debug("selected next state : \nUav:{} \nGMU: {}".format(selected_next_state.uav_position,
                                                                            selected_next_state.gmu_position))
        self.logger.debug("reward : {}".format(reward))

        self.model.update_gmu_for_simulation()
        if not self.model.is_terminal():
            tree_depth += 1
            delayed_reward = self.POCMP_DPW(selected_entry.child_node, tree_depth, start_time)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")

        return reward, delayed_reward

    def POCMP_POW(self, belief_node, tree_depth, start_time):
        delayed_reward = 0

        # choice random state from particles every simulation
        state = belief_node.sample_particle()
        if tree_depth == 0:
            self.model.reset_for_simulation()

        self.logger.debug("depth : {} ================================================================".format(tree_depth))
        self.logger.debug("state : {}".format(state.to_string()))

        # Search horizon reached
        if tree_depth >= self.model.max_depth:  # default = 100
            console(4, module, "Search horizon reached")
            return 0

        # use UCB table
        temp_action = self.A2CModel.get_action(np.array(state.as_DRL_state()))

        action, C_A, N_A, actionStatus = action_progWiden(self, belief_node, temp_action, self.model.pw_a_k, self.model.pw_a_alpha)
        self.logger.debug("C,N: [{},{}] action: {}".format(C_A, N_A, action.to_string()))
        self.logger.debug(actionStatus)

        step_result, is_legal = self.model.generate_step(state, action)

        # update visit count of child belief node
        N_O = belief_node.get_visit_count_observation(action)
        C_O = belief_node.get_number_observation(action)

        if C_O <= self.model.pw_o_k * (N_O**self.model.pw_o_alpha) :
            observation = step_result.observation
            flg = True
        else :
            observation = self.select_existing_observation(belief_node, action)
            flg = False

        child_belief_node, ChildStatus = belief_node.child(action, observation)
        self.logger.debug("Status : {}".format(ChildStatus))

        added = False
        if child_belief_node is None and belief_node.action_map.total_visit_count >= 0:
            child_belief_node, added = belief_node.create_or_get_child(action, observation)
            if step_result.reward == self.model.penalty :
                belief_node.penalty_count +=1

        child_belief_node.add_particle(step_result.next_state, 1)

        if not step_result.is_terminal or not is_legal:
            tree_depth += 1
            if added:
                reward = step_result.reward
                delayed_reward = self.A2CModel.get_value(np.array(step_result.next_state.as_DRL_state()))
            else:
                reward, delayed_reward = self.select_existing_step_for_pow(
                    belief_node, tree_depth, state, action, observation, start_time, delayed_reward
                )
                # delayed_reward = self.POCMP_POW(child_belief_node, tree_depth, start_time)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")
            reward = step_result.reward

        if flg :
            belief_node.update_visit_count_specific_observation(action, step_result.observation, 1)

        self.logger.debug("update visit count of specific observation : {}".format(
            belief_node.get_visit_count_specific_observation(action, step_result.observation)
        ))

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.UAV_deployment)

        q_value = action_mapping_entry.mean_q_value

        # off-policy Q learning update rule
        q_value += (reward + (self.model.discount * delayed_reward) - q_value)

        action_mapping_entry.update_visit_count(1)
        belief_node.update_visit_count_observation(action, 1)

        self.logger.debug("update total count of observation : {}".format(belief_node.get_visit_count_observation(action)))

        action_mapping_entry.update_q_value(q_value)
        self.logger.debug(" Q value : {}".format(q_value))
        # Add RAVE ?
        return q_value


    def select_existing_observation(self, belief_node, action):
        self.logger.debug("select existing step")
        obsEntries = belief_node.get_child_obs_entries(action)

        max = 0
        selected_entry = 0
        for entry in obsEntries :
            visit_count = entry.get_visit_count()
            if max < visit_count :
                max = visit_count
                selected_entry = entry

        selected_obs = selected_entry.observation
        self.logger.debug("selected observation : {}".format(selected_obs.observed_gmu_status))

        return selected_obs

    def select_existing_step_for_pow(self, belief_node, tree_depth, state, action, obs, start_time, delayed_reward):
        self.logger.debug("select existing step")
        child_node = belief_node.get_child(action, obs)

        selected_next_state = child_node.sample_particle()
        reward = self.model.get_reward(state, action, selected_next_state)

        self.logger.debug("selected next state : \nUav:{} \nGMU: {}".format(selected_next_state.uav_position, selected_next_state.gmu_position))
        self.logger.debug("reward : {}".format(reward))

        # self.model.update_gmu_for_simulation()
        if not self.model.is_terminal() :
            tree_depth +=1
            delayed_reward = self.POCMP_POW(child_node, tree_depth, start_time)
            tree_depth -=1
        else:
            console(4, module, "Reached terminal state.")

        return reward, delayed_reward


