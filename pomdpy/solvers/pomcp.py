from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import time, logging
import numpy as np
from pomdpy.util import console, summary
from pomdpy.action_selection import ucb_action, action_progWiden
from .belief_tree_solver import BeliefTreeSolver

module = "pomcp"


class POMCP(BeliefTreeSolver):

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
        super(POMCP, self).__init__(agent)
        self.logger = logging.getLogger('POMDPy.Simulation')
        self.logger.setLevel("INFO")
        self.fast_UCB = [[None for _ in range(POMCP.UCB_n)] for _ in range(POMCP.UCB_N)]

        for N in range(POMCP.UCB_N):
            for n in range(POMCP.UCB_n):
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
        return POMCP(agent)

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        """
        Look up and return the value in the UCB table corresponding to the params
        :param total_visit_count:
        :param action_map_entry_visit_count:
        :param log_n:
        :return:
        """
        assert self.fast_UCB is not None
        if total_visit_count < POMCP.UCB_N and action_map_entry_visit_count < POMCP.UCB_n:
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
        if self.disable_tree:   # False
            self.rollout_search(self.belief_tree_index)
        else:
            self.monte_carlo_approx(eps, start_time)

        summary.summary_simulationResult(self.model.writer, self.belief_tree_index, epoch+step)

        return ucb_action(self, self.belief_tree_index, True)

    def simulate(self, belief_node, eps, start_time):   # not use eps
        """
        :param belief_node:
        :return:
        """
        if self.model.DPW :
            return self.POCMP_DPW(belief_node, 0, start_time)
        else :
            return self.traverse(belief_node, 0, start_time)
    def POCMP_DPW(self, belief_node, tree_depth, start_time):
        delayed_reward = 0

        # choice random state from particles every simulation
        state = belief_node.sample_particle()
        self.logger.info("depth : {} ================================================================".format(tree_depth))
        self.logger.debug("state : {}".format(state.to_string()))

        # Time expired
        if time.time() - start_time > self.model.action_selection_timeout:
            console(4, module, "action selection timeout")
            return 0

        # Search horizon reached
        if tree_depth >= self.model.max_depth:  # default = 100
            console(4, module, "Search horizon reached")
            return 0

        # use UCB table
        if self.model.NN:
            pass
        else:
            temp_action = self.model.sample_random_actions()

        action, C_A, N_A, actionStatus = action_progWiden(self, belief_node, temp_action, self.model.pw_a_k, self.model.pw_a_alpha)
        self.logger.debug("C,N: [{},{}] action: {}".format(C_A, N_A, action.to_string()))
        self.logger.info(actionStatus)

        # update visit count of child belief node
        N_O = belief_node.get_visit_count_observation(action)
        C_O = belief_node.get_number_observation(action)

        if C_O <= self.model.pw_o_k * (N_O**self.model.pw_o_alpha) :
            reward, delayed_reward = self.create_new_step(
                belief_node, tree_depth, state, action, start_time, delayed_reward
            )
        else :
            reward, delayed_reward = self.select_existing_step(
                belief_node, tree_depth, action, start_time, delayed_reward
            )

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.UAV_deployment)

        q_value = action_mapping_entry.mean_q_value

        # off-policy Q learning update rule
        q_value += (reward + (self.model.discount * delayed_reward) - q_value)

        action_mapping_entry.update_visit_count(1)
        belief_node.update_visit_count_observation(action, 1)

        self.logger.info("update total count of observation : {}".format(belief_node.get_visit_count_observation(action)))

        action_mapping_entry.update_q_value(q_value)
        self.logger.debug(" Q value : {}".format(q_value))
        # Add RAVE ?
        return q_value

    def create_new_step(self, belief_node, tree_depth, state, action, start_time, delayed_reward):
        self.logger.info("create new step")
        step_result, is_legal = self.model.generate_step(state, action)
        self.logger.debug("observation : {}/{}".format(len(step_result.observation.observed_gmu_status),
                                                      step_result.observation.observed_gmu_status))

        # child belief node = observation
        child_belief_node, ChildStatus = belief_node.child(action, step_result.observation)
        self.logger.info("Status : {}".format(ChildStatus))

        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count >= 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal or not is_legal:
            tree_depth += 1
            if child_belief_node is not None:
                # Add S' to the new belief node
                # Add a state particle with the new state
                if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
                    child_belief_node.state_particles.append(step_result.next_state)
                delayed_reward = self.POCMP_DPW(child_belief_node, tree_depth, start_time)
            else:
                delayed_reward = self.rollout(belief_node)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")

        belief_node.update_visit_count_specific_observation(action, step_result.observation, 1)

        self.logger.info("update visit count of specific observation : {}".format(
            belief_node.get_visit_count_specific_observation(action, step_result.observation)
        ))

        return step_result.reward, delayed_reward

    def select_existing_step(self, belief_node, tree_depth, action, start_time, delayed_reward):
        self.logger.info("select existing step")
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

        reward = self.model.get_reward(selected_next_state)

        self.logger.info("the number of particles : {}".format(len(selected_entry.child_node.state_particles)))
        self.logger.debug("selected observation : {}".format(selected_obs.observed_gmu_status))
        self.logger.debug("selected next state : \nUav:{} \nGMU: {}".format(selected_next_state.uav_position, selected_next_state.gmu_position))
        self.logger.debug("reward : {}".format(reward))

        if not self.model.is_terminal(selected_next_state) :
            tree_depth +=1
            delayed_reward = self.POCMP_DPW(selected_entry.child_node, tree_depth, start_time)
            tree_depth -=1
        else:
            console(4, module, "Reached terminal state.")

        return reward, delayed_reward

    def traverse(self, belief_node, tree_depth, start_time):
        delayed_reward = 0

        state = belief_node.sample_particle()

        # Time expired
        if time.time() - start_time > self.model.action_selection_timeout:
            console(4, module, "action selection timeout")
            return 0

        action = ucb_action(self, belief_node, False)

        # Search horizon reached
        if tree_depth >= self.model.max_depth:
            console(4, module, "Search horizon reached")
            return 0

        step_result, is_legal = self.model.generate_step(state, action)

        child_belief_node = belief_node.child(action, step_result.observation)
        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count > 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal or not is_legal:
            tree_depth += 1
            if child_belief_node is not None:
                # Add S' to the new belief node
                # Add a state particle with the new state
                if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
                    child_belief_node.state_particles.append(step_result.next_state)
                delayed_reward = self.traverse(child_belief_node, tree_depth, start_time)
            else:
                delayed_reward = self.rollout(belief_node)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.bin_number)

        q_value = action_mapping_entry.mean_q_value

        # off-policy Q learning update rule
        q_value += (step_result.reward + (self.model.discount * delayed_reward) - q_value)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        # Add RAVE ?
        return q_value

