from builtins import str
from builtins import range
import time, random, logging
import abc
from pomdpy.util import console, mapping
from pomdpy.pomdp.belief_tree import BeliefTree
from pomdpy.action_selection import structure
from pomdpy.solvers import Solver
from tqdm import tqdm
import io
import numpy as np

module = "BeliefTreeSolver"


class BeliefTreeSolver(Solver):
    """
    All POMDP solvers must implement the abstract methods specified below
    Ex. See POMCP (Monte-Carlo Tree Search)

    Provides a belief search tree and supports on- and off-policy methods
    """
    def __init__(self, agent):
        super(BeliefTreeSolver, self).__init__(agent)
        # The agent owns Histories, the collection of History Sequences.
        # There is one sequence per run of the algorithm
        self.history = agent.histories.create_sequence()
        # flag for determining whether the solver is an on/off-policy learning algorithm
        self.disable_tree = False

        self.belief_tree = BeliefTree(agent)
        # Initialize the Belief Tree
        self.belief_tree.reset()
        self.belief_tree.initialize()

        # generate state particles for root node belief state estimation
        # This is for simulation
        self.model.reset_for_epoch()
        for i in range(self.model.n_start_states):  # default = 2000
            particle = self.model.sample_an_init_state()    # create random rock state
            self.belief_tree.root.add_particle(particle, 0)
        self.belief_tree.root.set_create_current_particle(True)
        self.belief_tree_index = self.belief_tree.root.copy()
    def monte_carlo_approx(self, eps, start_time):
        """
        Approximate Q(b, pi(b)) via monte carlo simulations, where b is the belief node pointed to by
        the belief tree index, and pi(b) is the action selected by the current behavior policy. For SARSA, this is
        simply pi(b) where pi is induced by current Q values. For Q-Learning, this is max_a Q(b',a')

        Does not advance the policy index
        :param eps
        :param start_time
        :return:
        """
        interval = int(self.model.n_sims / 2)
        pbar = tqdm(range(self.model.n_sims), ncols=70, miniters=interval)
        for _ in pbar: # default = 500
        # for i in range(self.model.n_sims):
            # Reset the Simulator
            self.simulate(self.belief_tree_index, eps, start_time)

    @abc.abstractmethod
    def simulate(self, belief, eps, start_time, prior_state_key):
        """
        Does a monte-carlo simulation from "belief" to approximate Q(b, pi(b))
        :param belief
        :param eps
        :param start_time
        :return:
        """

    @abc.abstractmethod
    def select_eps_greedy_action(self,epoch,step, eps, start_time, prior_state_key):
        """
        Call methods specific to the implementation of the solver
        to select an action
        :param eps
        :param start_time
        :return:
        """

    def prune(self, belief_node):
        """
        Prune the siblings of the chosen belief node and
        set that node as the new "root"
        :param belief_node: node whose siblings will be removed
        :return:
        """
        start_time = time.time()
        self.belief_tree.prune_siblings(belief_node)
        elapsed = time.time() - start_time
        console(3, module, "Time spent pruning = " + str(elapsed) + " seconds")

    def rollout_search(self, belief_node):
        """
        At each node, examine all legal actions and choose the actions with
        the highest evaluation
        :return:
        """
        legal_actions = belief_node.data.generate_legal_actions()
        # rollout each action once
        for i in range(legal_actions.__len__()):
            state = belief_node.sample_particle()
            action = legal_actions[i % legal_actions.__len__()]

            # model.generate_step casts the variable action from an int to the proper DiscreteAction subclass type
            step_result, is_legal = self.model.generate_step(state, action)

            if not step_result.is_terminal:
                child_node, added = belief_node.create_or_get_child(step_result.action, step_result.observation)
                child_node.state_particles.append(step_result.next_state)
                delayed_reward = self.rollout(child_node)
            else:
                delayed_reward = 0

            action_mapping_entry = belief_node.action_map.get_entry(step_result.action.bin_number)

            q_value = action_mapping_entry.mean_q_value

            # Random policy
            q_value += (step_result.reward + self.model.discount * delayed_reward - q_value)

            action_mapping_entry.update_visit_count(1)
            action_mapping_entry.update_q_value(q_value)

    def rollout(self, belief_node, action_method, prior_state_key):
        """
        Iterative random rollout search to finish expanding the episode starting at belief_node
        :param belief_node:
        :return:
        """
        state = belief_node.sample_particle()
        is_terminal = False
        discounted_reward_sum = 0.0
        discount = 1.0
        num_steps = 0
        while num_steps < self.model.max_depth and not is_terminal:
            if action_method == structure.ActionType.NN.value:
                pass
            elif action_method == structure.ActionType.Random.value:
                legal_action = self.model.sample_random_actions([])
            elif action_method == structure.ActionType.Near.value:
                legal_action = self.model.sample_near_actions(state.uav_position)

            step_result, is_legal = self.model.generate_step(state, legal_action)
            is_terminal = step_result.is_terminal
            discounted_reward_sum += step_result.reward * discount
            discount *= self.model.discount
            # advance to next state
            state = step_result.next_state

            num_steps += 1
        return discounted_reward_sum

    def update(self, state, step_result, prune=True):
        """
        Feed back the step result, updating the belief_tree,
        extending the history, updating particle sets, etc

        Advance the policy index to point to the next belief node in the episode

        :return:
        """
        # Update the Simulator with the Step Result
        # This is important in case there are certain actions that change the state of the simulator
        result = 0

        child_belief_node = self.belief_tree_index.get_child(step_result.action, step_result.observation)
        dissimilarity = 0
        # If the child_belief_node is None because the step result randomly produced a different observation,
        # grab any of the beliefs extending from the belief node's action node
        if child_belief_node is None:
            action_node = self.belief_tree_index.action_map.get_action_node(step_result.action)
            if action_node is None:
                # I grabbed a child belief node that doesn't have an action node. Use rollout from here on out.
                console(2, module, "Reached branch with no leaf nodes, using random rollout to finish the episode")
                self.disable_tree = True
                return

            # child_belief_node = self.grab_nearest_belief_node(action_node)
            child_belief_node, dissimilarity = self.grab_nearest_belief_node(action_node, step_result.observation)
            if child_belief_node is None:
                child_belief_node = self.create_child(action_node, step_result.observation)
                result = 1
            else:
                result = 2

        # If the new root does not yet have the max possible number of particles add some more
        if not child_belief_node.get_create_current_particle() :
            for i in range(self.model.min_particle_count) :
                # Generate particles for the new root node
                particle = self.model.generate_particles()
                child_belief_node.add_particle(particle, 0)

            child_belief_node.set_create_current_particle(True)

        # Failed to continue search- ran out of particles
        if child_belief_node is None:
            console(1, module, "Couldn't refill particles, must use random rollout to finish episode")
            self.disable_tree = True
            return

        self.belief_tree_index = child_belief_node

        if prune:
            self.prune(self.belief_tree_index)


        return result, dissimilarity

    def grab_nearest_belief_node(self, action_node, new_observation):
        obs_mapping_entries = list(action_node.observation_map.child_map.values())

        min = np.inf
        candidate_entry = []
        for entry in obs_mapping_entries:
            if entry.status_for_grabObs :
                return entry.child_node, entry.observation.check_dissimilarity(new_observation)

            if entry.child_node is not None:
                dissimilarity = entry.observation.check_dissimilarity(new_observation)

                if min == dissimilarity or len(candidate_entry) == 0:
                    candidate_entry.append(entry)
                    min = dissimilarity
                elif min > dissimilarity:
                    candidate_entry = [entry]
                    min = dissimilarity

        console(2, module, "Min dissmilarity : " + str(min))

        if min > self.model.grab_threshold:
            return None, min
        else:
            selected_entry = random.choice(candidate_entry)
            selected_entry.status_for_grabObs = True
            child_belief_node = selected_entry.child_node
            # self.belief_mapping.copy_belief_node(selected_entry.observation, new_observation)

            console(2, module, "Had to grab nearest belief node...variance added")

            return child_belief_node, min


    def create_child(self, action_node, obs):
        child_belief_node, added = self.belief_tree_index.create_child(action_node, obs)

        return child_belief_node