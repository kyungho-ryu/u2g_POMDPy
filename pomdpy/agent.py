from __future__ import print_function, division
import time
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider, summary
from experiments.scripts.pickle_wrapper import save_pkl

module = "agent"


class Agent:
    """
    Train and store experimental results for different types of runs

    """

    def __init__(self, model, solver):
        """
        Initialize the POMDPY agent
        :param model:
        :param solver:
        :return:
        """
        self.logger = logging.getLogger('POMDPy.Solver')
        self.logger.setLevel(logging.INFO)
        self.model = model
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self):

        if self.model.solver == 'ValueIteration':
            solver = self.solver_factory(self)

            self.run_value_iteration(solver, 1)

            if self.model.save:
                save_pkl(solver.gamma,
                         os.path.join(self.model.weight_dir,
                                      'VI_planning_horizon_{}.pkl'.format(self.model.planning_horizon)))

        elif not self.model.use_tf:
            self.multi_epoch()
        else:
            self.multi_epoch_tf()

        print('\n')
        console(2, module, 'epochs: ' + str(self.model.n_epochs))
        console(2, module, 'ave undiscounted return/step: ' + str(self.experiment_results.undiscounted_return.mean) +
                ' +- ' + str(self.experiment_results.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return/step: ' + str(self.experiment_results.discounted_return.mean) +
                ' +- ' + str(self.experiment_results.discounted_return.std_err()))
        console(2, module, 'ave time/epoch: ' + str(self.experiment_results.time.mean))

        self.logger.info('env: ' + self.model.env + '\t' +
                         'epochs: ' + str(self.model.n_epochs) + '\t' +
                         'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) + ' +- ' +
                         str(self.experiment_results.undiscounted_return.std_err()) + '\t' +
                         'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                         ' +- ' + str(self.experiment_results.discounted_return.std_err()) +
                         '\t' + 'ave time/epoch: ' + str(self.experiment_results.time.mean))

    def multi_epoch_tf(self):
        import tensorflow as tf
        tf.set_random_seed(int(self.model.seed) + 1)

        with tf.Session() as sess:
            solver = self.solver_factory(self, sess)

            for epoch in range(self.model.n_epochs + 1):

                self.model.reset_for_epoch()

                if epoch % self.model.test == 0:
                    epoch_start = time.time()

                    print('evaluating agent at epoch {}...'.format(epoch))

                    # evaluate agent
                    reward = 0.
                    discounted_reward = 0.
                    discount = 1.0
                    belief = self.model.get_initial_belief_state()
                    step = 0
                    while step < self.model.max_steps:
                        action, v_b = solver.greedy_predict(belief)
                        step_result = self.model.generate_step(action)

                        if not step_result.is_terminal:
                            belief = self.model.belief_update(belief, action, step_result.observation)

                        reward += step_result.reward
                        discounted_reward += discount * step_result.reward
                        discount *= self.model.discount

                        # show the step result
                        self.display_step_result(epoch, step_result)
                        step += 1
                        if step_result.is_terminal:
                            break

                    self.experiment_results.time.add(time.time() - epoch_start)
                    self.experiment_results.undiscounted_return.count += 1
                    self.experiment_results.undiscounted_return.add(reward)
                    self.experiment_results.discounted_return.count += 1
                    self.experiment_results.discounted_return.add(discounted_reward)

                    summary = sess.run([solver.experiment_summary], feed_dict={
                        solver.avg_undiscounted_return: self.experiment_results.undiscounted_return.mean,
                        solver.avg_undiscounted_return_std_dev: self.experiment_results.undiscounted_return.std_dev(),
                        solver.avg_discounted_return: self.experiment_results.discounted_return.mean,
                        solver.avg_discounted_return_std_dev: self.experiment_results.discounted_return.std_dev()
                    })
                    for summary_str in summary:
                        solver.summary_ops['writer'].add_summary(summary_str, epoch)

                    # TODO: save model at checkpoints
                else:

                    # train for 1 epoch
                    solver.train(epoch)

            if self.model.save:
                solver.save_alpha_vectors()
                print('saved alpha vectors!')

    def multi_epoch(self):
        # Create a new solver, purune belief tree
        solver = self.solver_factory(self)
        eps = self.model.epsilon_start
        steps = 1
        NUM_create_child_belief_node = 0
        NUM_grab_nearest_child_belief_node = 0
        prior_state = solver.model.get_an_init_prior_state()
        prior_state_key = prior_state.get_key()
        observation = solver.model.sample_an_init_observation()
        init_belief_mapping_index = solver.belief_mapping_index
        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            self.results = Results()

            if self.model.solver == 'POMCP-DPW' or self.model.solver=="POMCP-POW":
                eps, steps, NUM_create_child_belief_node, NUM_grab_nearest_child_belief_node = \
                    self.run_pomcp(solver, i + 1, eps, steps, NUM_create_child_belief_node, NUM_grab_nearest_child_belief_node, prior_state_key)

                solver.model.reset_for_epoch()
                solver.belief_mapping_index = init_belief_mapping_index
                for i in range(self.model.n_start_states):
                    particle = self.model.sample_an_init_state()  # create random rock state
                    solver.belief_mapping.add_particle(observation, particle, prior_state_key)


    def run_pomcp(self, solver, epoch, eps, steps, NUM_create_child_belief_node, NUM_grab_nearest_child_belief_node, prior_state_key):
        epoch_start = time.time()
        # -------------------------implement root belief tree-----------------------------------------------

        # Monte-Carlo start state
        # choice random state from particles (2000)

        state = solver.belief_mapping_index.sample_particle(prior_state_key)
        self.logger.debug("[{}]state:\n{}".format(epoch, state.to_string()))
        self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))

        # print("Agent/state.position", state.position)
        # print("Agent/state.rock_states", state.rock_states)
        reward = 0
        discounted_reward = 0
        discount = 1.0

        print_divider('large')
        print('\tEpoch #' + str(epoch))

        # episode start
        for i in range(self.model.max_steps):
            # state is changed
            start_time = time.time()

            # action will be of type Discrete Action
            print_divider('large')
            print('\tStep #' + str(i) + ' simulation is working\n')

            action = solver.select_eps_greedy_action(epoch, steps, eps, start_time, prior_state_key)
            print('\n')
            self.logger.debug("[{}/{}]'acition : {}".format(epoch, i, action.UAV_deployment))

            # update epsilon
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay

            self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))
            # state = not real state
            step_result, is_legal = solver.model.generate_step(state, action)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward

            discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result)

            if not step_result.is_terminal:
                result = solver.update(state, step_result, False)
                if result == 1 :
                    NUM_create_child_belief_node +=1
                elif result == 2 :
                    NUM_grab_nearest_child_belief_node +=1

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward, step_result.action, step_result.observation, step_result.next_state)

            # prob_attach_existing_belief_node = 1 - (NUM_create_child_belief_node/steps)
            summary.summary_result(
                self.model.writer, steps, step_result.reward, discounted_reward, steps-NUM_create_child_belief_node,
                self.model.get_simulationResult(state, action), time.time()- epoch_start
            )

            prior_state_key = state.get_key()
            state = solver.belief_mapping_index.sample_particle(prior_state_key)

            steps +=1

            if step_result.is_terminal or not is_legal :
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

            if steps % 10 ==0:
                break

        self.results.time.add(time.time() - epoch_start)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        # print_divider('large')
        # solver.history.show()
        self.results.show(epoch)
        #console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        #print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

        return eps, steps, NUM_create_child_belief_node, NUM_grab_nearest_child_belief_node

    def run_value_iteration(self, solver, epoch):
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        solver.value_iteration(self.model.get_transition_matrix(),
                               self.model.get_observation_matrix(),
                               self.model.get_reward_matrix(),
                               self.model.planning_horizon)

        b = self.model.get_initial_belief_state()

        for i in range(self.model.max_steps):

            # TODO: record average V(b) per epoch
            action, v_b = solver.select_action(b, solver.gamma)

            step_result = self.model.generate_step(action)

            if not step_result.is_terminal:
                b = self.model.belief_update(b, action, step_result.observation)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result)

            if step_result.is_terminal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

            # TODO: add belief state History sequence

        self.results.time.add(time.time() - run_start_time)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

    @staticmethod
    def display_step_result(step_num, step_result):
        """
        Pretty prints step result information
        :param step_num:
        :param step_result:
        :return:
        """
        console(3, module, 'Step Number = ' + str(step_num))
        console(3, module, 'Step Result.Action = ' + step_result.action.to_string())
        console(3, module, 'Step Result.Observation = ' + step_result.observation.to_string())
        # console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
        console(3, module, 'Step Result.Reward = ' + str(step_result.reward))

class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        print_divider('medium')
        console(2, module, 'Time')
        print_divider('medium')
        self.time.show()
        print_divider('medium')
