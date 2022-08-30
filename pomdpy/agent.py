from __future__ import print_function, division
import time, psutil
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider, summary, memory
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
        simulation_steps = 0
        steps = 0
        prior_state = solver.model.get_an_init_prior_state()
        prior_state_key = prior_state.get_key()
        init_belief_mapping_index = solver.belief_tree_index
        previous_action = []
        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            self.results = Results()
            print("init_belief_mapping_index", init_belief_mapping_index)
            eps, steps, simulation_steps,previous_action = self.run_pomcp(solver, i + 1, eps, simulation_steps, steps, previous_action, prior_state_key)

            solver.model.reset_for_epoch()
            solver.belief_tree_index = init_belief_mapping_index

            # start = time.time()
            # memory.check_momory(self.logger)
            # memory.clean_memory(self.logger)
            # memory.check_momory(self.logger)
            # self.logger.info("Summary delay : {}".format(time.time() - start))

    def run_pomcp(self, solver, epoch, eps, simulation_steps, steps, previous_action, prior_state_key):
        epoch_start = time.time()
        # -------------------------implement root belief tree-----------------------------------------------

        # Monte-Carlo start state
        # choice random state from particles (2000)

        state = solver.belief_tree_index.sample_particle(prior_state_key)
        self.logger.debug("[{}]state:\n{}".format(epoch, state.to_string()))
        self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))

        # print("Agent/state.position", state.position)
        # print("Agent/state.rock_states", state.rock_states)
        NUM_create_child_belief_node = 0
        NUM_grab_nearest_child_belief_node = 0
        dissimilarity = []
        reward = []
        discounted_reward = []
        discount=1
        initial_reward = 0
        second_reward = 0
        thrid_reward = 0
        totalA2GEnergy = []
        totalA2AEnergy = []
        totalPropEnergy = []
        totalEnergyConsumtion = []
        avgDnRage = []
        scaledEnergyConsumtion = []
        scaledDnRate = []
        NumActiveUav = []
        NumObservedGMU = []
        count = 0
        ucb_value = []
        q_value = []
        prediction_error = []
        print_divider('large')
        print('\tEpoch #' + str(epoch))

        new_action = []
        # episode start
        for i in range(self.model.max_steps):
            # state is changed
            start_time = time.time()

            # memory.check_momory(self.logger)
            # action will be of type Discrete Action
            print_divider('large')
            print('\tStep #' + str(i) + ' simulation is working\n')
            action, best_ucb_value, best_q_value = solver.select_eps_greedy_action(epoch, simulation_steps, eps, start_time, prior_state_key)
            ucb_value.append(best_ucb_value)
            q_value.append(best_q_value)
            new_action.append(action)
            print('\n')
            self.logger.debug("[{}/{}]'acition : {}".format(epoch, i, action.UAV_deployment))

            # update epsilon
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay

            self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))
            # state = not real state
            prediction_error.append(solver.model.get_dissimilarity_of_gmu_prediction(state.gmus))
            step_result, is_legal, eachReward = solver.model.generate_real_step(state, action)

            reward.append(step_result.reward)
            discounted_reward.append(discount * step_result.reward)
            # discounted_reward = discount * step_result.reward
            discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result, eachReward)

            start = time.time()
            if not step_result.is_terminal:
                result, new_dissimilarity = solver.update(state, step_result, True)
                if result ==1 :
                    NUM_create_child_belief_node +=1
                if result == 2 :
                    NUM_grab_nearest_child_belief_node +=1
            else :
                # new_dissimilarity = solver.get_dissimilarity(step_result)
                new_dissimilarity = -1

            self.logger.info("tree update delay : {}".format(time.time() - start))
            dissimilarity.append(new_dissimilarity)
            # print("END======================================================================")
            # memory.check_momory(self.logger)
            # exit()
            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward, step_result.action, step_result.observation, step_result.next_state)

            if initial_reward == 0 :
                initial_reward = step_result.reward

            if count == 1:
                second_reward = step_result.reward

            if count == 2:
                thrid_reward = step_result.reward

            _totalA2GEnergy, _totalA2AEnergy, _totalPropEnergy, _totalEnergyConsumtion, _avgDnRage, \
            _scaledEnergyConsumtion, _scaledDnRate, _NumActiveUav, _NumObservedGMU = self.model.get_simulationResult(state, action)
            totalA2GEnergy.append(_totalA2GEnergy)
            totalA2AEnergy.append(_totalA2AEnergy)
            totalPropEnergy.append(_totalPropEnergy)
            totalEnergyConsumtion.append(_totalEnergyConsumtion)
            avgDnRage.append(_avgDnRage)
            scaledEnergyConsumtion.append(_scaledEnergyConsumtion)
            scaledDnRate.append(_scaledDnRate)
            NumActiveUav.append(_NumActiveUav)
            NumObservedGMU.append(_NumObservedGMU)

            count +=1
            prior_state_key = state.get_key()
            state = solver.belief_tree_index.sample_particle(prior_state_key)

            simulation_steps +=1

            # summary.summary_result2(
            #     self.model.writer, simulation_steps, initial_reward, step_result.reward, discounted_reward,
            #     best_ucb_value, best_q_value, NUM_grab_nearest_child_belief_node, NUM_create_child_belief_node,
            #     new_dissimilarity, _totalA2GEnergy, _totalA2AEnergy, _totalPropEnergy, _totalEnergyConsumtion,
            #     _avgDnRage, _scaledEnergyConsumtion, _scaledDnRate, _NumActiveUav, _NumObservedGMU,
            #     solver.model.get_dissimilarity_of_gmu_prediction(state.gmus), count, (time.time() - epoch_start)
            # )

            if step_result.is_terminal or not is_legal :
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break
        steps += 1
        usedMemory = memory.check_momory(self.logger)

        actionEquality = []
        NumactionEquality = 0
        for i in range(len(previous_action)) :
            if previous_action[i] == new_action[i] :
                actionEquality.append(True)
                NumactionEquality +=1
            else :
                actionEquality.append(False)

        summary.summary_result(
            self.model.writer, simulation_steps, initial_reward, second_reward, thrid_reward,
            reward, discounted_reward, step_result.reward, ucb_value, q_value,
            NUM_grab_nearest_child_belief_node, NUM_create_child_belief_node,
            dissimilarity,totalA2GEnergy, totalA2AEnergy, totalPropEnergy, totalEnergyConsumtion,
            avgDnRage, scaledEnergyConsumtion, scaledDnRate, NumActiveUav, NumObservedGMU, prediction_error,
            usedMemory, NumactionEquality, count, (time.time() - epoch_start)
        )



        self.logger.info("Action equality : {}".format(actionEquality))

        return eps, steps, simulation_steps, new_action

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
    def display_step_result(step_num, step_result, eachReward):
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

        console(3, module, 'Step Result.EachReward = ' + str(eachReward))

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
