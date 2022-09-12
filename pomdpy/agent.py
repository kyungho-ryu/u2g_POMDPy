from __future__ import print_function, division
import time, psutil
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider, summary, memory
from DRL.structure import DRLType
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
        # solver = self.solver_factory(self)
        eps = self.model.epsilon_start
        simulation_steps = 0
        steps = 0
        # init_belief_mapping_index = solver.belief_tree_index
        solver = None
        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            self.results = Results()
            eps, steps, simulation_steps, solver = self.run_pomcp(solver, i + 1, eps, simulation_steps, steps)

            # solver.model.reset_for_epoch()
            # solver.belief_tree_index = init_belief_mapping_index

            # start = time.time()
            # memory.check_momory(self.logger)
            # memory.clean_memory(self.logger)
            # memory.check_momory(self.logger)
            # self.logger.info("Summary delay : {}".format(time.time() - start))

    def run_pomcp(self, solver, epoch, eps, simulation_steps, steps):
        epoch_start = time.time()
        # -------------------------implement root belief tree-----------------------------------------------
        new_solver = self.solver_factory(self)
        if solver != None :
            new_solver.A2CSample = solver.A2CSample
            new_solver.A2CModel = solver.A2CModel

        solver = new_solver

        # Monte-Carlo start state
        # choice random state from particles (2000)

        state = solver.belief_tree_index.sample_particle()
        self.logger.debug("[{}]state:\n{}".format(epoch, state.to_string()))
        self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))

        # print("Agent/state.position", state.position)
        # print("Agent/state.rock_states", state.rock_states)

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
            action, best_ucb_value, best_q_value = solver.select_eps_greedy_action(epoch, simulation_steps, eps, start_time)
            self.results.ucb_value.append(best_ucb_value)
            self.results.q_value.append(best_q_value)
            new_action.append(action.UAV_deployment)
            print('\n')
            self.logger.debug("[{}/{}]'acition : {}".format(epoch, i, action.UAV_deployment))

            # update epsilon
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay

            self.logger.info("GMU' prediction Length : {}".format(state.get_gmus_prediction_length()))
            # state = not real state
            self.results.prediction_error.append(solver.model.get_dissimilarity_of_gmu_prediction(state.gmus))
            step_result, is_legal, R1, R2 = solver.model.generate_real_step(state, action)

            if self.model.DRLType == DRLType.OS_PPOModel or self.model.DRLType == DRLType.OS_A2CModel :
                action_node = solver.belief_tree_index.action_map.get_action_node(action)
                solver.A2CSample.add_batch_sample(action_node.state_for_learning, action.UAV_deployment,
                                                  action_node.old_logprob_v, step_result.reward, step_result.is_terminal)

                if step_result.is_terminal :
                    solver.A2CSample.add_sample()

                    for _ in range(self.model.learning_iteration_for_PPO) :
                        advantage, loss, step, std_list, logStdStep = solver.A2CModel.update(solver.A2CSample)
                        summary.summary_NNResult(solver.model.writer, advantage, loss, step,
                                                 solver.A2CSample.get_batch_len(),
                                                 solver.A2CSample.NumSample, std_list, logStdStep)
                        self.A2CModel.logStdStep = step

                    solver.A2CSample.reset_batch()
                    solver.reset_A2CSample()
                else :
                    if self.model.inner_batch_for_NN <= solver.A2CSample.get_batch_len() :
                        solver.A2CSample.add_sample()
                        solver.A2CSample.reset_batch()

                    if self.model.batch_for_NN <= solver.A2CSample.NumSample :
                        for _ in range(self.model.learning_iteration_for_PPO) :
                            advantage, loss, step, std_list, logStdStep = solver.A2CModel.update(solver.A2CSample)
                            summary.summary_NNResult(solver.model.writer, advantage, loss, step,
                                                     self.model.inner_batch_for_NN,
                                                     solver.A2CSample.NumSample, std_list, logStdStep)
                            solver.A2CModel.logStdStep = step

                        solver.reset_A2CSample()

            if self.results.initial_reward == 0 :
                self.results.initial_reward = R1 + R2

            if self.results.count == 1:
                self.results.second_reward = R1 + R2

            if self.results.count == 2:
                self.results.thrid_reward = R1 + R2

            _totalA2GEnergy, _totalA2AEnergy, _totalPropEnergy, _totalEnergyConsumtion, _avgDnRage, \
            _scaledEnergyConsumtion, _scaledDnRate, _NumActiveUav, _NumObservedGMU = self.model.get_simulationResult(state, action)
            self.results.totalA2GEnergy.append(_totalA2GEnergy)
            self.results.totalA2AEnergy.append(_totalA2AEnergy)
            self.results.totalPropEnergy.append(_totalPropEnergy)
            self.results.totalEnergyConsumtion.append(_totalEnergyConsumtion)
            self.results.avgDnRage.append(_avgDnRage)
            self.results.scaledEnergyConsumtion.append(_scaledEnergyConsumtion)
            self.results.scaledDnRate.append(_scaledDnRate)
            self.results.NumActiveUav.append(_NumActiveUav)
            self.results.NumObservedGMU.append(_NumObservedGMU)

            self.results.reward.append(R1 + R2)
            self.results.discounted_reward.append(self.results.discount * R1 + R2)
            # discounted_reward = discount * step_result.reward
            self.results.discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result, [R1, R2])

            start = time.time()
            if not step_result.is_terminal:
                result, new_dissimilarity = solver.update(state, step_result, True)
                if result ==1 :
                    self.results.NUM_create_child_belief_node +=1
                if result == 2 :
                    self.results.NUM_grab_nearest_child_belief_node +=1
            else :
                # new_dissimilarity = solver.get_dissimilarity(step_result)
                new_dissimilarity = -1

            self.logger.info("tree update delay : {}".format(time.time() - start))
            self.results.dissimilarity.append(new_dissimilarity)

            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward, step_result.action, step_result.observation, step_result.next_state)


            self.results.count +=1
            state = solver.belief_tree_index.sample_particle()


            if simulation_steps % 1 == 0 :
                self.results.summary_result(self.model.writer, self.logger, R1 + R2,
                                            epoch_start, simulation_steps)
                self.results = None
                self.results = Results()
                # self.logger.info("Action equality : {}".format(actionEquality))

            simulation_steps +=1

            if step_result.is_terminal or not is_legal :
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                self.results.summary_result(self.model.writer, self.logger, R1 + R2,
                                            epoch_start, simulation_steps)
                break

        steps += 1


        return eps, steps, simulation_steps, solver

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
        self.NUM_create_child_belief_node = 0
        self.NUM_grab_nearest_child_belief_node = 0
        self.dissimilarity = []
        self.reward = []
        self.discounted_reward = []
        self.discount=1
        self.initial_reward = 0
        self.second_reward = 0
        self.thrid_reward = 0
        self.totalA2GEnergy = []
        self.totalA2AEnergy = []
        self.totalPropEnergy = []
        self.totalEnergyConsumtion = []
        self.avgDnRage = []
        self.scaledEnergyConsumtion = []
        self.scaledDnRate = []
        self.NumActiveUav = []
        self.NumObservedGMU = []
        self.count = 0
        self.prediction_error = []
        self.ucb_value = []
        self.q_value = []

    def summary_result(self, writer, logger, reward, epoch_start, simulation_steps):
        usedMemory = memory.check_momory(logger)


        summary.summary_result(
            writer, simulation_steps, self.initial_reward,
            self.second_reward, self.thrid_reward,
            self.reward, self.discounted_reward, reward,
            self.ucb_value, self.q_value,
            self.NUM_grab_nearest_child_belief_node, self.NUM_create_child_belief_node,
            self.dissimilarity, self.totalA2GEnergy, self.totalA2AEnergy,
            self.totalPropEnergy, self.totalEnergyConsumtion, self.avgDnRage,
            self.scaledEnergyConsumtion, self.scaledDnRate, self.NumActiveUav,
            self.NumObservedGMU, self.prediction_error, usedMemory,
            self.count, (time.time() - epoch_start)
        )

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
