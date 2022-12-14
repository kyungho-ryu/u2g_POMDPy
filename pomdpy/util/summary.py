import numpy as np

def summary_simulationResult(writer, beliefTree, epoch) :
    tree_depth = 0

    obsList = [beliefTree]
    while True :
        group = "depth'"+str(tree_depth) + '/'

        N_h = 0
        C_h = 0
        N_ha = 0
        C_ha = 0
        particle = 0
        newObsList = []

        LenAction = 0

        for i in range(len(obsList)) :
            actionMapping = obsList[i].action_map
            N_h += actionMapping.total_visit_count
            C_h += actionMapping.get_number_of_action()
            particle += obsList[i].get_num_total_particle()

            for actionEntry in actionMapping.get_all_entries() :
                if actionEntry.child_node == None :
                    continue
                N_ha += actionEntry.child_node.get_visit_count()
                C_ha += actionEntry.child_node.get_count_child()

                newObsList += actionEntry.child_node.get_child_all_nodes()
                LenAction +=1

        if len(newObsList) == 0 or tree_depth == 2:
            break

        writer.add_scalar(group+'N_h', N_h/len(obsList), epoch)
        writer.add_scalar(group+'C_h', C_h/len(obsList), epoch)
        writer.add_scalar(group+'particle', particle/len(obsList),epoch)

        writer.add_scalar(group+'N_ha', N_ha/LenAction,epoch)
        writer.add_scalar(group+'c_ha', C_ha/LenAction,epoch)


        obsList = newObsList
        tree_depth +=1
        break

    writer.add_scalar("depth", tree_depth, epoch)

def summary_NNResult(writer, advantage, loss, step, MaxDepth, NumSample, std_list, logStdStep) :
    group = "NN/"
    for i in range(len(std_list)) :
        writer.add_scalar(group + 'logstd', std_list[i], logStdStep+i)
    writer.add_scalar(group + 'advantage', advantage, step)
    writer.add_scalar(group + 'loss', loss, step)

    writer.add_scalar(group + 'MaxDepth', MaxDepth, step)
    writer.add_scalar(group + 'NumSample', NumSample, step)


def summary_result(writer, epoch, reward, discounted_reward, last_reward,
                    ucb_value, q_value, visit_count, NUM_grab_nearest_child_belief_node, NUM_create_child_belief_node,
                   dissimilarity, totalA2GEnergy, totalA2AEnergy, totalPropEnergy,
                   totalEnergyConsumtion, avgDnRage, scaledEnergyConsumtion, scaledDnRate,
                   NumActiveUav, NumObservedGMU, prediction_error, usedMemory,
                   propRealState, propDifference,count, time) :

    group = "Reward/"
    writer.add_scalar(group+'R', np.mean(reward), epoch)
    writer.add_scalar(group+'lastR', last_reward, epoch)
    writer.add_scalar(group+'discounted_R', np.mean(discounted_reward), epoch)
    writer.add_scalar(group + 'UCB', np.mean(ucb_value), epoch)
    writer.add_scalar(group + 'Q', np.mean(q_value), epoch)


    group = "Energy/"
    writer.add_scalar(group + 'A2GEnergy', np.mean(totalA2GEnergy), epoch)
    writer.add_scalar(group + 'A2AEnergy', np.mean(totalA2AEnergy), epoch)
    writer.add_scalar(group + 'PropEnergy', np.mean(totalPropEnergy), epoch)

    group = "TotalReward/"
    writer.add_scalar(group + 'energy', np.mean(totalEnergyConsumtion), epoch)
    writer.add_scalar(group + 'dataRate', np.mean(avgDnRage), epoch)
    writer.add_scalar(group + 'scaledEnergy', np.mean(scaledEnergyConsumtion), epoch)
    writer.add_scalar(group + 'scaledDataRate', np.mean(scaledDnRate), epoch)


    group = "etc/"
    writer.add_scalar(group + 'activeUav', np.mean(NumActiveUav), epoch)
    writer.add_scalar(group + 'observedGMU', np.mean(NumObservedGMU), epoch)
    writer.add_scalar(group + 'action visit count (N)', np.mean(visit_count), epoch)
    writer.add_scalar(group + 'attachProbabality', (count - NUM_create_child_belief_node)/count, epoch)
    writer.add_scalar(group + 'GrabProbabality', NUM_grab_nearest_child_belief_node, epoch)
    writer.add_scalar(group + 'dissimilarity', np.mean(dissimilarity), epoch)
    writer.add_scalar(group + 'prediction_error', np.mean(prediction_error), epoch)
    writer.add_scalar(group + 'propRealState', np.mean(propRealState), epoch)
    writer.add_scalar(group + 'propDifference', np.mean(propDifference), epoch)
    writer.add_scalar(group + 'usedMemory', usedMemory, epoch)
    # writer.add_scalar(group + 'Action Equality', NumactionEquality, epoch)

    writer.add_scalar("Time", time, epoch)


def summary_result2(writer, epoch, init_reward, reward, discounted_reward,
                   ucb_value, q_value, NUM_grab_nearest_child_belief_node, NUM_create_child_belief_node,
                   dissimilarity, totalA2GEnergy, totalA2AEnergy, totalPropEnergy,
                   totalEnergyConsumtion, avgDnRage, scaledEnergyConsumtion, scaledDnRate,
                   NumActiveUav, NumObservedGMU, prediction_error, count, time) :

    group = "Reward/"
    writer.add_scalar(group+'R', reward, epoch)
    writer.add_scalar(group+'initR', init_reward, epoch)
    writer.add_scalar(group+'discounted_R', discounted_reward, epoch)
    writer.add_scalar(group + 'UCB', ucb_value, epoch)
    writer.add_scalar(group + 'Q', q_value, epoch)


    group = "Energy/"
    writer.add_scalar(group + 'A2GEnergy', totalA2GEnergy, epoch)
    writer.add_scalar(group + 'A2AEnergy', totalA2AEnergy, epoch)
    writer.add_scalar(group + 'PropEnergy', totalPropEnergy, epoch)

    group = "TotalReward/"
    writer.add_scalar(group + 'energy', totalEnergyConsumtion, epoch)
    writer.add_scalar(group + 'dataRate', avgDnRage, epoch)
    writer.add_scalar(group + 'scaledEnergy', scaledEnergyConsumtion, epoch)
    writer.add_scalar(group + 'scaledDataRate', scaledDnRate, epoch)


    group = "etc/"
    writer.add_scalar(group + 'activeUav', NumActiveUav, epoch)
    writer.add_scalar(group + 'observedGMU', NumObservedGMU, epoch)
    writer.add_scalar(group + 'attachProbabality', (count - NUM_create_child_belief_node)/count, epoch)
    writer.add_scalar(group + 'GrabProbabality', NUM_grab_nearest_child_belief_node, epoch)
    if dissimilarity != -1 :
        writer.add_scalar(group + 'dissimilarity', dissimilarity, epoch)
    writer.add_scalar(group + 'prediction_error', prediction_error, epoch)

    writer.add_scalar("Time", time, epoch)

