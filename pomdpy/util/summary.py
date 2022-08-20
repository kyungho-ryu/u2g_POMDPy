import numpy as np

def summary_simulationResult(writer, beliefTree, best_ucb_value, best_q_value, epoch) :
    group = "Reward/"
    writer.add_scalar(group + 'UCB', best_ucb_value, epoch)
    writer.add_scalar(group + 'Q', best_q_value, epoch)

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
            particle += len(obsList[i].state_particles)

            for actionEntry in actionMapping.get_all_entries() :
                if actionEntry.child_node == None :
                    continue
                N_ha += actionEntry.child_node.get_visit_count()
                C_ha += actionEntry.child_node.get_count_child()

                newObsList += actionEntry.child_node.get_child_all_nodes()
                LenAction +=1

        if len(newObsList) == 0 :
            break

        writer.add_scalar(group+'N_h', N_h/len(obsList), epoch)
        writer.add_scalar(group+'C_h', C_h/len(obsList), epoch)
        writer.add_scalar(group+'particle', particle/len(obsList),epoch)

        writer.add_scalar(group+'N_ha', N_ha/LenAction,epoch)
        writer.add_scalar(group+'c_ha', C_ha/LenAction,epoch)


        obsList = newObsList
        tree_depth +=1

    writer.add_scalar("depth", tree_depth, epoch)


def summary_result(writer, epoch, reward, discounted_reward, prob_attach_existing_belief_node, simulationResult, time) :
    group = "Reward/"
    writer.add_scalar(group+'R', reward, epoch)
    writer.add_scalar(group+'discounted_R', discounted_reward, epoch)

    group = "Energy/"
    writer.add_scalar(group + 'A2GEnergy', simulationResult[0], epoch)
    writer.add_scalar(group + 'A2AEnergy', simulationResult[1], epoch)
    writer.add_scalar(group + 'PropEnergy', simulationResult[2], epoch)

    group = "TotalReward/"
    writer.add_scalar(group + 'energy', simulationResult[3], epoch)
    writer.add_scalar(group + 'dataRate', simulationResult[4], epoch)
    writer.add_scalar(group + 'scaledEnergy', simulationResult[5], epoch)
    writer.add_scalar(group + 'scaledDataRate', simulationResult[6], epoch)


    group = "etc/"
    writer.add_scalar(group + 'activeUav', simulationResult[7], epoch)
    writer.add_scalar(group + 'observedGMU', simulationResult[8], epoch)
    writer.add_scalar(group + 'attachProbabality', prob_attach_existing_belief_node, epoch)

    writer.add_scalar("Time", time, epoch)


