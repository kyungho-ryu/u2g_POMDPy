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

    actions = list(beliefTree.action_map.entries.values())
    best_q_value = -np.inf
    for action_entry in actions:
        # Skip illegal actions
        if not action_entry.is_legal:
            continue

        current_q = action_entry.mean_q_value

        if current_q >= best_q_value:
            best_q_value = current_q


    writer.add_scalar("Q", best_q_value, epoch)