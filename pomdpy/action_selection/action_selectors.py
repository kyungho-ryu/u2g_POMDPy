import random
import numpy as np


# UCB1 action selection algorithm
def ucb_action(mcts, current_node):
    best_actions = []
    best_q_value = -np.inf
    best_ucb_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    actions = list(mapping.entries.values()) # move (4) + sample (1) + check (8)
    random.shuffle(actions)
    for action_entry in actions:
        # Skip illegal actions
        if not action_entry.is_legal:
            continue

        current_q = action_entry.mean_q_value

        # If the UCB coefficient is 0, this is greedy Q selection
        current_ucb = current_q + mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if current_ucb >= best_ucb_value:
            if current_ucb > best_ucb_value:
                best_actions = []
            best_ucb_value = current_ucb
            best_q_value = current_q
            # best actions is a list of Discrete Actions
            best_actions.append(action_entry.get_action())

    assert best_actions.__len__() is not 0

    return random.choice(best_actions), best_ucb_value, best_q_value

def action_progWiden(mcts, current_node, temp_action, k, alpha):
    mapping = current_node.action_map

    # print("C : {}, kxN^alpha : {}".format(mapping.number_of_children, k * (mapping.total_visit_count**alpha)))
    # if log :
    #     print(mapping.get_number_of_action())
    #     print(mapping.total_visit_count, k, alpha)
    #     print(k * (mapping.total_visit_count**alpha))
    if mapping.get_number_of_action() <= k * (mapping.total_visit_count**alpha) :

        action, C, N =  mapping.create_current_action_node(temp_action)
        return action, C, N, "Create New action"
    else :
        best_actions = []
        N = mapping.total_visit_count
        log_n = np.log(N + 1)
        best_q_value = -np.inf

        actions = list(mapping.entries.values())
        random.shuffle(actions)
        for action_entry in actions:
            # Skip illegal actions
            if not action_entry.is_legal:
                continue

            current_q = action_entry.mean_q_value
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)
            if current_q >= best_q_value:
                if current_q > best_q_value:
                    best_actions = []
                best_q_value = current_q
                # best actions is a list of Discrete Actions
                best_actions.append(action_entry.get_action())
        assert best_actions.__len__() is not 0

        action = random.choice(best_actions)
        return action, mapping.get_number_of_action(), N, "Select action according to Value function"

def e_greedy(current_node, epsilon):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    actions = list(mapping.entries.values())
    random.shuffle(actions)

    if np.random.uniform(0, 1) < epsilon:
        for action_entry in actions:
            if not action_entry.is_legal:
                continue
            else:
                return action_entry.get_action()
        # No legal actions
        raise RuntimeError('No legal actions to take')
    else:
        # Greedy choice
        for action_entry in actions:
            # Skip illegal actions
            if not action_entry.is_legal:
                continue

            current_q = action_entry.mean_q_value

            if current_q >= best_q_value:
                if current_q > best_q_value:
                    best_actions = []
                best_q_value = current_q
                # best actions is a list of Discrete Actions
                best_actions.append(action_entry.get_action())

        assert best_actions.__len__() is not 0

        return random.choice(best_actions)