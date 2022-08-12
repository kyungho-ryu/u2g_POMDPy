import random
import numpy as np


# UCB1 action selection algorithm
def ucb_action(mcts, current_node, greedy):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    print("mapping", mapping)
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
        if not greedy:
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if current_q >= best_q_value:
            if current_q > best_q_value:
                best_actions = []
            best_q_value = current_q
            # best actions is a list of Discrete Actions
            best_actions.append(action_entry.get_action())

    assert best_actions.__len__() is not 0

    return random.choice(best_actions)

def action_progWiden(mcts, current_node, temp_action, k, alpha):
    mapping = current_node.action_map

    # print("C : {}, kxN^alpha : {}".format(mapping.number_of_children, k * (mapping.total_visit_count**alpha)))
    if mapping.current_action_count <= k * (mapping.total_visit_count**alpha) :

        action, C, N =  mapping.create_current_action_node(temp_action)

        return action, C, N
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
        return action, mapping.current_action_count, N

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