import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from tqdm import tqdm

def play_episode(env, policy, first_visit=True):
    data = []
    #visited_state_actions = set()

    done = False
    state, _ = env.reset()
    while not done:
        action = np.random.choice(range(env.action_space.n), p=policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)

        #if first_visit and (state, action) not in visited_state_actions:
        #visited_state_actions.add((state, action))
        data.append((state, action, reward))

        state = next_state
        done = terminated or truncated

    return data

def mc_soft_first_visit(env, no_episodes=int(8e5), gamma=1, epsilon=0.2):
    no_states = env.observation_space.n
    no_actions = env.action_space.n


    q = np.zeros((no_states, no_actions))
    c = np.zeros((no_states, no_actions))

    policy = np.ones((no_states, no_actions)) / no_actions

    progress = tqdm(range(no_episodes), postfix={})
    for _ in progress:
        progress.set_postfix({"epsilon": epsilon})
        state_action_rewards = play_episode(env, policy, True)

        G = 0
        for state, action, reward in reversed(state_action_rewards):
            G = gamma * G + reward
            c[state, action] += 1
            q[state, action] += (G - q[state, action]) / c[state, action]

            a_star = q[state].argmax()
            policy[state] = epsilon / no_actions
            policy[state, a_star] += 1 - epsilon

    return policy, q

def mc_importance_sampling_first_visit(env, no_episodes=int(1e5), gamma=1, epsilon=0.1):
    no_states = env.observation_space.n
    no_actions = env.action_space.n
    q = np.zeros((no_states, no_actions))
    c = np.zeros((no_states, no_actions))

    behaviour_policy = np.ones((no_states, no_actions)) / no_actions
    policy = np.ones((no_states, no_actions)) / no_actions

    progress = tqdm(range(no_episodes), postfix={})
    for _ in progress:
        state_action_rewards = play_episode(env, behaviour_policy, True)
        G = 0
        W = 1.0

        # Process episode in reverse for first-visit MC
        visited = set()
        for state, action, reward in reversed(state_action_rewards):
            G = gamma * G + reward

            # Only first visit updates
            if (state, action) not in visited:
                visited.add((state, action))

                c[state, action] += W
                q[state, action] += (W / c[state, action]) * (G - q[state, action])

                # Update target policy to be epsilon-soft greedy wrt q
                best_action = np.argmax(q[state])
                policy[state] = epsilon / no_actions
                policy[state, best_action] += 1 - epsilon

                # If behavior_policy[state, action] is zero, break to avoid div by zero
                if behaviour_policy[state, action] == 0:
                    break

                W *= policy[state, action] / behaviour_policy[state, action]

                if W == 0:
                    break
    return policy, q


def main():
    env = TimeLimit(gym.make('CliffWalking-v0'), 1000)
    actions = ["↑", "→", "↓", "←"]
    policy, q = mc_importance_sampling_first_visit(env)
    argmax_policy = np.argmax(policy, axis=1).reshape(4, -1).tolist()
    argmax_policy = [[actions[x] for x in row] for row in argmax_policy]
    for row in argmax_policy:
        print(row)


if __name__ == '__main__':
    main()