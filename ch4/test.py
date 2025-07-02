import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm


def play_episode(env, policy):
    data = []
    state, _ = env.reset()
    done = False
    while not done:
        action = np.random.choice(range(env.action_space.n), p=policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        data.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return data


def mc_importance_sampling_first_visit(env, no_episodes=int(1e5), gamma=1.0, epsilon=0.1):
    no_states = env.observation_space.n
    no_actions = env.action_space.n

    # Initialize Q and cumulative weights C
    q = np.zeros((no_states, no_actions))
    c = np.zeros((no_states, no_actions))

    # Initialize behavior policy as uniform random (exploratory)
    behavior_policy = np.ones((no_states, no_actions)) / no_actions

    # Initialize target policy as epsilon-soft greedy w.r.t q (start uniform)
    policy = np.ones((no_states, no_actions)) / no_actions

    for _ in tqdm(range(no_episodes)):
        episode = play_episode(env, behavior_policy)

        G = 0
        W = 1.0

        # Process episode in reverse for first-visit MC
        visited = set()
        for state, action, reward in reversed(episode):
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
                if behavior_policy[state, action] == 0:
                    break

                W *= policy[state, action] / behavior_policy[state, action]

                if W == 0:
                    break

    return policy, q


def main():
    env = gym.make('CliffWalking-v0')
    actions = ["↑", "→", "↓", "←"]
    policy, q = mc_importance_sampling_first_visit(env, no_episodes=100000, epsilon=0.1)

    # Print policy arrows in grid form
    argmax_policy = np.argmax(policy, axis=1).reshape(4, -1)
    for row in argmax_policy:
        print([actions[a] for a in row])


if __name__ == "__main__":
    main()