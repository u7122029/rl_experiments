import gymnasium as gym
from right_only_agent import RightOnlyAgent
import matplotlib.pyplot as plt

def show_details(env):
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Min/max positions:", env.env.env.env.min_position, env.env.env.env.max_position)
    print("Min/max speed:", -env.env.env.env.max_speed, env.env.env.env.max_speed)
    print("Goal position:", env.env.env.env.goal_position)
    print("Reward threshold:", env.spec.reward_threshold)
    print("Max episode steps:", env.spec.max_episode_steps)

def play_episode(env, agent, seed=None):
    observation, _ = env.reset(seed=seed)
    agent.reset()

    reward, terminated = 0, 0

    episode_reward, elapsed_steps = 0, 0
    done = False

    positions, velocities = [], []
    while not done:
        positions.append(observation[0])
        velocities.append(observation[1])
        action = agent.step(observation, reward, terminated)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        done = terminated or truncated

    agent.close()
    return episode_reward, elapsed_steps, positions, velocities

def main():
    env = gym.make("MountainCar-v0")
    agent = RightOnlyAgent()
    show_details(env)
    episode_reward, elapsed_steps, positions, velocities = play_episode(env, agent)

    plt.figure()
    plt.title("Position and Velocity of Car")
    plt.xlabel("Steps")
    plt.plot(positions, label="pos")
    plt.plot(velocities, label="vel")
    plt.show()


if __name__ == "__main__":
    main()