#  MIT License
#
#  Copyright (c) 2020 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import numpy as np

from collections import deque

from agent import ControlAgent


class DDPG:

    def __init__(self, env, agent: ControlAgent) -> None:
        self.env = env
        self.agent = agent  # type: ControlAgent

        # get the default brain
        self.brain_name = env.brain_names[0]

    def train(self, n_episodes=1000, max_t=10000):

        scores = []
        scores_last = deque(maxlen=100)
        solved = False

        for i_episode in range(1, n_episodes + 1):
            # Reset environment, state and scores
            self.agent.reset()
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations

            # We will tracking scores for each agent
            episode_scores = np.zeros(self.agent.n_agents)

            for t in range(max_t):
                # A single step of interaction with the environment for each agent
                actions = self.agent.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.agent.step(states, actions, rewards, next_states, dones)

                # Sum up rewards separately for each agent
                episode_scores += np.array(rewards)

                # Prepare for next timestep of iteraction
                # new states become the current states
                states = next_states

                # Check if any of the agents has finished. Finish to keep all
                # trajectories in this batch the same size.
                if np.any(dones):
                    break

            # Update scores
            episode_score = np.mean(episode_scores)
            scores_last.append(episode_score)
            scores.append(episode_score)

            print("\rEpisode {} average score: {:.2f}".format(i_episode, np.mean(episode_score)))

            if i_episode % 10 == 0:
                print("\r     Last 100 episodes {} average score: {:.2f}".format(i_episode, np.mean(scores_last)))

            if np.mean(scores_last) > 30.0 and not solved:
                print("Environment solved in {} episodes. Average score of all agents over the "
                      "last 100 episodes: {:.2f}".format(i_episode, np.mean(scores_last)))
                solved = True

            self.agent.save("checkpoint{}.pth")

        return scores

    def test(self, n_episodes=10, max_t=1000):

        scores = []

        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            states = env_info.vector_observations
            self.agent.reset()

            # We will tracking scores for each agent
            episode_scores = np.zeros(self.agent.n_agents)

            for t in range(max_t):
                actions = self.agent.act(states, add_noise=False)
                env_info = self.env.step(actions)[self.brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # Sum up rewards separately for each agent
                episode_scores += np.array(rewards)

                # Prepare for next timestep of iteraction
                # new states become the current states
                states = next_states

                # Check if any of the agents has finished. Finish to keep all
                # trajectories in this batch the same size.
                if np.any(dones):
                    break

            scores.append(np.mean(episode_scores))

        return scores
