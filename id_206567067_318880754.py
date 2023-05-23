import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

NUM_ROUNDS = 10


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.rewards = np.zeros((num_users, num_arms))  # each row represents an arm, col 1 is sum, col 2 is num
        # chosen
        self.num_chosen = np.zeros((num_users, num_arms))
        self.UCB = np.zeros((num_users, num_arms))
        self.Ts = [num_rounds * user_probability for user_probability in users_distribution]
        self.most_recent_user = None
        self.most_recent_arm = None
        self.phase_len = phase_len
        self.rounds_elapsed = 0
        self.deactivated = set()
        self.num_arms = num_arms
        self.arms_thresh = arms_thresh
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.
        self.users_distribution = users_distribution
        self.save_me = None

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        arms_tried = int(sum(self.num_chosen[user_context][:]))
        if self.save_me is not None:
            chosen_arm = self.save_me
        elif arms_tried < self.num_arms:
            chosen_arm = arms_tried
        else:
            chosen_arm = np.argmax(self.UCB[user_context][:])

        weighted_reward = [np.average(self.UCB[:][arm], weights=self.users_distribution)
                           for arm in range(self.num_arms)]
        best_weighted_reward = np.argmax(np.array(weighted_reward))
        if best_weighted_reward != chosen_arm and self.save_me is None:
            if self.arms_thresh[best_weighted_reward] - self.exposure_list[best_weighted_reward] == (
                    self.phase_len - ((self.rounds_elapsed + 1) % self.phase_len)):
                self.save_me = best_weighted_reward
                chosen_arm = best_weighted_reward
                print(f"saving arm: {chosen_arm} round{self.rounds_elapsed}")


        self.most_recent_user = user_context
        self.most_recent_arm = chosen_arm
        self.num_chosen[user_context][int(chosen_arm)] += 1
        self.exposure_list[int(chosen_arm)] += 1
        if (self.rounds_elapsed + 1) % self.phase_len == 0:
            self.save_me = None
            self.deactivate_arms()
        self.rounds_elapsed += 1
        return chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        user = self.most_recent_user
        arm = self.most_recent_arm
        self.rewards[user][arm] += reward
        mu = self.rewards[user][arm] / self.num_chosen[user][arm]
        self.UCB[user][arm] = mu + (2 * np.log(self.Ts[user]) / self.num_chosen[user][arm]) ** 0.5

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_123456789_987654321"

    def deactivate_arms(self):
        """
        this function is called at the end of each phase and deactivates arms that haven't gotten enough exposure
        (deactivated arm == arm that has departed)
        """
        for arm in range(self.num_arms):
            if self.exposure_list[arm] < self.arms_thresh[arm]:
                if arm not in self.deactivated: print("\n arm " + str(arm) + f" is deactivated!")
                self.deactivated.add(arm)
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.
        self.save_me = None
