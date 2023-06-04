import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

NUM_ROUNDS = 10

np.random.seed(318880754)


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.saving_ratio = []
        self.checking_point = min(10000, num_rounds)
        self.ratio_lim = 30
        self.rewards = np.zeros((num_users, num_arms))  # each row represents an arm, col 1 is sum, col 2 is num
        # chosen
        self.num_chosen = np.zeros((num_users, num_arms))
        self.UCB = np.zeros((num_users, num_arms))
        self.Ts = [num_rounds * user_probability for user_probability in users_distribution]
        self.most_recent_user = None
        self.most_recent_arm = None
        self.phase_len = phase_len
        self.rounds_elapsed = 0
        self.deactivated_arms = set()
        self.num_arms = num_arms
        self.arms_thresh = arms_thresh
        self.exposure_list = np.zeros(self.num_arms)  # maybe redundant # initiate the exposure list for the next phase.
        self.users_distribution = users_distribution
        # self.best_arm_score = {"arm": -1, "score": -1}

        self.save_arms = False
        self.safety_list = np.ones(num_arms)  # arm[i] = 1 => arm 1 has not yet passed the threshold
        self.phases_passed = 0
        self.active_arms = set(np.arange(num_arms))
        self.picks_till_safe = np.array([thresh for thresh in arms_thresh])
        self.rounds_left_in_phase = phase_len
        self.done_saving = False
        self.intervention_count = 0

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """

        arms_tried = int(sum(self.num_chosen[user_context][:]))
        # if self.save_me is not None:
        #     chosen_arm = self.save_me
        if arms_tried < self.num_arms:  # try each arm
            chosen_arm = arms_tried
        # else:
        #     chosen_arm = np.argmax(self.UCB[user_context][:])
        elif self.save_arms and not self.done_saving:
            if self.rounds_elapsed > self.checking_point:  # UCB stabilized, lets see how much we're intervening
                intervention_ratio = np.average(self.saving_ratio)
                if intervention_ratio > self.ratio_lim:  # or self.lost_arms():
                    self.done_saving = True
            # hyperparameter checks
            self.safety_list = (self.picks_till_safe > 0).astype(int)
            at_risk_amount = np.sum(self.safety_list)
            if int(at_risk_amount) == 1:
                chosen_arm = np.where(self.safety_list == 1)[0][0]
            elif at_risk_amount > 1:
                picks_left = np.array([picks if picks > 0 else np.NINF for picks in self.picks_till_safe])
                sorted_arms = np.lexsort((-self.UCB[user_context, :], -picks_left))  # sort by picks left,
                # break ties by ucb, ties are broken by min, so add minus to flip
                chosen_arm = sorted_arms[0]
            else:
                chosen_arm = np.argmax(self.UCB[user_context][:])
        else:
            chosen_arm = np.argmax(self.UCB[user_context][:])

        # weighted_reward = np.average(self.UCB, weights=self.users_distribution, axis=0)
        #
        # best_weighted_arm = self.best_arm_score["arm"]
        # if best_weighted_arm != chosen_arm and self.save_me is None:
        #     if self.arms_thresh[best_weighted_arm] - self.exposure_list[best_weighted_arm] == (
        #             self.phase_len - ((self.rounds_elapsed + 1) % self.phase_len)):
        #         self.save_me = best_weighted_arm
        #         chosen_arm = best_weighted_arm
        #         print(f"saving arm: {chosen_arm} round{self.rounds_elapsed}")

        self.most_recent_user = user_context
        self.most_recent_arm = chosen_arm
        self.picks_till_safe[chosen_arm] -= 1
        self.num_chosen[user_context][int(chosen_arm)] += 1
        self.exposure_list[int(chosen_arm)] += 1
        chosen_arm = self.assert_arm_is_active(chosen_arm)
        return chosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # print(f"reward at round {self.rounds_elapsed}= {reward}")
        # print(f'reward:     {reward}')
        user = self.most_recent_user
        arm = self.most_recent_arm
        self.rewards[user][arm] += reward
        mu = self.rewards[user][arm] / self.num_chosen[user][arm]
        # TODO: test
        self.UCB[user][arm] = mu + (2 * np.log(self.Ts[user]) / self.num_chosen[user][arm] * self.users_distribution[
            user]) ** 0.5
        if (self.rounds_elapsed + 1) % self.phase_len == 0:
            self.deactivate_arms()
        self.rounds_elapsed += 1
        self.rounds_left_in_phase -= 1
        self.check_threshold(arm)
        self.update_saving_status()

        # arm_score = np.average(self.UCB[:, arm], weights=self.users_distribution, axis=0)
        # if arm_score > self.best_arm_score["score"]:
        #     self.best_arm_score["arm"] = arm
        #     self.best_arm_score["score"] = arm_score

    # TODO: once we start saving,dont stop
    def update_saving_status(self):
        if np.sum(self.picks_till_safe[self.picks_till_safe > 0]) + 1 == self.rounds_left_in_phase \
                and sum(self.safety_list) > 0:
            self.save_arms = True
            # if self.rounds_elapsed > self.checking_point:
            self.saving_ratio.append(100 * self.rounds_left_in_phase / self.phase_len)

    def lost_arms(self):
        if len(self.deactivated_arms) == 0:
            return False
        else:
            return True

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_206567067_318880754"

    def deactivate_arms(self):
        """
        this function is called at the end of each phase and deactivates arms that haven't gotten enough exposure
        (deactivated arm == arm that has departed)
        and resets class objects that rely on phases
        """
        if len(self.active_arms) > 1:
            for arm in range(self.num_arms):
                if self.exposure_list[arm] < self.arms_thresh[arm]:
                    if arm not in self.deactivated_arms:
                        print("\n arm " + str(arm) + f" is deactivated!")
                        self.deactivated_arms.add(arm)
                        self.UCB[:, arm] = -1
                        self.active_arms.remove(arm)
                        self.picks_till_safe[arm] = -1
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.
        self.picks_till_safe = np.array([thresh for thresh in self.arms_thresh])
        self.rounds_left_in_phase = self.phase_len
        self.safety_list = np.ones(self.num_arms)
        self.save_arms = False

    def check_threshold(self, arm):
        if self.picks_till_safe[arm] <= 0:
            self.safety_list[arm] = 0

    def assert_arm_is_active(self, arm):
        if arm not in self.deactivated_arms:
            return arm
        else:
            print('oops')
            return random.choice([x for x in self.active_arms])
