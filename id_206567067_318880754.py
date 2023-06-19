import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.num_rounds = num_rounds
        self.checking_point = min(10000, num_rounds)
        self.rewards = np.zeros((num_users, num_arms))
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
        self.start_saving_arms = False
        self.safety_list = np.ones(num_arms)  # arm[i] = 1 => arm 1 has not yet passed the threshold
        self.phases_passed = 0
        self.active_arms = set(np.arange(num_arms))
        self.picks_till_safe = np.array([thresh for thresh in arms_thresh])
        self.rounds_left_in_phase = phase_len
        self.saving_protocol_off = False
        self.estimated_ERM = np.zeros(self.UCB.shape)

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """

        arms_tried = int(sum(self.num_chosen[user_context][:]))

        if self.rounds_elapsed == self.checking_point:  # at the checking point, see if we need to still be saving arms
            self.estimate_ERM()
            self.keep_saving_protocol()  # keep saving protocol on or not???

        if arms_tried < self.num_arms:  # initialize each arm
            chosen_arm = arms_tried

        elif self.start_saving_arms and not self.saving_protocol_off:
            self.safety_list = (self.picks_till_safe > 0).astype(int)
            at_risk_amount = np.sum(self.safety_list)
            if int(at_risk_amount) == 1:
                chosen_arm = np.where(self.safety_list == 1)[0][0]
            elif at_risk_amount > 1:
                picks_left = np.array([picks if picks > 0 else np.NINF for picks in self.picks_till_safe])
                sorted_arms = np.lexsort((-self.UCB[user_context, :], -picks_left))
                # sort by picks left, break ties by ucb, ties are broken by min, so add minus to flip
                chosen_arm = sorted_arms[0]
            else:
                chosen_arm = np.argmax(self.UCB[user_context][:])
        else:
            chosen_arm = np.argmax(self.UCB[user_context][:])

        chosen_arm = self.assert_arm_is_active(chosen_arm)
        self.most_recent_user = user_context
        self.most_recent_arm = chosen_arm
        self.picks_till_safe[chosen_arm] -= 1
        self.num_chosen[user_context][int(chosen_arm)] += 1
        self.exposure_list[int(chosen_arm)] += 1
        return chosen_arm

    def estimate_ERM(self):
        """
        estimate ERM based on UCB and rewards so far
        """
        self.estimated_ERM = np.copy(self.UCB)
        self.estimated_ERM[self.rewards == 0] = self.rewards[self.rewards == 0]
        self.estimated_ERM = np.round(self.estimated_ERM, 2)

    def keep_saving_protocol(self):
        """
        estimate projected reward for each arm and with protocol, choose if to keep protocol on or not
        """
        single_arms = np.matmul(self.users_distribution, self.estimated_ERM)  # reward proportion per arm
        rewards_perarm_perphase = [proportion * picks for proportion, picks in zip(single_arms, self.arms_thresh)]
        remaining_time = self.phase_len - np.sum(self.arms_thresh)
        time_distribution = self.users_distribution * remaining_time
        remaining_reward = np.max(self.estimated_ERM, axis=1) @ time_distribution
        single_arms_total_reward = single_arms * self.num_rounds
        rewards_per_phase_saving_arms = np.sum(rewards_perarm_perphase) + remaining_reward
        total_reward_saving = rewards_per_phase_saving_arms * self.num_rounds / self.phase_len
        total_rewards = np.append(single_arms_total_reward, total_reward_saving)
        choice = np.argmax(total_rewards)
        if choice != self.num_arms:
            self.saving_protocol_off = True
        # calc total reward per outcome(keep 1 arm or all) pick max option
        # either keep arms and then have done_saving= false indefinitely
        # or choose an arm and kick the rest

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        user = self.most_recent_user
        arm = self.most_recent_arm
        self.rewards[user][arm] += reward
        mu = self.rewards[user][arm] / self.num_chosen[user][arm]
        self.UCB[user][arm] = mu + (2 * np.log(self.Ts[user]) / self.num_chosen[user][arm] * self.users_distribution[
            user]) ** 0.5
        if (self.rounds_elapsed + 1) % self.phase_len == 0:
            self.deactivate_arms()
        self.rounds_elapsed += 1
        self.rounds_left_in_phase -= 1
        self.check_threshold(arm)
        self.update_saving_status()

    def update_saving_status(self):
        """
        see if critical time has started and we need to activate the protocol
        """
        if np.sum(self.picks_till_safe[self.picks_till_safe > 0]) + 1 == self.rounds_left_in_phase \
                and sum(self.safety_list) > 0 and self.start_saving_arms is False:
            self.start_saving_arms = True

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
        self.start_saving_arms = False

    def check_threshold(self, arm):
        """
        updates safety list if arm has reached the threshold
        """
        if self.picks_till_safe[arm] <= 0:
            self.safety_list[arm] = 0

    def assert_arm_is_active(self, arm):
        if arm not in self.deactivated_arms:
            return arm
        else:
            print('oops')
            return random.choice([x for x in self.active_arms])
