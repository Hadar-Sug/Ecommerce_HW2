import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        # TODO: Decide what/if to store. Could be used in the future
        pass

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: This is your place to shine. Go crazy!
        # idea - have a class for each user, and have an algorithm that store the date for about each arm
        # each arm is a producer
        return 0

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        pass

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_123456789_987654321"


class User:
    def __init__(self, id):
        self.id = id
        self.algorithm = None


class UCB:
    def __init__(self, num_rounds, num_arms, arms_thresh: np.array, reward_dist: np.array):
        """
        :input: num_rounds - number of rounds
                        phase_len - number of rounds at each phase
                        num_arms - number of content providers
                        num_users - number of users
                        arms_thresh - the exposure demands of the content providers (np array of size num_arms)
                        reward_dist - expected reward matrix (2D np array of size (num_arms x num_users))
                              ERM[i][j] is the expected reward of user i from content provider j
        """
