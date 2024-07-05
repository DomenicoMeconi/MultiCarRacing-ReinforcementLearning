import argparse
import gym
import torch
import numpy as np
import random
import json
import gym_multi_car_racing
from tqdm import tqdm
import torch
from algo import off_policy_svg0, update
from networks import PolicyNetwork, CriticNetwork
from utils import *
import os


ACTION_DIM = 3  
NUM_AGENTS = 4

######## HYPERPARAMETERS ########
BATCH_SIZE = 64 #64 for 1 agent                
GAMMA = 0.99
POPULATION_SIZE = 12                     
MUTATION_RATE = 0.1             
MUTATION_SCALE = 0.2            
INITIAL_RATING = 1200           
K = 32                                                                              
K_STEPS = 20
TSELECT = 0.35                  
LEARNING_RATE_CRITIC = 0.0002 #0.002 for 1 agent, 0.0008 for 4 agents
LEARNING_RATE_POLICY = [0.000005 for _ in range(POPULATION_SIZE)] #0.000005 for 1 agent
#################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
preview = False

class CriticManager:
    def __init__(self, learning_rate_critic):
        self.critic_network = CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        self.target_critic = CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        self.target_critic.load_state_dict(self.critic_network.state_dict())

class Agent:
    def __init__(self, learning_rate_policy):
        self.policy_network = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        self.target_policy = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        self.replay_buffer = []
        self.frames_processed = 0
        self.eligible = False
        self.rating = INITIAL_RATING
        self.device = device

        self.target_policy.load_state_dict(self.policy_network.state_dict())



def evaluate_agents(agents, environment, population, critic_manager = None, eval=False, update_networks=False):
    indexes = [population.index(agent) for agent in agents]
    print(f"Evaluating agents {indexes}...")

    total_reward = np.zeros(len(agents))
    states = environment.reset()
    states = preprocess(states)
    done = False
    pbar = tqdm(total=1000)

    while not done:
        states_tensor = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        actions = []

        # Get actions from the policy of each agent
        for agent, state_tensor in zip(agents, states_tensor):
            state_tensor = state_tensor.view(1, 3, 96, 96).to(device)

            action = agent.policy_network(state_tensor)
            action = action.cpu().detach().numpy()

            # Clip actions to the environment's action space
            for act in action:
                act[0] = np.clip(act[0] ,-1,1)
                act[1] = np.clip(act[1] ,0,1)
                act[2] = np.clip(act[2],0,1)
            
            actions.append(action)
        
        # Step in the environment
        next_states, rewards, done, info = environment.step(np.array(actions))
        next_states = preprocess(next_states)

        if not eval:
            # For each agent, append the experience to the replay buffer, update the networks and empty the cache
            for agent, state, action, reward, next_state in zip(agents, states_tensor, actions, rewards, next_states):
                state = state.view(1, 3, 96, 96).cpu()
                next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1)
                next_state = next_state.view(1, 3, 96, 96)
                action = torch.tensor(action, dtype=torch.float32)

                agent.replay_buffer.append((state, action, reward, next_state))
                agent.frames_processed += 1
                update_replay_buffer(agent)
                
                if update_networks:
                    update(agent, population, GAMMA, BATCH_SIZE, critic_manager.critic_network, critic_manager.target_critic)
                    torch.cuda.empty_cache()

        pbar.update(1)
        total_reward += rewards
        states = next_states
        if eval:
            environment.render()
        elif preview:
            environment.render()

    pbar.close()
    print(f"Total reward: {total_reward[0]:.2f} || {total_reward[1]:.2f} || {total_reward[2]:.2f} || {total_reward[3]:.2f}")
    return total_reward

def save_checkpoint(population, critic_manager):
    for agent in population:
        if not os.path.exists(f'checkpoint/agent{population.index(agent)}'):
            os.makedirs(f'checkpoint/agent{population.index(agent)}')
            
        torch.save(agent.policy_network, f'checkpoint/agent{population.index(agent)}/policy_network.pth')
        torch.save(agent.target_policy, f'checkpoint/agent{population.index(agent)}/target_policy_network.pth')

    torch.save(critic_manager.target_critic, f'checkpoint/target_critic_network.pth')
    torch.save(critic_manager.critic_network, f'checkpoint/critic_network.pth')

    ratings = [agent.rating for agent in population]
    with open('checkpoint/ratings.json', 'w') as f:
        json.dump(ratings, f)
    
    frame_counts = [agent.frames_processed for agent in population]
    with open('checkpoint/frame_counts.json', 'w') as f:
        json.dump(frame_counts, f)

def pbt_training(population, environment, generations, critic_manager, checkpoint=False, update_networks=False):
    best_rewards = [-1000, -1000] # first reward is the best reward, and so on

    for generation in range(generations):
        print(f"Generation {generation + 1}...")
        print(f"Current best rewards: {best_rewards[0]:.2f} || {best_rewards[1]:.2f}") 

        # Choosing 4 random agents each time until the population is exhausted
        population_copy = population.copy()
        population_generation = []

        # If we load models from a checkpoint, we fill the replay buffers before starting the update of the networks
        if checkpoint and generation > 3:
            update_networks = True

        while len(population_copy) > 0:
            # Select NUM_AGENTS (4) random agents
            agents = random.sample(population_copy, NUM_AGENTS)
            population_generation.extend(agents)

            # Run episode on the selected agents
            rewards = evaluate_agents(agents, 
                                      environment, 
                                      population,
                                      critic_manager=critic_manager, 
                                      update_networks=update_networks)

            # Remove agents from the population copy
            population_copy = [agent for agent in population_copy if agent not in agents]
            
            # Collect rewards
            for agent, reward in zip(agents, rewards):
                agent.reward = reward


        # Save checkpoint
        save_checkpoint(population, critic_manager) 
        
        # Save the policy and critic if we find a new best reward
        if not os.path.exists('best_models'):
            os.makedirs('best_models')

        for agent in population:
            if agent.reward > best_rewards[0]:
                print(f"New best reward found: {agent.reward}")
                best_rewards[1] = best_rewards[0]
                best_rewards[0] = agent.reward
                if best_rewards[1] > -1000:
                    old_best_policy = torch.load('best_models/best_policy_network.pth')
                    torch.save(old_best_policy, 'best_models/second_best_policy_network.pth')
                torch.save(agent.policy_network, 'best_models/best_policy_network.pth')
                
            elif agent.reward > best_rewards[1]:
                print(f"New second best reward found: {agent.reward}")
                best_rewards[1] = agent.reward
                torch.save(agent.policy_network, 'models/second_best_policy_network.pth')

        # For match results, update Elo ratings
        for i in range(0, len(population_generation), 4):
            # Given 4 agents (1,2,3,4) we have that:
            # Team 1 is always composed of agents 1 and 3
            # Team 2 is always composed of agents 2 and 4
            team1_reward = population_generation[i].reward + population_generation[i+2].reward
            team2_reward = population_generation[i+1].reward + population_generation[i+3].reward
            result = 'win' if team1_reward > team2_reward else 'lose' if team1_reward < team2_reward else 'draw'
            update_elo_rating(population_generation[i], population_generation[i+1], result, K)
            update_elo_rating(population_generation[i+2], population_generation[i+3], result, K)

        # Selection and Mutation
        for agent in population:
            if eligible(agent):
                agent2 = random.choice([a for a in population if a != agent])
                if eligible(agent2):
                    selo = 1 / (1 + 10 ** ((agent2.rating - agent.rating) / 400))
                    if selo < TSELECT:
                        print(f"Agent {population.index(agent)} is mutated copying agent {population.index(agent2)}")
                        agent.frames_processed = 0

                        agent.policy_network.load_state_dict(agent2.policy_network.state_dict())
                        agent.policy_network.optimizer.load_state_dict(agent2.policy_network.optimizer.state_dict())

                        agent.target_policy.load_state_dict(agent2.target_policy.state_dict())

                        mutate(agent.policy_network, MUTATION_RATE, MUTATION_SCALE)

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-c', '--checkpoint', action='store_true')
    args = parser.parse_args()

    environment = gym.make("MultiCarRacing-v0", num_agents=NUM_AGENTS, direction='CCW', 
                           use_random_direction=True, backwards_flag=True, 
                           h_ratio=0.25, use_ego_color=True)
    environment.reset()

    for i in range(NUM_AGENTS):
        environment.viewer[i].window.on_key_press = key_press
        environment.viewer[i].window.on_key_release = key_release

    num_generations = 10000

    use_4 = False

    population = initialize_population(POPULATION_SIZE, LEARNING_RATE_POLICY)

    if not args.evaluate:
        critic_manager = CriticManager(LEARNING_RATE_CRITIC)

    if args.train:
        if args.checkpoint:
            print("Loading checkpoint...")                 
            if use_4:
                for agent in population:
                    num = random.randint(0, 3)
                    agent.policy_network = torch.load(f"good/agent{num}/policy_network.pth").to(device)
                    agent.target_policy = torch.load(f"good/agent{num}/target_policy_network.pth").to(device)
                critic_manager.critic_network = torch.load('good/agent0/critic_network.pth').to(device)
                critic_manager.target_critic = torch.load('good/agent0/target_critic_network.pth').to(device)
            
            else:
                ratings = json.load(open('checkpoint/ratings.json'))
                frame_counts = json.load(open('checkpoint/frame_counts.json'))
                for agent in population:
                    agent.policy_network = torch.load(f'checkpoint/agent{population.index(agent)}/policy_network.pth').to(device)
                    agent.target_policy = torch.load(f'checkpoint/agent{population.index(agent)}/target_policy_network.pth').to(device)
                    agent.rating = ratings[population.index(agent)]
                    agent.frames_processed = frame_counts[population.index(agent)]
                critic_manager.critic_network = torch.load('checkpoint/critic_network.pth').to(device)
                critic_manager.target_critic = torch.load('checkpoint/target_critic_network.pth').to(device)

        pbt_training(population, environment, num_generations, critic_manager, checkpoint=True)

    if args.evaluate:
        if args.checkpoint:
            agents = random.sample(population, NUM_AGENTS)
            for agent in agents:
                agent.policy_network = torch.load(f'checkpoint/agent{population.index(agent)}/policy_network.pth').to(device)
                agent.policy_network.eval()
            rewards = evaluate_agents(agents, environment, population, eval=True)
        else:
            for agent in population[:NUM_AGENTS]:
                if population.index(agent) == 0 or population.index(agent) == 1:
                    agent.policy_network = torch.load('best_models/best_policy_network.pth')
                else:
                    agent.policy_network = torch.load('best_models/second_best_policy_network.pth') 

                agent.policy_network.eval()
            rewards = evaluate_agents(population[:NUM_AGENTS], environment, population, eval=True)

        if rewards[0] + rewards[1] > rewards[1] + rewards[3]:
            print("Team Blue won!")
        else:
            print("Team Red won!")

if __name__ == "__main__":
    main()
