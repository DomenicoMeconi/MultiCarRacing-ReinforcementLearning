import argparse
import gym
import torch
import numpy as np
import random
import json
import gym_multi_car_racing
from tqdm import tqdm
from algo import off_policy_svg0, off_policy_svg1
from networks import PolicyNetwork, CriticNetwork


ACTION_DIM = 3  
NUM_AGENTS = 4

######## HYPERPARAMETERS ########
BATCH_SIZE = 512                
GAMMA = 0.99                    
MUTATION_RATE = 0.1             
MUTATION_SCALE = 0.2            
INITIAL_RATING = 1200           
K = 32                           
TSELECT = 0.35                  
LEARNING_RATE_CRITIC = 0.0007
#initializing 20 different learning rates for the policy network, from 0.00001 with a step of 0.00001 
LEARNING_RATE_POLICY =  [0.000001 + i * 0.000001 for i in range(20)]
#################################

class Agent:
    def __init__(self, algo, learning_rate_policy, critic):
        self.policy_network = PolicyNetwork(ACTION_DIM, learning_rate_policy)
        self.critic_network = critic
        self.replay_buffer = []
        self.frames_processed = 0
        self.eligible = False
        self.rating = INITIAL_RATING

def initialize_population(pop_size, algo):
    critic = CriticNetwork(algo, ACTION_DIM, LEARNING_RATE_CRITIC)
    population = [Agent(algo,LEARNING_RATE_POLICY[i], critic) for i in range(pop_size)]
    return population

def mutate(network):
    for param in network.parameters():
        if random.random() < MUTATION_RATE:
            param.data += MUTATION_SCALE * torch.randn_like(param.data)

def update_replay_buffer(agent):
    if len(agent.replay_buffer) > 2500:
        agent.replay_buffer = agent.replay_buffer[-2500:]


def eligible(agent):
    if agent.eligible:
        return agent.frames_processed > 40000
    else:
        if agent.frames_processed > 200000:
            agent.eligible = True
            return True
        else:   
            return False

def evaluate_agents(agents, environment, population, render=False):
    indexes = [population.index(agent) for agent in agents]
    print(f"Evaluating agents {indexes}...")
    total_reward = 0
    states = environment.reset()
    done = False
    while not done:
        states_tensor = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to [agents, channels, height, width]
        actions = []
        # Select each agent with its corresponding state and compute action based with the current policy
        for agent,state_tensor in zip(agents, states_tensor): 
            state_tensor = state_tensor.view(1, 3, 96, 96)
            action = agent.policy_network(state_tensor).detach().numpy()
            for act in action:
                act[0] = np.clip(act[0],-1,1)
                act[1] = np.clip(act[1],0,1)
                act[2] = np.clip(act[2],0,1)
            actions.append(action)
        next_states, rewards, done, info = environment.step(np.array(actions))

        # Append the experience to the replay buffer for each agent
        for agent, state, action, reward, next_state in zip(agents, states_tensor, actions, rewards, next_states):
            state = state.view(1, 3, 96, 96)
            next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1)
            next_state = next_state.view(1, 3, 96, 96)
            action = torch.tensor(action, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)

            agent.replay_buffer.append((state, action, reward, next_state))
            agent.frames_processed += 1
                
        total_reward += rewards
        states = next_states
        if render:
            environment.render()

    print(f"Total reward: {total_reward[0]:.2f} || {total_reward[1]:.2f} || {total_reward[2]:.2f} || {total_reward[3]:.2f}")
    return total_reward

def update_elo_rating(agent_i, agent_j, result):
    Ri = agent_i.rating
    Rj = agent_j.rating
    selo = 1 / (1 + 10 ** ((Rj - Ri) / 400))
    s = 1 if result == 'win' else 0 if result == 'lose' else 0.5
    agent_i.rating += K * (s - selo)
    agent_j.rating -= K * (s - selo)

def pbt_training(population, environment, generations, algo):
    best_rewards = [-1000, -1000] # first reward is the best reward, and so on

    for generation in range(generations):
        print(f"Generation {generation + 1}...")
        print(f"Current best rewards: {best_rewards[0]:.2f} || {best_rewards[1]:.2f}")

        if len(population) < 5 and (generation + 1) % 5 == 0:
            for agent in population:
                torch.save(agent.policy_network, f'checkpoint_4_agents/policy_network_{population.index(agent)}.pth')
                torch.save(agent.critic_network, f'checkpoint_4_agents/critic_network.pth')
            
        elif (generation + 1) % 5 == 0:
            ratings = [agent.rating for agent in population]
            frame_counts = [agent.frames_processed for agent in population]
            for agent in population:
                torch.save(agent.policy_network, f'checkpoint/policy_network_{population.index(agent)}.pth')
                torch.save(agent.critic_network, f'checkpoint/critic_network.pth')
            with open('checkpoint/ratings.json', 'w') as f:
                json.dump(ratings, f)
            with open('checkpoint/frame_counts.json', 'w') as f:
                json.dump(frame_counts, f)
            

        # Choosing 4 random agents each time until the population is exhausted
        population_copy = population.copy()
        population_generation = []
        while len(population_copy) > 0:
            agents = random.sample(population_copy, 4)
            population_generation.extend(agents)

            # Run episode on the selected agents
            rewards = evaluate_agents(agents, environment, population)

            # Remove agents from the population copy
            population_copy = [agent for agent in population_copy if agent not in agents]
            
            # Collect rewards and update critic and policy networks
            
            for agent, reward in zip(agents, rewards):
                agent.reward = reward

        # Save the policy and critic if we find a new best reward
        for agent in population:
            if agent.reward > best_rewards[0]:
                print(f"New best reward found: {agent.reward}")
                best_rewards[1] = best_rewards[0]
                best_rewards[0] = agent.reward
                if best_rewards[1] > -1000:
                    old_best_policy = torch.load('models/best_policy_network.pth')
                    old_best_critic = torch.load('models/best_critic_network.pth')
                    torch.save(old_best_policy, 'models/second_best_policy_network.pth')
                    torch.save(old_best_critic, 'models/second_best_critic_network.pth')

                torch.save(agent.policy_network, 'models/best_policy_network.pth')
                torch.save(agent.critic_network, 'models/best_critic_network.pth')
                
            elif agent.reward > best_rewards[1]:
                print(f"New second best reward found: {agent.reward}")
                best_rewards[1] = agent.reward
                torch.save(agent.policy_network, 'models/second_best_policy_network.pth')
                torch.save(agent.critic_network, 'models/second_best_critic_network.pth')

        for agent in tqdm(population_generation, desc="Updating networks"):
            
            if algo == 'svg0':
                off_policy_svg0(agent, BATCH_SIZE, GAMMA)
            elif algo == 'svg1':
                off_policy_svg1(agent, BATCH_SIZE, GAMMA)

            update_replay_buffer(agent)
            
                        

        # For match results, update Elo ratings
        print("Updating Elo ratings...")
        for i in range(0, len(population_generation), 4):
            # Given 4 agents (1,2,3,4) we have that:
            # Team 1 is always composed of agents 1 and 3
            # Team 2 is always composed of agents 2 and 4
            team1_reward = population_generation[i].reward + population_generation[i+2].reward
            team2_reward = population_generation[i+1].reward + population_generation[i+3].reward
            result = 'win' if team1_reward > team2_reward else 'lose' if team1_reward < team2_reward else 'draw'
            update_elo_rating(population_generation[i], population_generation[i+1], result)
            update_elo_rating(population_generation[i+2], population_generation[i+3], result)
        print("Elo ratings updated!")

        # Selection and Mutation
        print("Searching for mutations...")
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
                        agent.policy_network.scheduler.load_state_dict(agent2.policy_network.scheduler.state_dict())

                        agent.critic_network.load_state_dict(agent2.critic_network.state_dict())
                        agent.critic_network.optimizer.load_state_dict(agent2.critic_network.optimizer.state_dict())
                        agent.critic_network.scheduler.load_state_dict(agent2.critic_network.scheduler.state_dict())

                        mutate(agent.policy_network)
        print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-c', '--checkpoint', action='store_true')
    args = parser.parse_args()

    render_mode = "human" if args.evaluate else None

    environment = gym.make("MultiCarRacing-v0", num_agents=NUM_AGENTS, direction='CCW', 
                           use_random_direction=True, backwards_flag=True, 
                           h_ratio=0.25, use_ego_color=True, render_mode = render_mode)

    population_size = 4 #20
    num_generations = 1000
    algo = 'svg0'

    population = initialize_population(population_size, algo)

    if args.train:
        torch.manual_seed(0)
        if args.checkpoint:
            print("Loading checkpoint...")
            ratings = json.load(open('checkpoint/ratings.json'))
            frame_counts = json.load(open('checkpoint/frame_counts.json'))
            #choose the 4 index of the highest rated agents
            if population_size == 4:
                indexes = np.argsort(ratings)[-4:]
                for agent,index in zip(population,indexes):
                    agent.policy_network = torch.load(f'checkpoint/policy_network_{index}.pth')
                    agent.critic_network = torch.load('checkpoint/critic_network.pth')
                    agent.rating = ratings[index]
                    agent.frames_processed = frame_counts[index]

            for agent in population:
                agent.policy_network = torch.load(f'checkpoint/policy_network_{population.index(agent)}.pth')
                agent.critic_network = torch.load('checkpoint/critic_network.pth')
                agent.rating = ratings[population.index(agent)]
                agent.frames_processed = frame_counts[population.index(agent)]

        pbt_training(population, environment, num_generations, algo)

    if args.evaluate:
        if args.checkpoint:
            agents = random.sample(population, 4)
            for agent in agents:
                agent.policy_network = torch.load(f'checkpoint/policy_network_{population.index(agent)}.pth')
                agent.policy_network.eval()
            rewards = evaluate_agents(agents, environment, population, render=True)

        else:
            for agent in population[:NUM_AGENTS]:
                if random.random() < 0.5:
                    agent.policy_network = torch.load('models/best_policy_network.pth')
                else:
                    agent.policy_network = torch.load('models/second_best_policy_network.pth')  
                agent.policy_network.eval()
            rewards = evaluate_agents(population[:NUM_AGENTS], environment, population, render=True)

        if rewards[0] + rewards[1] > rewards[1] + rewards[3]:
            print("Team Blue won!")
        else:
            print("Team Red won!")

if __name__ == "__main__":
    main()
