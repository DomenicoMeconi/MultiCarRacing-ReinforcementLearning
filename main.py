import argparse
import gym
import torch
import numpy as np
import random
import json
import gym_multi_car_racing
from tqdm import tqdm
from algo import off_policy_svg0, update
import wandb
from networks import PolicyNetwork, CriticNetwork
import matplotlib.pyplot as plt
from pyglet.window import key
import gc

import torch
import torch.backends.cudnn as cudnn

ACTION_DIM = 3  
NUM_AGENTS = 4

######## HYPERPARAMETERS ########
BATCH_SIZE = 64 #64 for 1 agent                
GAMMA = 0.99                    
MUTATION_RATE = 0.1             
MUTATION_SCALE = 0.2            
INITIAL_RATING = 1200           
K = 32                                                                              
K_STEPS = 20
TSELECT = 0.35                  
LEARNING_RATE_CRITIC = [0.0002 for _ in range(1)] #0.002 for 1 agent, 0.0008 for 4 agents
#initializing 20 different learning rates for the policy network, from 0.00001 with a step of 0.00001 
LEARNING_RATE_POLICY = [0.000005 for _ in range(12)] #0.000005 for 1 agent
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
preview = False

critic_network = CriticNetwork(ACTION_DIM, LEARNING_RATE_CRITIC[0]).to(device)
target_critic = CriticNetwork(ACTION_DIM, LEARNING_RATE_CRITIC[0]).to(device)

target_critic.load_state_dict(critic_network.state_dict())

def key_press(k, mod):
    if k == key.SPACE:
        global preview
        preview = True


def key_release(k, mod):
    if k == key.SPACE:
        global preview
        preview = False

class Agent:
    def __init__(self, learning_rate_policy):
        self.policy_network = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        self.target_policy = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        #self.critic_network = critic_network #CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        #self.target_critic = target_critic #CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        self.update = 0
        self.replay_buffer = []
        self.frames_processed = 0
        self.eligible = False
        self.rating = INITIAL_RATING
        self.device = device

        #self.target_critic.load_state_dict(self.critic_network.state_dict())
        self.target_policy.load_state_dict(self.policy_network.state_dict())   

def initialize_population(pop_size):
    population = [Agent(LEARNING_RATE_POLICY[i]) for i in range(pop_size)]
    return population

def mutate(network):
    for param in network.parameters():
        if random.random() < MUTATION_RATE:
            param.data += MUTATION_SCALE * torch.randn_like(param.data)

def update_replay_buffer(agent):
    if len(agent.replay_buffer) > 4000:
        agent.replay_buffer = agent.replay_buffer[-4000:]


def eligible(agent):
    if agent.eligible:
        return agent.frames_processed > 25000
    else:
        if agent.frames_processed > 100000:
            agent.eligible = True
            return True
        else:   
            return False
def replace_color(data, original, new_value):
    r1, g1, b1 = original
    r2, g2, b2 = new_value

    red, green, blue = data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:,:3][mask] = [r2, g2, b2]
    return data

def preprocess(img, greyscale=False):
    img = img.copy() 

    # Unify grass color
    img = replace_color(img, original=(102, 229, 102), new_value=(102, 204, 102))

    if greyscale:
        img = img.mean(dim=3, keepdim=True)

    # Scale from 0 to 1
    img = img / img.max()

    # Unify track color
    img[(img > 0.411) & (img < 0.412)] = 0.4
    img[(img > 0.419) & (img < 0.420)] = 0.4

    # Change color of kerbs
    game_screen = img[:, 0:83, :]
    game_screen[game_screen == 1] = 0.80
    
    return img

def evaluate_agents(agents, environment, population, eval=False, checkpoint=False):
    indexes = [population.index(agent) for agent in agents]
    print(f"Evaluating agents {indexes}...")
    total_reward = np.zeros(len(agents))
    states = environment.reset()
    states = preprocess(states)
    done = False
    it = 0
    pbar = tqdm(total=1000)
    while not done:
        it += 1

        states_tensor = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        actions = []
        # Select each agent with its corresponding state and compute action based with the current policy
        for agent, state_tensor in zip(agents, states_tensor):
            state_tensor = state_tensor.view(1, 3, 96, 96).to(device)
            # Infer eta_k using the critic network
            #generate an action using a normal distribution

            action = agent.policy_network(state_tensor)
            action = action.cpu().detach().numpy()
            
            for act in action:
                act[0] = np.clip(act[0] ,-1,1)
                act[1] = np.clip(act[1] ,0,1)
                act[2] = np.clip(act[2],0,1)
                act[0] = act[0]
                act[1] = act[1]
                act[2] = act[2]

            
            actions.append(action)
        next_states, rewards, done, info = environment.step(np.array(actions))

        next_states = preprocess(next_states)

        # Append the experience to the replay buffer for each agent
        if not eval:
            for agent, state, action, reward, next_state in zip(agents, states_tensor, actions, rewards, next_states):
                state = state.view(1, 3, 96, 96).cpu()
                next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1)
                next_state = next_state.view(1, 3, 96, 96)
                action = torch.tensor(action, dtype=torch.float32)

                agent.replay_buffer.append((state, action, reward, next_state))
                agent.frames_processed += 1
                update_replay_buffer(agent)
                
                if not checkpoint:
                    update(agent, population, GAMMA, BATCH_SIZE, critic_network, target_critic)
                    torch.cuda.empty_cache()

        '''
                agent.update += 1
                if agent.update % 100 == 0:
                    agent.target_policy.load_state_dict(agent.policy_network.state_dict())
                    agent.target_critic.load_state_dict(agent.critic_network.state_dict())
                off_policy_svg0(agent, GAMMA, K_STEPS)
                '''
        '''
        if rewards[0] < 0:
            it += 1
        else:
            it = 0
        if it > 200:
            done = True
        '''
        pbar.update(1)
        total_reward += rewards
        states = next_states
        if eval:
            environment.render()
        elif preview:
            environment.render()

    pbar.close()
            
    if NUM_AGENTS != 1:
        print(f"Total reward: {total_reward[0]:.2f} || {total_reward[1]:.2f} || {total_reward[2]:.2f} || {total_reward[3]:.2f}")
    else:
        print(f"Total reward: {total_reward[0]:.2f}")
    return total_reward


def update_elo_rating(agent_i, agent_j, result):
    Ri = agent_i.rating
    Rj = agent_j.rating
    selo = 1 / (1 + 10 ** ((Rj - Ri) / 400))
    s = 1 if result == 'win' else 0 if result == 'lose' else 0.5
    agent_i.rating += K * (s - selo)
    agent_j.rating -= K * (s - selo)

def pbt_training(population, environment, generations, checkpoint=False):
    best_rewards = [858, 793] # first reward is the best reward, and so on

    for generation in range(generations):
        print(f"Generation {generation + 1}...")
        print(f"Current best rewards: {best_rewards[0]:.2f} || {best_rewards[1]:.2f}")

        if len(population) < 2 and (generation + 1) % 5 == 0:
            ratings = [agent.rating for agent in population]
            frame_counts = [agent.frames_processed for agent in population]
            for agent in population:
                torch.save(agent.policy_network, f'checkpoint_4_agents/policy_network_{population.index(agent)}.pth')
                torch.save(agent.critic_network, f'checkpoint_4_agents/critic_network{population.index(agent)}.pth')
                torch.save(agent.target_policy, f'checkpoint_4_agents/target_policy_network_{population.index(agent)}.pth')
                torch.save(agent.target_critic, f'checkpoint_4_agents/target_critic_network{population.index(agent)}.pth')
            with open('checkpoint_4_agents/ratings.json', 'w') as f:
                json.dump(ratings, f)
            with open('checkpoint_4_agents/frame_counts.json', 'w') as f:
                json.dump(frame_counts, f)
            
        elif len(population) >= 2:
            ratings = [agent.rating for agent in population]
            frame_counts = [agent.frames_processed for agent in population]
            for agent in population:
                torch.save(agent.policy_network, f'checkpoint/agent{population.index(agent)}/policy_network.pth')
                torch.save(agent.target_policy, f'checkpoint/agent{population.index(agent)}/target_policy_network.pth')
            torch.save(target_critic, f'checkpoint/target_critic_network.pth')
            torch.save(critic_network, f'checkpoint/critic_network.pth')

            with open('checkpoint/ratings.json', 'w') as f:
                json.dump(ratings, f)
            with open('checkpoint/frame_counts.json', 'w') as f:
                json.dump(frame_counts, f)
            

        # Choosing 4 random agents each time until the population is exhausted
        population_copy = population.copy()
        population_generation = []
        if checkpoint and generation == 3:
            checkpoint = False

        while len(population_copy) > 0:
            agents = random.sample(population_copy, NUM_AGENTS)
            population_generation.extend(agents)

            # Run episode on the selected agents
            rewards = evaluate_agents(agents, environment, population, checkpoint=checkpoint)

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
                torch.save(critic_network, 'models/best_critic_network.pth')
                
            elif agent.reward > best_rewards[1]:
                print(f"New second best reward found: {agent.reward}")
                best_rewards[1] = agent.reward
                torch.save(agent.policy_network, 'models/second_best_policy_network.pth')
                torch.save(critic_network, 'models/second_best_critic_network.pth')
        
        '''
        for agent in tqdm(population):
            off_policy_svg0(agent, GAMMA, K_STEPS, BATCH_SIZE)
        '''

        # For match results, update Elo ratings
        for i in range(0, len(population_generation), 4):
            # Given 4 agents (1,2,3,4) we have that:
            # Team 1 is always composed of agents 1 and 3
            # Team 2 is always composed of agents 2 and 4
            team1_reward = population_generation[i].reward + population_generation[i+2].reward
            team2_reward = population_generation[i+1].reward + population_generation[i+3].reward
            result = 'win' if team1_reward > team2_reward else 'lose' if team1_reward < team2_reward else 'draw'
            update_elo_rating(population_generation[i], population_generation[i+1], result)
            update_elo_rating(population_generation[i+2], population_generation[i+3], result)

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

                        #agent.critic_network.load_state_dict(agent2.critic_network.state_dict())
                        #agent.critic_network.optimizer.load_state_dict(agent2.critic_network.optimizer.state_dict())

                        agent.target_policy.load_state_dict(agent2.target_policy.state_dict())
                        #agent.target_critc.load_state_dict(agent2.target_critic.state_dict())

                        mutate(agent.policy_network)

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

    population_size = 12 #20
    num_generations = 10000

    use_4 = False

    population = initialize_population(population_size)

    if args.train:
        if args.checkpoint:
            print("Loading checkpoint...")
            if population_size == 1:
                for agent in population:
                    agent.policy_network = torch.load('checkpoint_4_agents/policy_network_0.pth').to(device)
                    agent.critic_network = torch.load('checkpoint_4_agents/critic_network0.pth').to(device)
                    agent.target_policy = torch.load('checkpoint_4_agents/target_policy_network_0.pth').to(device)
                    agent.target_critic = torch.load('checkpoint_4_agents/target_critic_network0.pth').to(device)                   

            else:
                if use_4:
                    for agent in population:
                        num = random.randint(0, 3)
                        agent.policy_network = torch.load(f"good/agent{num}/policy_network.pth").to(device)
                        agent.target_policy = torch.load(f"good/agent{num}/target_policy_network.pth").to(device)
                    critic_network = torch.load('good/agent0/critic_network.pth').to(device)
                    target_critic = torch.load('good/agent0/target_critic_network.pth').to(device)
                
                else:
                    ratings = json.load(open('checkpoint/ratings.json'))
                    frame_counts = json.load(open('checkpoint/frame_counts.json'))
                    for agent in population:
                        agent.policy_network = torch.load(f'checkpoint/agent{population.index(agent)}/policy_network.pth').to(device)
                        agent.target_policy = torch.load(f'checkpoint/agent{population.index(agent)}/target_policy_network.pth').to(device)
                        agent.rating = ratings[population.index(agent)]
                        agent.frames_processed = frame_counts[population.index(agent)]
                    critic_network = torch.load('checkpoint/critic_network.pth').to(device)
                    target_critic = torch.load('checkpoint/target_critic_network.pth').to(device)

        pbt_training(population, environment, num_generations, checkpoint=True)

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
                    agent.policy_network = torch.load('models/best_policy_network.pth')
                else:
                    agent.policy_network = torch.load('models/second_best_policy_network.pth') 

                agent.policy_network.eval()
            rewards = evaluate_agents(population[:NUM_AGENTS], environment, population, eval=True)

        if rewards[0] + rewards[1] > rewards[1] + rewards[3]:
            print("Team Blue won!")
        else:
            print("Team Red won!")

if __name__ == "__main__":
    main()
