import random
import torch
import torch.nn as nn
from tqdm import tqdm
import gc

#try soft actor critic
################################ SVG0 ################################
def compute_q_retrace(agent, batch, GAMMA):
    # Extract states, actions, rewards, and next_states from the batch
    states, actions, rewards, next_states = torch.Tensor().to(agent.device), torch.Tensor().to(agent.device), torch.Tensor().to(agent.device), torch.Tensor().to(agent.device)
    for state, action, reward, next_state in batch:
        states = torch.cat((states, state))
        actions = torch.cat((actions, action))
        rewards = torch.cat((rewards, torch.Tensor([reward]).to(agent.device)))
        next_states = torch.cat((next_states, next_state))
    rewards = rewards.unsqueeze(1)
    # Compute Q values for current states and actions
    q_current = agent.target_critic(batch[0][0], batch[0][1])

    # Compute Q values for next states using the target policy
    next_actions = agent.target_policy(next_states)
    q_next = agent.target_critic(next_states, next_actions).detach()

    # Compute Q values for current states and actions
    q_values = agent.target_critic(states, actions).detach()

    gamma_powers = torch.tensor([GAMMA ** t for t in range(len(batch))], device=agent.device).unsqueeze(1)

    # Calculate Q sums
    q_sums = gamma_powers * (rewards +  GAMMA * q_next - q_values)
    q_sums = q_sums.sum()

    # Return the final Q retrace value
    return q_current + q_sums


def train_critic_svg0(agent, index, GAMMA, K_STEPS):

    # Compute Q-retrace targets
    if index + K_STEPS > len(agent.replay_buffer):
        K_STEPS = len(agent.replay_buffer) - index

    q_retrace = compute_q_retrace(agent, agent.replay_buffer[index:index+K_STEPS] , GAMMA)

    q = agent.critic_network(agent.replay_buffer[index][0], agent.replay_buffer[index][1])
    
    critic_loss = nn.MSELoss()(q, q_retrace)

    agent.critic_network.optimizer.zero_grad()
    critic_loss.backward()
    #torch.nn.utils.clip_grad_norm_(agent.critic_network.parameters(), max_norm=1.0)
    agent.critic_network.optimizer.step()
    
    #update_target_network(agent.critic_network, target_network)
    
    return critic_loss

def entropy_loss(logits):
    prob = nn.Softmax(dim=-1)(logits)
    log_prob = nn.LogSoftmax(dim=-1)(logits)
    entropy = -(prob * log_prob).sum(-1).mean()
    return entropy

def train_policy_svg0(agent, index, ENTROPY_COEFF=0.005):
    #policy = copy.deepcopy(agent.policy_network)
    
    #for index in batch_indices:
    state, action, _, _ = agent.replay_buffer[index]
    
    # Infer eta_k using the critic network
    q_value = agent.critic_network(state, agent.policy_network(state))

    # Compute policy loss as negative Q-value
    policy_loss = -q_value.mean()
    
    # Add entropy loss
    
    logits = agent.policy_network(state)
    ent_loss = entropy_loss(logits)
    policy_loss += ENTROPY_COEFF * ent_loss
    

    agent.policy_network.optimizer.zero_grad()
    policy_loss.backward()
    #torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=1.0)
    agent.policy_network.optimizer.step()

    return policy_loss.item()

def off_policy_svg0(agent, GAMMA, K_STEPS, BATCH_SIZE=None):
    
    if BATCH_SIZE is None:
        #sample 1 index from replay buffer
        index = random.sample(range(len(agent.replay_buffer)), 1)
        
        loss = train_critic_svg0(agent, index[0], GAMMA, K_STEPS)
        policy_loss = train_policy_svg0(agent, len(agent.replay_buffer)-1)
    else:
        batch_indices = random.sample(range(len(agent.replay_buffer)), BATCH_SIZE)
        for i,index in enumerate(batch_indices):
            if i % 100 == 0:
                agent.target_policy.load_state_dict(agent.policy_network.state_dict())
                agent.target_critic.load_state_dict(agent.critic_network.state_dict())
            loss = train_critic_svg0(agent, index, GAMMA, K_STEPS)
            policy_loss = train_policy_svg0(agent, index)
        
        agent.target_policy.load_state_dict(agent.policy_network.state_dict())
        agent.target_critic.load_state_dict(agent.critic_network.state_dict())

    #agent.policy_network.scheduler.step()
    #agent.critic_network.scheduler.step()

################################ END SVG0 ################################
STATE_SIZE = (3, 96, 96)
ACTION_SIZE = 3


def update_target_network(target_network, main_network):
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data = (1 - 0.005) * target_param.data + 0.005 * main_param.data

def update(agent, population, GAMMA, BATCH_SIZE=64, critic_network=None, target_critic=None):
    #sample BATCH_SIZE batch from replay buffer
    if len(agent.replay_buffer) < BATCH_SIZE:
        BATCH_SIZE = len(agent.replay_buffer)
    batch_indices = random.sample(range(len(agent.replay_buffer)), BATCH_SIZE)

    state_batch = torch.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1], STATE_SIZE[2]))
    action_batch = torch.zeros((BATCH_SIZE, ACTION_SIZE))
    reward_batch = torch.zeros(BATCH_SIZE)
    next_state_batch = torch.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1], STATE_SIZE[2]))

    for i, index in enumerate(batch_indices):
        state, action, reward, next_state = agent.replay_buffer[index]
        state_batch[i] = state
        action_batch[i] = action
        reward_batch[i] = reward
        next_state_batch[i] = next_state
    state_batch = state_batch.to(agent.device)
    action_batch = action_batch.to(agent.device)
    reward_batch = reward_batch.to(agent.device)
    next_state_batch = next_state_batch.to(agent.device)
    
    reward_batch = reward_batch.unsqueeze(1)

    learn(agent, state_batch, action_batch, reward_batch, next_state_batch, GAMMA, critic_network, target_critic)

    del state_batch, action_batch, reward_batch, next_state_batch
    torch.cuda.empty_cache()


def learn(agent, state, action, reward, next_state, GAMMA, critic_network, target_critic):
    if critic_network is None and target_critic is None:
        with torch.no_grad():
            next_action = agent.target_policy(next_state)
            critic_out = agent.target_critic(next_state, next_action)
            y = reward + GAMMA * critic_out

        q = agent.critic_network(state, action)
        critic_loss = nn.MSELoss()(q, y)

        agent.critic_network.optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_network.optimizer.step()

        critic_out = agent.critic_network(state, agent.policy_network(state))
        policy_loss = -critic_out.mean()

        agent.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        agent.policy_network.optimizer.step()
    
    else:
        with torch.no_grad():
            next_action = agent.target_policy(next_state)
            critic_out = target_critic(next_state, next_action)
            y = reward + GAMMA * critic_out

        q = critic_network(state, action)
        critic_loss = nn.MSELoss()(q, y)

        critic_network.optimizer.zero_grad()
        critic_loss.backward()
        critic_network.optimizer.step()

        critic_out = critic_network(state, agent.policy_network(state))
        policy_loss = -critic_out.mean()

        agent.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        agent.policy_network.optimizer.step()

    update_target_network(target_critic, critic_network)
    update_target_network(agent.target_policy, agent.policy_network)

