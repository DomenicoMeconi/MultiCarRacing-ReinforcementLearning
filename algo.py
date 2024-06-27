import random
import torch
import torch.nn as nn
import copy
#try soft actor critic
################################ SVG0 ################################
def train_critic_svg0(agent, batch, GAMMA):
    q_new = copy.deepcopy(agent.critic_network)
    for i, (state, action, reward, next_state) in enumerate(batch):
       
        target_q = reward + GAMMA * agent.critic_network(next_state, agent.policy_network(next_state)) 
        q = q_new(state,action)

        critic_loss = nn.MSELoss()(q, target_q)

        q_new.optimizer.zero_grad()
        critic_loss.backward()
        q_new.optimizer.step()

        if i % 50 == 0:    
            agent.critic_network.load_state_dict(q_new.state_dict())

    agent.critic_network.load_state_dict(q_new.state_dict())
    #agent.critic_network.scheduler.step()
    

def train_policy_svg0(agent, batch, GAMMA):
    for state, _, _, _ in batch:
        policy_loss = -(agent.critic_network(state, agent.policy_network(state))).mean()

        agent.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        agent.policy_network.optimizer.step()
    #agent.policy_network.scheduler.step()

def off_policy_svg0(agent, BATCH_SIZE, GAMMA):
    if len(agent.replay_buffer) < BATCH_SIZE:
        return
    
    batch = random.sample(agent.replay_buffer, BATCH_SIZE)
    
    train_critic_svg0(agent, batch, GAMMA)
    train_policy_svg0(agent, batch, GAMMA)

################################ END SVG0 ################################

################################   SVG1   ################################
def compute_importance_weight(state, old_policy, new_policy):
    with torch.no_grad():
        old_prob = old_policy(state)
        new_prob = new_policy(state)
    importance_weight = old_prob / (new_prob + 1e-8)
    return importance_weight
    

def train_generative_model(agent, batch):
    for state, action, _, next_state in batch:
        
        agent.generative_model.optimizer.zero_grad()
        agent.generative_model.train()
        pred_next_state = agent.generative_model(state, action)
        loss = nn.MSELoss()(pred_next_state, next_state)
        loss.backward()
        agent.generative_model.optimizer.step()

def train_value_function_svg1(agent, batch, GAMMA):
    v_new = copy.deepcopy(agent.critic_network)
    for i, (state, _, reward, next_state) in enumerate(batch):
       
        y = reward + GAMMA * agent.critic_network(next_state) 
        v = v_new(state)

        critic_loss = 0.5*nn.MSELoss()(v, y)
        v_new.optimizer.zero_grad()
        critic_loss.backward()
        v_new.optimizer.step()

        if i % 50 == 0:    
            agent.critic_network.load_state_dict(v_new.state_dict())
        
    agent.critic_network.load_state_dict(v_new.state_dict())


def train_policy_function_svg1(agent, batch, GAMMA):
    old_policy = copy.deepcopy(agent.policy_network)
    for state, _, reward, next_state in batch:
        
        with torch.no_grad():
            V = reward + GAMMA * agent.critic_network(next_state)
        
        importance_weight = compute_importance_weight(state, old_policy, agent.policy_network)
        policy_loss = -(importance_weight * V).mean() # *agent.policy_network(state)
        print(policy_loss)

        agent.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        agent.policy_network.optimizer.step()

def off_policy_svg1(agent, BATCH_SIZE, GAMMA):
    if len(agent.replay_buffer) < BATCH_SIZE:
        return
    
    batch = random.sample(agent.replay_buffer, BATCH_SIZE)
    
    train_value_function_svg1(agent, batch ,GAMMA)
    train_policy_function_svg1(agent, batch, GAMMA)

################################ END SVG1 ################################