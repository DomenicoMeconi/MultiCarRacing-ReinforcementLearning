from pyglet.window import key
from main import Agent
import random
import torch

def key_press(k, mod):
    if k == key.SPACE:
        global preview
        preview = True


def key_release(k, mod):
    if k == key.SPACE:
        global preview
        preview = False

def initialize_population(POPULATION_SIZE, LEARNING_RATE_POLICY):
    population = [Agent(LEARNING_RATE_POLICY[i]) for i in range(POPULATION_SIZE)]
    return population

def mutate(network, MUTATION_RATE, MUTATION_SCALE):
    for param in network.parameters():
        if random.random() < MUTATION_RATE:
            param.data += MUTATION_SCALE * torch.randn_like(param.data)

def update_elo_rating(agent_i, agent_j, result, K=32):
    Ri = agent_i.rating
    Rj = agent_j.rating
    selo = 1 / (1 + 10 ** ((Rj - Ri) / 400))
    s = 1 if result == 'win' else 0 if result == 'lose' else 0.5
    agent_i.rating += K * (s - selo)
    agent_j.rating -= K * (s - selo)

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