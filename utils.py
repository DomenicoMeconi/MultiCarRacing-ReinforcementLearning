from main import Agent
import random
import torch

def initialize_population(POPULATION_SIZE, LEARNING_RATE_POLICY):
    """
    Initialize a population of agents.

    Args:
        POPULATION_SIZE: The size of the population.
        LEARNING_RATE_POLICY: The learning rate policy to use for each agent.
    Returns:
        A list of agents.
    """

    population = [Agent(LEARNING_RATE_POLICY[i]) for i in range(POPULATION_SIZE)]
    return population

def mutate(network, MUTATION_RATE, MUTATION_SCALE):
    """
    Mutate the network by adding Random noise to its parameters.

    Args:
        network: The network to mutate.
        MUTATION_RATE: The probability of mutating each parameter.
        MUTATION_SCALE: The scale of the noise added to the parameters.
    """
    for param in network.parameters():
        if random.random() < MUTATION_RATE:
            param.data += MUTATION_SCALE * torch.randn_like(param.data)

def update_elo_rating(agent_i, agent_j, result, K=32):
    """
    Update the Elo rating of two agents based on the result of a game.

    Args:
        agent_i: The first agent.
        agent_j: The second agent.
        result: The result of the game. 'win' if agent_i won, 'lose' if agent_i lost, 'draw' if it was a draw.
        K: The K factor used in the Elo rating update formula.
    """
    Ri = agent_i.rating
    Rj = agent_j.rating
    selo = 1 / (1 + 10 ** ((Rj - Ri) / 400))
    s = 1 if result == 'win' else 0 if result == 'lose' else 0.5
    agent_i.rating += K * (s - selo)
    agent_j.rating -= K * (s - selo)

def update_replay_buffer(agent):
    """
    Keep the replay buffer size constant.

    Args:
        agent: The agent to update the replay buffer of.
    """
    if len(agent.replay_buffer) > 4000:
        agent.replay_buffer = agent.replay_buffer[-4000:]

def eligible(agent):
    """
    Check if an agent is eligible for evaluation.

    Args:
        agent: The agent to check.
    Returns:
        True if the agent is eligible for evaluation, False otherwise.
    """
    if agent.eligible:
        return agent.frames_processed > 50000
    else:
        if agent.frames_processed > 200000:
            agent.eligible = True
            return True
        else:   
            return False
        
def replace_color(data, original, new_value):
    """Replace a color in the image with a new value."""
    r1, g1, b1 = original
    r2, g2, b2 = new_value

    red, green, blue = data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:,:3][mask] = [r2, g2, b2]
    return data

def preprocess(img, greyscale=False):
    """
    Preprocess the image.
    
    Args:
        img: The image to preprocess.
        greyscale: Whether to convert the image to greyscale.
    Returns:
        The preprocessed image.
    """
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