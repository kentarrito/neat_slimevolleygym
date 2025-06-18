"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import slimevolleygym

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = False

"""
Example of how to use Gym env, in single or multiplayer setting

Humans can override controls:

blue Agent:
W - Jump
A - Left
D - Right

Yellow Agent:
Up Arrow, Left Arrow, Right Arrow
"""

if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

manualAction = [0, 0, 0] # forward, backward, jump
otherManualAction = [0, 0, 0]
manualMode = False
otherManualMode = False

# taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
def key_press(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.UP:    manualAction[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

    if k == key.D:     otherManualAction[0] = 1
    if k == key.A:     otherManualAction[1] = 1
    if k == key.W:     otherManualAction[2] = 1
    if (k == key.D or k == key.A or k == key.W): otherManualMode = True

def key_release(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.UP:    manualAction[2] = 0
    if k == key.D:     otherManualAction[0] = 0
    if k == key.A:     otherManualAction[1] = 0
    if k == key.W:     otherManualAction[2] = 0

policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

env = gym.make("SlimeVolley-v0")
env.seed(np.random.randint(0, 10000))
#env.seed(689)
env.reset()

if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

obs = env.reset()

steps = 0
total_reward = 0
action = np.array([0, 0, 0])

done = False


import os

import neat
import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


'''
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2
'''

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs = env.reset()
        step=0
        total_reward = 0
        done = False
        while not done:
            step+=1

            #print("obs: ", obs)

            output = net.activate(tuple(obs))  # xi : (value1, ...),  output : (value1, ...)
            net_action = np.array(output)
            net_action = np.round(net_action).astype(int)

            #action = policy.predict(obs)

            obs, reward, done, _ = env.step(net_action)

            square_distance = ((obs[0]-obs[4])**2 + (obs[1]-obs[5])**2)

            total_reward += reward - square_distance

        total_reward = total_reward # if reward = -1, the more step the better. vice versa in the opposite case.

        #env.close()
        #print("cumulative score", total_reward)

        genome.fitness = total_reward

    #print("best_reward: ", best_reward)
    #print("total_net_action: ", total_net_action)
    #print("total_action", total_action)
    
    #env.close()
    #print("cumulative score", total_reward)



def run(config_file):
    global RENDER_MODE, env, obs
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 20)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = {-1: 'x', -2: 'y', -3: 'vx', -4: 'vy', -5: 'bx', -6: 'by', -7: 'bvx', -8: 'bvy', -9: 'ox', -10: 'oy', -11: 'ovx', -12: 'ovy', 0: 'forward', 1:"backward", 2:"jump"}
    #node_names = {}

    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    """
    Example of how to use Gym env, in single or multiplayer setting

    Humans can override controls:

    blue Agent:
    W - Jump
    A - Left
    D - Right

    Yellow Agent:
    Up Arrow, Left Arrow, Right Arrow
    """

    RENDER_MODE = True

    if RENDER_MODE:
        from pyglet.window import key
        from time import sleep

    manualAction = [0, 0, 0] # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 1
        if k == key.RIGHT: manualAction[1] = 1
        if k == key.UP:    manualAction[2] = 1
        if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

        if k == key.D:     otherManualAction[0] = 1
        if k == key.A:     otherManualAction[1] = 1
        if k == key.W:     otherManualAction[2] = 1
        if (k == key.D or k == key.A or k == key.W): otherManualMode = True

    def key_release(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 0
        if k == key.RIGHT: manualAction[1] = 0
        if k == key.UP:    manualAction[2] = 0
        if k == key.D:     otherManualAction[0] = 0
        if k == key.A:     otherManualAction[1] = 0
        if k == key.W:     otherManualAction[2] = 0

    policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))
    #env.seed(689)
    env.reset()

    if RENDER_MODE:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

    obs = env.reset()

    steps = 0
    total_reward = 0
    action = np.array([0, 0, 0])

    done = False

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #[x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
    
    while not done:

        if manualMode: # override with keyboard
            action = manualAction
        else:
            output = winner_net.activate(tuple(obs))  # xi : (value1, ...),  output : (value1, ...)
            #genome.fitness -= (output[0] - xo[0]) ** 2
            net_action = np.array(output)
            net_action = np.round(net_action).astype(int)
            #print("obs: ", obs)
            #print("net_action: ", net_action)
            #action = policy.predict(obs)

        if otherManualMode:
            otherAction = otherManualAction
            obs, reward, done, _ = env.step(net_action, otherAction)
        else:
            #print("obs 1: ", obs)
            obs, reward, done, _ = env.step(net_action)
            #print("obs 2: ", obs)

        if reward > 0 or reward < 0:
            manualMode = False
            otherManualMode = False

        total_reward += reward

        if RENDER_MODE:
            env.render()
            sleep(0.02) # 0.01

    env.close()
    print("cumulative score", total_reward)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 2)

    return winner


if __name__=="__main__":

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    winner = run(config_path)

    