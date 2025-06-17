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

RENDER_MODE = True

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
    global obs
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        output = net.activate(xi)  # xi : (value1, ...),  output : (value1, ...)
        #genome.fitness -= (output[0] - xo[0]) ** 2
        
        if manualMode: # override with keyboard
            action = manualAction
        else:
            action = policy.predict(obs)

        if otherManualMode:
            otherAction = otherManualAction
            obs, reward, done, _ = env.step(action, otherAction)
        else:
            obs, reward, done, _ = env.step(action)

        if reward > 0 or reward < 0:
            manualMode = False
            otherManualMode = False

        total_reward += reward

        genome.fitness -= total_reward

    if RENDER_MODE:
        env.render()
        sleep(0.02) # 0.01
    
    env.close()
    print("cumulative score", total_reward)



def run(config_file):
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
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}

    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__=="__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)