# NEAT

## neat/population.py

Population class/ run : Main loop

## gym for slimevolleygym

How I made it
1. installed latest version of gym (0.26.2)
2. copied gym(==0.19.0)/envs/classic_control/rendering to the correspoinding library
3. added line `from gym.envs.classic_control import rendering` to gym/envs/classic_control/__init__.py
4. installed pyglet==1.5.27


## script description

neat_test3.py: 
normal evolution from reward/step 
cold start problem. It didn’t really evolve meaningfully

neat_test4.py:
add a term from distance between ball and agent
-> it was a bit better, but still evolve really slowly

neat_test5.py:
follow policy first
by setting a proper parameter, it can evolve a bit meaning fully. However, it’s still not as good as it. I think back propagation is needed.