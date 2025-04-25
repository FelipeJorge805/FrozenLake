import os
import torch
import cartpole

# Add this before the training loop
checkpoint_path = "cartpole_checkpoint.pth"

# Load if exists
start_episode = 0
EPSILON = 1.0  # fallback if no checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    cartpole.policy_net.load_state_dict(checkpoint['model_state_dict'])
    cartpole.target_net.load_state_dict(checkpoint['model_state_dict'])  # optional: sync target too
    cartpole.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPSILON = checkpoint['epsilon']
    start_episode = checkpoint['episode'] + 1
    print(f"✅ Loaded checkpoint from episode {start_episode}")

# Training loop
EPISODES = 10
highest = 0
for episode in range(start_episode, start_episode + EPISODES):
    state, _ = cartpole.env.reset()
    done = False
    total_reward = 0

    while not done:
        cartpole.env.render()
        action = cartpole.select_action(state, EPSILON)
        next_state, reward, done, _, _ = cartpole.env.step(action)
        cartpole.memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
        cartpole.train()

    EPSILON = max(cartpole.EPSILON_MIN, EPSILON * cartpole.EPSILON_DECAY)

    if episode % cartpole.TARGET_UPDATE == 0:
        cartpole.target_net.load_state_dict(cartpole.policy_net.state_dict())

    if highest < total_reward:
        highest = total_reward
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}, Highest: {highest}")
    else:
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}")

# Save checkpoint after training
torch.save({
    'episode': episode,
    'model_state_dict': cartpole.policy_net.state_dict(),
    'optimizer_state_dict': cartpole.optimizer.state_dict(),
    'epsilon': EPSILON,
}, checkpoint_path)
print(f"💾 Saved checkpoint at episode {episode}")
