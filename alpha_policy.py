import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + shortcut)
        return x
            
        
 
class AlphaZeroDraughtsNet(nn.Module):
    
    def __init__(self, board_size = 8, num_res_blocks =5, num_channels = 64):
        super().__init__()
        self.board_size = board_size
        action_space_size = board_size**4
        
        # --- Shared Convolutional Body ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(5,num_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU     ()
        )
        
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_res_blocks)]
        )
        
        #policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.policy_fc = nn.Linear(32*board_size*board_size, action_space_size)
        
        #Value HEad 
        
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels ,4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        self.value_fc = nn.Sequential(
            nn.Linear(4 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # Squeeze output to [-1, 1]
        )
        
        
    def forward(self,x):
        
        x = self.initial_conv(x)
        
        for block in self.res_blocks:
            x = block(x)
            
            
        #policy head forward pass 
        policy_x = self.policy_conv(x)
        policy_x = torch.flatten(policy_x,start_dim=1)
        policy_logits = self.policy_fc(policy_x)
        
        
        #value head . This network try to understand who is winning thats why we reduce the channels form 64 -> 4 in order to force a bottleneck in order to keep the most importand information on who is winnng 
        value_x = self.value_conv(x)
        value_x = torch.flatten(value_x, start_dim=1)
        value_logits = self.value_fc(value_x) 
        
        
        return policy_logits, value_logits
        
if __name__ == '__main__':
    # Example of how to use the network
    net = AlphaZeroDraughtsNet()
    
    # Create a dummy observation batch (batch size = 1)
    dummy_obs = torch.randn(1, 5, 8, 8) 
    
    policy_logits, value = net(dummy_obs)
    
    print("--- Network Output Shapes ---")
    print(f"Policy Logits Shape: {policy_logits.shape}") # Should be [1, 4096]
    print(f"Value Shape: {value.shape}")                 # Should be [1]
    
    # To get probabilities, you would apply softmax
    policy_probs = F.softmax(policy_logits, dim=1)
    print(f"\nPolicy Probabilities Shape: {policy_probs.shape}")
    print(f"Sum of probabilities: {torch.sum(policy_probs):.2f}")