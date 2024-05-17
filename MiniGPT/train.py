"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
if config.to_log:
    wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)

if not Path.exists(config.save_path):
    Path.mkdir(BigramConfig.save_path, parents=True, exist_ok=True)

### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train_model(model, train_dataloader, eval_dataloader, num_epochs):
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
 
        
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if config.to_log and (i + 1) % config.log_interval == 0:
                wandb.log({"train_loss": running_loss / (i + 1)})
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / (i+1):.4f}")

            if config.to_log and (i + 1) % config.save_iterations == 0:
                print(f"Save {config.save_path}/model_epoch{epoch}_step{i+1}.pth")
                torch.save(model.state_dict(), f"{config.save_path}/model_epoch{epoch}_step{i+1}.pth")


            if i >= len(train_dataloader) - 1:
                break


        # # Validation
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for i, data in enumerate(eval_dataloader):
        #         inputs, labels = data
        #         inputs, labels = inputs.to(device), labels.to(device)

        #         outputs = model(inputs)
        #         loss = criterion(outputs.view(-1, config.vocab_size), labels.view(-1))
        #         val_loss += loss.item()
        #         if i >= len(eval_dataloader) - 1:
        #             break

        # val_loss /= len(eval_dataloader)

        # if config.to_log:
        #     wandb.log({"val_loss": val_loss})



# Call the training function
train_model(model, train_dataloader, eval_dataloader,1)
