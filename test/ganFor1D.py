'''
    @Auther: 11768
    @Date: 2023/7/12 10:01
    @Project_Name: Autoformer-main
    @File: ganFor1D.py
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# 定义训练函数
def train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn, num_epochs, batch_size):
    for epoch in range(num_epochs):
        for i in range(len(data) // batch_size):
            # 更新判别器
            discriminator.zero_grad()

            real_data = data[i * batch_size: (i + 1) * batch_size]
            real_label = torch.ones(batch_size, 1)
            real_output = discriminator(real_data)
            real_loss = loss_fn(real_output, real_label)
            real_loss.backward()

            fake_input = torch.randn(batch_size, generator.input_size)
            fake_data = generator(fake_input).detach()
            fake_label = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_data)
            fake_loss = loss_fn(fake_output, fake_label)
            fake_loss.backward()

            discriminator_optimizer.step()

            # 更新生成器
            generator.zero_grad()

            fake_input = torch.randn(batch_size, generator.input_size)
            fake_data = generator(fake_input)
            fake_label = torch.ones(batch_size, 1)
            fake_output = discriminator(fake_data)
            gan_loss = loss_fn(fake_output, fake_label)
            gan_loss.backward()

            generator_optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch: {epoch + 1}, Discriminator Loss: {real_loss.item() + fake_loss.item()}, GAN Loss: {gan_loss.item()}")


# 设置参数
input_size = 1
output_size = 1
hidden_size = 10
num_epochs = 2000
batch_size = 128

# file_path = r"F:\ProjectDir\Autoformer-main\data\ETT\ETTh1.csv"
# df = pd.read_csv(file_path)
# target_ = df['OT']
# df_data_ = df.drop(['OT', 'date'], axis=1, inplace=False)
# data = torch.from_numpy(np.array(target_)).float()
# data_ = torch.from_numpy(np.array(df_data_)).float()
# 准备数据
data = np.random.randn(1000, input_size)
data = torch.from_numpy(data).float()

# 创建生成器和判别器模型实例
generator = Generator(input_size, output_size)
discriminator = Discriminator(output_size)

# 定义损失函数和优化器
loss_fn = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# 训练 GAN 模型
train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn, num_epochs, batch_size)

# 生成一维数据
fake_input = torch.randn(1000, generator.input_size)
# fake_input = torch.randn(len(data), generator.input_size)
fake_data = generator(fake_input).detach().numpy()

# 可视化生成的数据
plt.hist(fake_data, bins=50)
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()
