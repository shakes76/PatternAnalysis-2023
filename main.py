from parameters import *
from modules import *
from torchvision import datasets, transforms as T
from train import *


def main():
    train_model()
    ######
    # Example ADaIN
    ######
    # content = torch.randn(1, 4, 4, 128)
    # style = torch.randn(128)
    #
    # adain_layer = ADaIN(128)
    # print(adain_layer(content, style))

    ######
    # Example MappingNetwork
    ######
    # z = torch.randn(1, z_dim)
    #
    # mapping_network = MappingNetwork(z_dim, num_layers, hidden_dim)
    #
    # style = mapping_network(z)
    #
    # print(style)

    ######
    # Example Scaling Factor
    ######
    # noise = torch.randn(1, 4, 4, 8)
    # factor = ScalingFactor(8)
    # scaled_noise = factor(noise)
    # print(noise)
    # print(scaled_noise)

    ######
    # Example Generator Block
    # Creates block with input size 4x4x128
    # Outputs image of size 8x8x64
    ######
    # x = torch.randn(1, 4, 4, 128)
    # noise = torch.randn(1, 64, 8, 8)
    # w_vec = torch.randn(1, 64)
    #
    # generatorBlock = StyleGANGeneratorBlock(128, 64)
    #
    # generated_image = generatorBlock(x, w_vec, noise)
    # print(generated_image.size())

    ######
    # Example Generator network
    ######
    # gen = StyleGANGenerator(128, 128, 4)
    # noise = torch.randn(5, 128, 4, 4)
    # z_vec = torch.randn(5, 1, 1, 128)
    # 
    # result = gen(z_vec)
    # print(result)

    # train_dataset = datasets.CelebA(root, "train", download=True)


if __name__ == "__main__":
    main()
