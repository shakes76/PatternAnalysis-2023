from parameters import *
from modules import *


def main():
    # Example ADaIN
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)

    adain_layer = ADaIN()
    print(adain_layer(content, style))


if __name__ == "__main__":
    main()
