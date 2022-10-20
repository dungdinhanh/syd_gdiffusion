from guided_diffusion_hfai.script_util import *


if __name__ == '__main__':
    netdq = create_infoq(True)
    print(netdq)
    # print(netdq.parameters())
    # for param in netdq.parameters():
    #     print(param)
    print(netdq.dnet)
